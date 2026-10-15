"""pytorchexample: A Flower / PyTorch app."""

import json
import shutil
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import random
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
)
from collections.abc import Iterable
from flwr.serverapp import Grid

from pytorchexample.state_builder import SimulationEnvironment
from flwr.serverapp.strategy import FedAvg


class Net(nn.Module):
    """Small CIFAR-10 CNN using GroupNorm for federated stability."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


normalize = Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
train_transforms = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), normalize])
eval_transforms = Compose([ToTensor(), normalize])


def apply_transforms(batch, transforms):
    """Apply a split-specific transform to a FederatedDataset batch."""
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def partition_cache_dir(num_partitions, alpha):
    """Return the on-disk cache location for this partition configuration."""
    config_name = f"num-{num_partitions}_alpha-{alpha:g}_seed-42"
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "federated_partitions"
        / config_name
    )


def _partition_cache_files_exist(cache_dir, num_partitions):
    """Check that every persisted split has its metadata and Arrow data."""
    for partition_id in range(num_partitions):
        partition_dir = cache_dir / f"partition-{partition_id}"
        if not (partition_dir / "dataset_dict.json").is_file():
            return False

        for split in ("train", "test"):
            split_dir = partition_dir / split
            if not (
                (split_dir / "dataset_info.json").is_file()
                and (split_dir / "state.json").is_file()
                and any(split_dir.glob("data-*.arrow"))
            ):
                return False

    return True


def prepare_partitions(num_partitions, alpha=0.5):
    """Create and persist every train/validation partition for reuse."""
    cache_dir = partition_cache_dir(num_partitions, alpha)
    manifest_path = cache_dir / "manifest.json"
    lock_dir = cache_dir / ".prepare.lock"
    metadata = {
        "dataset": "uoft-cs/cifar10",
        "num_partitions": num_partitions,
        "alpha": alpha,
        "seed": 42,
        "validation_fraction": 0.2,
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_acquired = False
    for _ in range(1800):
        if manifest_path.is_file() and _partition_cache_files_exist(
            cache_dir, num_partitions
        ):
            with manifest_path.open(encoding="utf-8") as manifest_file:
                saved_metadata = json.load(manifest_file)
            if saved_metadata != metadata:
                raise ValueError(f"Partition metadata does not match cache: {cache_dir}")
            return cache_dir

        try:
            lock_dir.mkdir()
            lock_acquired = True
            break
        except FileExistsError:
            time.sleep(1)

    if not lock_acquired:
        if manifest_path.is_file() and _partition_cache_files_exist(
            cache_dir, num_partitions
        ):
            with manifest_path.open(encoding="utf-8") as manifest_file:
                saved_metadata = json.load(manifest_file)
            if saved_metadata != metadata:
                raise ValueError(f"Partition metadata does not match cache: {cache_dir}")
            return cache_dir
        raise TimeoutError(
            f"Timed out waiting for partition cache creation at {cache_dir}"
        )

    try:
        cache_is_complete = manifest_path.is_file() and _partition_cache_files_exist(
            cache_dir, num_partitions
        )
        if cache_is_complete:
            with manifest_path.open(encoding="utf-8") as manifest_file:
                saved_metadata = json.load(manifest_file)
            if saved_metadata != metadata:
                raise ValueError(f"Partition metadata does not match cache: {cache_dir}")
        else:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha,
                min_partition_size=10,
                self_balancing=True,
                seed=42,
            )
            federated_dataset = FederatedDataset(
                dataset=metadata["dataset"],
                partitioners={"train": partitioner},
            )

            for partition_id in range(num_partitions):
                partition_dir = cache_dir / f"partition-{partition_id}"
                if partition_dir.exists():
                    shutil.rmtree(partition_dir)
                partition = federated_dataset.load_partition(partition_id)
                splits = partition.train_test_split(test_size=0.2, seed=42)
                DatasetDict(
                    {"train": splits["train"], "test": splits["test"]}
                ).save_to_disk(str(partition_dir))

            manifest_tmp = cache_dir / "manifest.json.tmp"
            with manifest_tmp.open("w", encoding="utf-8") as manifest_file:
                json.dump(metadata, manifest_file, indent=2)
            manifest_tmp.replace(manifest_path)
    finally:
        shutil.rmtree(lock_dir)

    return cache_dir


def load_data(partition_non_id, num_partitions, batch_size, alpha=0.5):
    """Load one persisted Dirichlet partition of CIFAR10."""
    cache_dir = prepare_partitions(num_partitions, alpha)

    partition_train_test = load_from_disk(
        str(cache_dir / f"partition-{partition_non_id}")
    )

    # Construct dataloaders
    train_dataset = partition_train_test["train"].with_transform(
        lambda batch: apply_transforms(batch, train_transforms)
    )
    test_dataset = partition_train_test["test"].with_transform(
        lambda batch: apply_transforms(batch, eval_transforms)
    )
    
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader


@lru_cache(maxsize=1)
def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(
        lambda batch: apply_transforms(batch, eval_transforms)
    )
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    net.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_trainloss = running_loss / (epochs * len(trainloader))
    train_accuracy = correct / total

    return avg_trainloss, train_accuracy


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Garante comportamento determinístico nos algoritmos do PyTorch/cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _divergence_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Calcula a divergência KL entre duas distribuições de probabilidade p e q.

    Args:
        p (np.ndarray): Distribuição de probabilidade p.
        q (np.ndarray): Distribuição de probabilidade q.

    Returns:
        float: Divergência KL D_KL(p || q).
    """
    # Adiciona um pequeno valor para evitar log(0)
    epsilon = 1e-10
    p_safe = np.clip(p, epsilon, 1)
    q_safe = np.clip(q, epsilon, 1)

    return np.sum(p_safe * np.log(p_safe / q_safe))


def _compute_class_proportions(labels, num_classes: int = 10) -> np.ndarray:
    """Calcula a proporção P_i(c) de cada classe c em uma partição de dados do
    cliente i.

    Parametros:
        labels: Lista, np.ndarray ou torch.Tensor contendo os rótulos do
        cliente.
        num_classes: Número total de classes do dataset (padrão: 10 para
        CIFAR-10).

    Retorna:
        np.ndarray de tamanho (num_classes,) com as proporções normalizadas
        P_i.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    elif isinstance(labels, list):
        labels = np.array(labels)

    # Caso a partição esteja vazia
    if len(labels) == 0:
        return np.full(num_classes, 1.0 / num_classes, dtype=np.float32)

    # Conta a frequência de cada classe de 0 até num_classes - 1
    counts = np.bincount(labels, minlength=num_classes)

    # Converte para frequências relativas (proporções)
    proportions = counts / len(labels)

    return proportions.astype(np.float32)


def _compute_proportions_for_clients(client_labels: list[np.ndarray], num_classes: int = 10) -> np.ndarray:
    """Calcula as proporções P_i(c) para cada cliente i.

    Args:
        client_labels (list[np.ndarray]): Lista de rótulos de cada cliente.
        num_classes (int): Número total de classes do dataset.

    Returns:
        np.ndarray: Matriz de proporções P_i(c) para cada cliente.
    """
    proportions = np.array(
        [_compute_class_proportions(labels, num_classes) for labels in client_labels],
        dtype=np.float32,
    )
    return proportions


def compute_divergence_kl_for_clients(
    client_labels: list[np.ndarray], global_labels: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    """Calcula a divergência KL D_KL(P_i || P_global) para cada cliente i.

    Args:
        client_labels (list[np.ndarray]): Lista de rótulos de cada cliente.
        global_labels (np.ndarray): Rótulos globais.
        num_classes (int): Número total de classes do dataset.

    Returns:
        np.ndarray: Vetor de divergências KL para cada cliente.
    """

    global_proportions = _compute_class_proportions(global_labels, num_classes)
    client_proportions = _compute_proportions_for_clients(client_labels, num_classes)

    divergences = np.array(
        [
            _divergence_kl(client_prop, global_proportions)
            for client_prop in client_proportions
        ],
        dtype=np.float32,
    )
    return divergences


def _generate_bandwidths(
    num_clients: int,
    min_mbps: float = 1.0,
    max_mbps: float = 50.0,
    seed: int = 42,
) -> np.ndarray:
    """Gera a banda larga B_i para N clientes heterogêneos.

    Retorna os valores convertidos para Bytes/segundo (B/s).
    """
    np.random.seed(seed)
    # Sorteia a banda em Megabits por segundo (Mbps)
    bandwidth_mbps = np.random.uniform(min_mbps, max_mbps, size=num_clients)

    # Converte Mbps -> Bytes por segundo: (Mbps * 1.000.000) / 8
    bandwidth_bytes_per_sec = (bandwidth_mbps * 1e6) / 8.0

    return bandwidth_bytes_per_sec.astype(np.float32)


def _generate_latencies(
    num_clients: int,
    min_ms: float = 10.0,
    max_ms: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    """Gera a latência de rede delta_i para N clientes.

    Retorna os valores convertidos para segundos.
    """
    np.random.seed(seed + 100)  # Semente distinta para não alinhar com a banda
    # Sorteia a latência em milissegundos (ms)
    latencies_ms = np.random.uniform(min_ms, max_ms, size=num_clients)

    # Converte ms -> segundos
    latencies_sec = latencies_ms / 1000.0

    return latencies_sec.astype(np.float32)


def compute_model_size_bytes(model: torch.nn.Module) -> int:
    """Calcula o tamanho total em bytes do modelo neural PyTorch (pesos +
    biases).
    """
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _compute_communication_costs(
    model_size_bytes: float,
    bandwidths_bytes_per_sec: np.ndarray,
    latencies_sec: np.ndarray,
) -> np.ndarray:
    """Calcula o custo c_i (em segundos) para cada cliente conforme a fórmula

    do MDP:

    c_i = (|theta| / B_i) + delta_i
    """
    costs_sec =  (model_size_bytes / bandwidths_bytes_per_sec) + latencies_sec
    return costs_sec.astype(np.float32)


def generate_client_communication_costs(
    model: torch.nn.Module,
    num_clients: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Função wrapper para gerar diretamente o vetor c_i de tamanho

    (num_clients,)
    """
    model_bytes = compute_model_size_bytes(model)
    bandwidths = _generate_bandwidths(num_clients, seed=seed)
    latencies = _generate_latencies(num_clients, seed=seed)

    costs = _compute_communication_costs(model_bytes, bandwidths, latencies)
    return costs


def _count_message_record_bytes(messages: Iterable[Message]) -> int:
    """Count Flower record payloads, excluding message and transport overhead."""
    return sum(
        record.count_bytes()
        for message in messages
        if not message.has_error()
        for record in message.content.values()
    )


class SimulationFedAvg(FedAvg):
    """FedAvg that enforces the simulation environment's client availability."""

    def __init__(self, *args, simulation_environment: SimulationEnvironment, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_environment = simulation_environment

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        config["eligible-clients-mask"] = (
            self.simulation_environment.eligible_clients_mask.astype(int).tolist()
        )
        messages = list(super().configure_train(server_round, arrays, config, grid))
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(messages)
        )
        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(replies)
        )
        eligible_replies = [
            reply
            for reply in replies
            if not reply.has_error() and bool(reply.content["metrics"]["eligible"])
        ]
        return super().aggregate_train(server_round, eligible_replies)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        messages = list(
            super().configure_evaluate(server_round, arrays, config, grid)
        )
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(messages)
        )
        return messages

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        replies = list(replies)
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(replies)
        )
        return super().aggregate_evaluate(server_round, replies)


