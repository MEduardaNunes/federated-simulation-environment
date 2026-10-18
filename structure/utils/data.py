import json
import shutil
import time
from functools import lru_cache
from pathlib import Path
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor
)
from structure.utils.dataset_config import (
    DEFAULT_DATASET,
    get_dataset_config,
)

def _build_transforms(dataset, training):
    config = get_dataset_config(dataset)
    transforms = []
    if training and dataset == "CIFAR-10":
        transforms.extend([RandomCrop(32, padding=4), RandomHorizontalFlip()])
    transforms.extend([ToTensor(), Normalize(config.mean, config.std)])
    return Compose(transforms)


def apply_transforms(batch, transforms):
    """
    Aplica um transformer no batch de imagens.

    Parâmetros:
        - batch: conjunto de imagens
        - transforms: transformer que controla a transformação da imagem

    Retorna:
        - batch: novo conjunto de imagens com transformações aplicadas
    """
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def partition_cache_dir(
    num_partitions, alpha, seed=42, dataset=DEFAULT_DATASET
):
    """
    Retorna o cache em disco do dataset.

    Parâmetros:
        - num_partictions: quantidade de partições
        - alpha: valor de alfa de dirichlet
        - seed: valor da seed

    Retorna:
        - O cache em disco do dataset
    """
    config = get_dataset_config(dataset)
    config_name = f"num-{num_partitions}_alpha-{alpha:g}_seed-{seed}"
    if config.hub_id != get_dataset_config(DEFAULT_DATASET).hub_id:
        config_name = f"{dataset.lower()}_{config_name}"

    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "federated_partitions"
        / config_name
    )


def _partition_cache_files_exist(cache_dir, num_partitions):
    """
    Verifica se cada partição do dataset têm seus metadados e dados.

    Parâmetros:
        - cache_dir: caminho para diretório com as partições
        - num_partitions: quantidade de partições do dataset

    Retorna:
        - True se todas partições existem e têm seus metadados e dados,
        False caso contrário
    """
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


def prepare_partitions(
    num_partitions, alpha=0.5, seed=42, dataset=DEFAULT_DATASET
):
    """
    Cria as partições federadas do dataset e salva em diretório.

    Parâmetros:
        - num_partitions: número de partições
        - alpha: valor alfa para divisão com dirichlet
        - dataset: dataset a ser particionado

    Retorna:
        - cache_dir: diretório das partições do dataset
    """

    config = get_dataset_config(dataset)
    cache_dir = partition_cache_dir(num_partitions, alpha, seed, dataset)
    manifest_path = cache_dir / "manifest.json"
    lock_dir = cache_dir / ".prepare.lock"
    metadata = {
        "dataset": config.hub_id,
        "num_partitions": num_partitions,
        "alpha": alpha,
        "seed": seed,
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
            if config.partitioning == "dirichlet":
                partitioner = DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by=config.label_column,
                    alpha=alpha,
                    min_partition_size=10,
                    self_balancing=True,
                    seed=seed,
                )

            elif config.partitioning == "IidPartitioner":
                partitioner = IidPartitioner(
                    num_partitions=num_partitions,
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
                if config.image_column != "img":
                    partition = partition.rename_column(
                        config.image_column, "img"
                    )
                if config.label_column != "label":
                    partition = partition.rename_column(
                        config.label_column, "label"
                    )
                columns_to_remove = [
                    column
                    for column in partition.column_names
                    if column not in {"img", "label"}
                ]
                if columns_to_remove:
                    partition = partition.remove_columns(columns_to_remove)
                splits = partition.train_test_split(
                    test_size=0.2, seed=seed
                )
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


def load_data(
    partition_non_id,
    num_partitions,
    batch_size,
    alpha=0.5,
    seed=42,
    dataset=DEFAULT_DATASET,
):
    """
    Carrega os dados de uma partição do disco para memória.

    Parâmetros:
        - partition_non_id: número da partição a ser carregada
        - num_partitions: quantidade de partições
        - batch_size: tamanho do batch de treinamento e teste
        - alpha: valor do alpha de Dirichlet
        - seed: valor da seed

    Retorna:
        - trainloader: dataloader de treinamento
        - testloader: dataloader de teste
    """

    # Caminho para partição
    cache_dir = prepare_partitions(num_partitions, alpha, seed, dataset)
    partition_train_test = load_from_disk(
        str(cache_dir / f"partition-{partition_non_id}")
    )

    # Construção dos datasets utilizando transformers
    train_dataset = partition_train_test["train"].with_transform(
        lambda batch: apply_transforms(
            batch, _build_transforms(dataset, training=True)
        )
    )
    test_dataset = partition_train_test["test"].with_transform(
        lambda batch: apply_transforms(
            batch, _build_transforms(dataset, training=False)
        )
    )
    
    # Dataloader de treinamento e teste
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader


@lru_cache(maxsize=1)
def load_centralized_dataset(
    dataset=DEFAULT_DATASET,
    num_partitions=100,
    alpha=0.5,
    seed=42,
):
    """
    Carrega os dados de avaliação global e retorna um dataloader.

    Retorna:
        - dataloader dos dados particionado em batches
    """
    dataset_name = dataset
    if dataset_name == DEFAULT_DATASET:
        test_dataset = load_dataset(
            get_dataset_config(dataset_name).hub_id, split="test"
        )
    else:
        cache_dir = prepare_partitions(
            num_partitions, alpha, seed, dataset_name
        )
        test_datasets = [
            load_from_disk(str(cache_dir / f"partition-{partition_id}"))[
                "test"
            ]
            for partition_id in range(num_partitions)
        ]
        test_dataset = concatenate_datasets(test_datasets)

    transformed_dataset = test_dataset.with_format("torch").with_transform(
        lambda batch: apply_transforms(
            batch,
            _build_transforms(dataset_name, training=False),
        )
    )
    return DataLoader(transformed_dataset, batch_size=128)