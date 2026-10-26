import json
import shutil
import time
from functools import lru_cache
from pathlib import Path
from datasets import DatasetDict, load_dataset, load_from_disk
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
    num_partitions, alpha=None, seed=42, dataset=DEFAULT_DATASET
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
    config_name = f"num-{num_partitions}"

    if alpha is not None:
        config_name += f"_alpha-{alpha:g}"

    config_name += f"_seed-{seed}"
    config_name = f"{dataset.lower()}_{config_name}"

    return (
        Path(__file__).resolve().parents[2]
        / "data"
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
        train_dir = partition_dir / "train"
        if not (partition_dir / "dataset_dict.json").is_file():
            return False
        if not (
            (train_dir / "dataset_info.json").is_file()
            and (train_dir / "state.json").is_file()
            and any(train_dir.glob("data-*.arrow"))
        ):
            return False

    validation_dir = cache_dir / "validation"
    return (
        (validation_dir / "dataset_info.json").is_file()
        and (validation_dir / "state.json").is_file()
        and any(validation_dir.glob("data-*.arrow"))
    )


def _normalize_dataset_columns(dataset, config):
    if config.image_column != "img":
        dataset = dataset.rename_column(config.image_column, "img")
    if config.label_column != "label":
        dataset = dataset.rename_column(config.label_column, "label")
    columns_to_remove = [
        column for column in dataset.column_names if column not in {"img", "label"}
    ]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def _load_source_dataset(config):
    local_dataset_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / config.hub_id.rsplit("/", 1)[-1]
    )
    if local_dataset_path.exists():
        local_dataset = load_from_disk(str(local_dataset_path))
        if isinstance(local_dataset, DatasetDict):
            if "train" not in local_dataset:
                raise ValueError(
                    f"Local dataset at {local_dataset_path} has no 'train' split"
                )
            return local_dataset["train"]
        return local_dataset

    return load_dataset(config.hub_id, split="train")


def prepare_partitions(
    num_partitions, alpha=None, seed=42, dataset=DEFAULT_DATASET
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
        "partitioning_strategy": "global-validation-before-client-partition",
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
            manifest_path.unlink(missing_ok=True)
            source_dataset = _load_source_dataset(config)
            source_dataset = _normalize_dataset_columns(source_dataset, config)
            splits = source_dataset.train_test_split(
                test_size=metadata["validation_fraction"], seed=seed
            )
            client_dataset = splits["train"].shuffle(seed=seed)
            validation_dataset = splits["test"]

            if config.partitioning == "dirichlet":
                partitioner = DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by="label",
                    alpha=alpha,
                    min_partition_size=10,
                    self_balancing=True,
                    seed=seed,
                )

            elif config.partitioning == "IidPartitioner":
                partitioner = IidPartitioner(
                    num_partitions=num_partitions,
                )

            partitioner.dataset = client_dataset

            for partition_id in range(num_partitions):
                partition_dir = cache_dir / f"partition-{partition_id}"

                if partition_dir.exists():
                    shutil.rmtree(partition_dir)

                partition = partitioner.load_partition(partition_id)
                DatasetDict({"train": partition}).save_to_disk(
                    str(partition_dir)
                )

            validation_dir = cache_dir / "validation"
            if validation_dir.exists():
                shutil.rmtree(validation_dir)
            validation_dataset.save_to_disk(str(validation_dir))

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
    alpha=None,
    seed=42,
    dataset=DEFAULT_DATASET,
):
    """
    Carrega os dados de uma partição do disco para memória.

    Parâmetros:
        - partition_non_id: número da partição a ser carregada
        - num_partitions: quantidade de partições
        - batch_size: tamanho do batch de treinamento
        - alpha: valor do alpha de Dirichlet
        - seed: valor da seed

    Retorna:
        - trainloader: dataloader de treinamento da partição
    """

    # Caminho para partição
    cache_dir = prepare_partitions(num_partitions, alpha, seed, dataset)
    partition_dataset = load_from_disk(
        str(cache_dir / f"partition-{partition_non_id}")
    )

    train_dataset = partition_dataset["train"].with_transform(
        lambda batch: apply_transforms(
            batch, _build_transforms(dataset, training=True)
        )
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


@lru_cache(maxsize=1)
def load_centralized_dataset(
    dataset=DEFAULT_DATASET,
    num_partitions=100,
    alpha=None,
    seed=42,
):
    """
    Carrega os dados reservados para validação global e retorna um dataloader.

    Retorna:
        - dataloader dos dados particionado em batches
    """
    cache_dir = prepare_partitions(
        num_partitions, alpha, seed, dataset
    )
    validation_dataset = load_from_disk(str(cache_dir / "validation"))
    transformed_dataset = validation_dataset.with_format("torch").with_transform(
        lambda batch: apply_transforms(
            batch,
            _build_transforms(dataset, training=False),
        )
    )
    return DataLoader(transformed_dataset, batch_size=128)