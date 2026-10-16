import json
import shutil
import time
from functools import lru_cache
from pathlib import Path
from datasets import DatasetDict, load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor
)


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