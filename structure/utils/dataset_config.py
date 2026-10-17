from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DatasetConfig:
    hub_id: str
    image_column: str
    label_column: str
    input_channels: int
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    partitioning: Literal["dirichlet", "natural_id"]


DEFAULT_DATASET = "CIFAR-10"

DATASETS = {
    "CIFAR-10": DatasetConfig(
        hub_id="uoft-cs/cifar10",
        image_column="img",
        label_column="label",
        input_channels=3,
        num_classes=10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        partitioning="dirichlet",
    ),
    "FEMNIST": DatasetConfig(
        hub_id="flwrlabs/femnist",
        image_column="image",
        label_column="character",
        input_channels=1,
        num_classes=62,
        mean=(0.1307,),
        std=(0.3081,),
        partitioning="dirichlet",
    ),
}


def get_dataset_config(dataset: str) -> DatasetConfig:
    try:
        return DATASETS[dataset]
        
    except KeyError as error:
        supported = ", ".join(DATASETS)
        raise ValueError(
            f"Unsupported dataset {dataset!r}. Choose one of: {supported}."
        ) from error
