import torch
import torch.nn as nn
from structure.utils.dataset_config import (
    DEFAULT_DATASET,
    get_dataset_config,
)


class Net(nn.Module):
    """
    CNN para classificação do CIFAR-10 ou FEMNIST.
    """

    def __init__(self, dataset=DEFAULT_DATASET):
        super().__init__()
        config = get_dataset_config(dataset)

        self.features = nn.Sequential(
            nn.Conv2d(config.input_channels, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Linear(128, config.num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes) -> None:
        super().__init__()

        sizes = [obs_dim, *hidden_sizes, n_actions]
        layers = []

        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(n_in, n_out))

            if i < len(sizes) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)