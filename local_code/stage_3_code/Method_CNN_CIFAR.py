if __package__:
    from .cnn_models import CNNBase
else:
    from cnn_models import CNNBase
import torch.nn as nn


class Method_CNN_CIFAR(CNNBase):
    """Three-block CIFAR-10 CNN with configurable spatial pooling and hidden width."""

    def __init__(self, hidden_dim=128, pooling="max"):
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if pooling not in ("max", "avg"):
            raise ValueError("pooling must be 'max' or 'avg'")

        super().__init__(in_channels=3, num_classes=10)

        pool_layer = nn.MaxPool2d if pooling == "max" else nn.AvgPool2d
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            pool_layer(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            pool_layer(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            pool_layer(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 10),
        )
