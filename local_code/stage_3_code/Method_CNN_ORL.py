from cnn_models import CNNBase
import torch.nn as nn

class Method_CNN_ORL(CNNBase):
    def __init__(self):
        super().__init__(
            in_channels=1,
            num_classes=40
        )

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((7, 6))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 40)
        )
