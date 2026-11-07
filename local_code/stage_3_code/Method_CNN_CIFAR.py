from cnn_models import CNNBase

class Method_CNN_CIFAR(CNNBase):
    def __init__(self):
        super().__init__(
            in_channels=3,
            num_classes=10
        )
