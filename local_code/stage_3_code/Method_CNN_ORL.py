from cnn_models import CNNBase

class Method_CNN_ORL(CNNBase):
    def __init__(self):
        super().__init__(
            in_channels=1,
            num_classes=40
        )
