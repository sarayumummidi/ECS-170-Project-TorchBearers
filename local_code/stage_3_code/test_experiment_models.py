"""Small CPU tests; run directly or with python -m unittest using this module."""

import unittest

import torch
from torch import nn

if __package__:
    from .cnn_models import CNNBase
    from .Method_CNN_CIFAR import Method_CNN_CIFAR
    from .Method_CNN_ORL import Method_CNN_ORL
else:
    from cnn_models import CNNBase
    from Method_CNN_CIFAR import Method_CNN_CIFAR
    from Method_CNN_ORL import Method_CNN_ORL


MODEL_CASES = (
    (Method_CNN_ORL, (1, 1, 112, 92), 40, 256, 128, 1_479_912),
    (Method_CNN_CIFAR, (1, 3, 32, 32), 10, 128, 256, 357_258),
)


def legacy_model(in_channels, num_classes, hidden_dim):
    """Reconstruct the original layers, including the discarded base initialization."""
    model = CNNBase(in_channels, num_classes)
    model.features = nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1),
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
    )
    if in_channels == 1:
        model.features.append(nn.AdaptiveAvgPool2d((7, 6)))
        flattened_dim = 128 * 7 * 6
    else:
        flattened_dim = 128 * 4 * 4
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flattened_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, num_classes),
    )
    return model


class ExperimentModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

    def check_model(self, model, input_shape, num_classes, hidden_dim, pool_type):
        self.assertEqual(model.classifier[1].out_features, hidden_dim)
        self.assertEqual(model.classifier[4].in_features, hidden_dim)
        self.assertIsInstance(model.classifier[-1], nn.Linear)
        spatial_pools = [
            layer
            for layer in model.features
            if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d))
        ]
        self.assertEqual(len(spatial_pools), 3)
        for pool in spatial_pools:
            self.assertIs(type(pool), pool_type)
            self.assertEqual(pool.kernel_size, 2)
        if input_shape[1] == 1:
            self.assertIsInstance(model.features[-1], nn.AdaptiveAvgPool2d)
            self.assertEqual(model.features[-1].output_size, (7, 6))

        model.eval()
        with torch.inference_mode():
            logits = model(torch.zeros(input_shape))
        self.assertEqual(tuple(logits.shape), (1, num_classes))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_defaults(self):
        for model_type, shape, classes, width, _, parameters in MODEL_CASES:
            with self.subTest(model=model_type.__name__):
                model = model_type()
                self.check_model(model, shape, classes, width, nn.MaxPool2d)
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)

    def test_hidden_width_and_pooling_variants(self):
        for model_type, shape, classes, default_width, variant_width, _ in MODEL_CASES:
            for width in (default_width, variant_width):
                for pooling, pool_type in (
                    ("max", nn.MaxPool2d),
                    ("avg", nn.AvgPool2d),
                ):
                    with self.subTest(
                        model=model_type.__name__, width=width, pooling=pooling
                    ):
                        model = model_type(hidden_dim=width, pooling=pooling)
                        self.check_model(model, shape, classes, width, pool_type)

    def test_default_state_dict_and_initialization_compatibility(self):
        for model_type, shape, classes, width, _, _ in MODEL_CASES:
            with (
                self.subTest(model=model_type.__name__),
                torch.random.fork_rng(devices=[]),
            ):
                torch.manual_seed(170)
                expected = legacy_model(shape[1], classes, width)
                torch.manual_seed(170)
                actual = model_type()
                expected_state = expected.state_dict()
                actual_state = actual.state_dict()
                self.assertEqual(list(actual_state), list(expected_state))
                for name, tensor in expected_state.items():
                    self.assertTrue(torch.equal(actual_state[name], tensor), name)
                actual.load_state_dict(expected_state, strict=True)
                expected.eval()
                actual.eval()
                with torch.inference_mode():
                    inputs = torch.zeros(shape)
                    self.assertTrue(torch.equal(actual(inputs), expected(inputs)))

    def test_invalid_hidden_dim(self):
        for model_type, *_ in MODEL_CASES:
            for hidden_dim in (0, -1, 1.5, "128", None, True, False):
                with self.subTest(model=model_type.__name__, hidden_dim=hidden_dim):
                    with self.assertRaisesRegex(
                        ValueError, "hidden_dim must be a positive integer"
                    ):
                        model_type(hidden_dim=hidden_dim)

    def test_invalid_pooling(self):
        for model_type, *_ in MODEL_CASES:
            for pooling in ("", "MAX", "median", None, 2, []):
                with self.subTest(model=model_type.__name__, pooling=pooling):
                    with self.assertRaisesRegex(
                        ValueError, "pooling must be 'max' or 'avg'"
                    ):
                        model_type(pooling=pooling)


if __name__ == "__main__":
    unittest.main()
