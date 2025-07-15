import pytest
import torch

from robustML.advertrain.dependencies.cleverhans.fast_gradient_method import \
    fast_gradient_method

torch.manual_seed(0)


def mock_model_fn(x):
    return x


def test_adversarial_example_generation():
    x = torch.randn(3, 3)
    eps = 0.1
    norm = 2

    adv_x = fast_gradient_method(mock_model_fn, x, eps, norm)

    assert adv_x.shape == x.shape
    assert torch.max(torch.abs(adv_x - x)) <= eps


def test_error_on_negative_eps():
    x = torch.randn(3, 3)
    eps = -0.1
    norm = 2

    with pytest.raises(ValueError):
        fast_gradient_method(mock_model_fn, x, eps, norm)
