import numpy as np
import pytest
import torch

from RobustML.advertrain.dependencies.cleverhans.projected_gradient_descent import \
    projected_gradient_descent

torch.manual_seed(0)


def mock_model_fn(x):
    return x


def test_adversarial_example_generation():
    x = torch.randn(3, 3)
    eps = 0.1
    eps_iter = 0.01
    nb_iter = 10
    norm = np.inf

    adv_x = projected_gradient_descent(mock_model_fn, x, eps, eps_iter, nb_iter, norm)

    assert adv_x.shape == x.shape


def test_error_on_invalid_eps():
    x = torch.randn(3, 3)
    eps = -0.1
    eps_iter = 0.01
    nb_iter = 10
    norm = np.inf

    with pytest.raises(ValueError):
        projected_gradient_descent(mock_model_fn, x, eps, eps_iter, nb_iter, norm)


def test_error_on_invalid_eps_iter():
    x = torch.randn(3, 3)
    eps = 0.1
    eps_iter = 0.2
    nb_iter = 10
    norm = np.inf

    with pytest.raises(ValueError):
        projected_gradient_descent(mock_model_fn, x, eps, eps_iter, nb_iter, norm)


def test_error_on_invalid_clip_values():
    x = torch.randn(3, 3)
    eps = 0.1
    eps_iter = 0.01
    nb_iter = 10
    norm = np.inf
    clip_min = 1
    clip_max = 0

    with pytest.raises(ValueError):
        projected_gradient_descent(mock_model_fn, x, eps, eps_iter, nb_iter, norm, clip_min, clip_max)
