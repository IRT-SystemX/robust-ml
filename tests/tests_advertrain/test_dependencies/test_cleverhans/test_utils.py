import numpy as np
import pytest
import torch

from RobustML.advertrain.dependencies.cleverhans.utils import (clip_eta,
                                                               optimize_linear)


def test_clip_eta_inf_norm():
    eta = torch.tensor([0.5, -0.5, 1.0, -1.0])
    eps = 0.3
    norm = np.inf

    clipped_eta = clip_eta(eta, norm, eps)
    assert torch.all(clipped_eta <= eps) and torch.all(clipped_eta >= -eps)


def test_clip_eta_l2_norm():
    eta = torch.tensor([0.3, 0.4])
    eps = 0.5
    norm = 2

    clipped_eta = clip_eta(eta, norm, eps)
    assert torch.sqrt(torch.sum(clipped_eta ** 2)) <= eps


def test_clip_eta_error_on_invalid_norm():
    eta = torch.tensor([0.5, -0.5])
    eps = 0.3
    norm = 0

    with pytest.raises(ValueError):
        clip_eta(eta, norm, eps)


def test_optimize_linear_inf_norm():
    grad = torch.tensor([0.1, -0.2, 0.3, -0.4])
    eps = 0.5
    norm = np.inf

    optimized_perturbation = optimize_linear(grad, eps, norm)
    assert torch.equal(optimized_perturbation, torch.tensor([0.5, -0.5, 0.5, -0.5]))


def test_optimize_linear_l2_norm():
    grad = torch.tensor([0.3, 0.4])
    eps = 0.5
    norm = 2

    optimized_perturbation = optimize_linear(grad, eps, norm)
    scale = eps / torch.sqrt(torch.sum(grad ** 2))
    expected_perturbation = grad * scale
    assert torch.allclose(optimized_perturbation, expected_perturbation)


def test_optimize_linear_error_on_invalid_norm():
    grad = torch.tensor([0.1, -0.2])
    eps = 0.5
    norm = 0

    with pytest.raises(ValueError):
        optimize_linear(grad, eps, norm)
