import pytest
import torch

from robustML.advertrain.dependencies.autoattack import (APGDAttack, L0_norm,
                                                         L1_norm,
                                                         L1_projection,
                                                         L2_norm)

torch.manual_seed(0)


@pytest.fixture
def sample_tensor():
    return torch.tensor([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])


def test_L1_norm(sample_tensor):
    assert L1_norm(sample_tensor).equal(torch.tensor([6.0, 6.0]))


def test_L2_norm(sample_tensor):
    assert L2_norm(sample_tensor).equal(L2_norm(sample_tensor))


def test_L0_norm(sample_tensor):
    assert L0_norm(sample_tensor).equal(torch.tensor([3, 3]))


class MockModel(torch.nn.Module):
    def forward(self, x):
        return torch.randn(x.shape[0], 10)
