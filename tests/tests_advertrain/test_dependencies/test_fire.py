import pytest
import torch

from RobustML.advertrain.dependencies.fire import (entropy_loss, fire_loss,
                                                   noise_loss)

torch.manual_seed(0)


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.lin(x)


@pytest.fixture
def mock_model():
    return MockModel()


def test_entropy_loss():
    logits = torch.randn(32, 2)
    loss = entropy_loss(logits)

    assert loss.item() >= 0


def test_fire_loss(mock_model):
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
    device = torch.device('cpu')

    total_loss, nat_loss, rob_loss, ent_loss = fire_loss(
        model=mock_model,
        x_natural=x,
        y=y,
        optimizer=optimizer,
        epoch=1,
        device=device
    )

    assert total_loss.item() >= 0
    assert nat_loss.item() >= 0
    assert rob_loss.item() >= 0
    assert ent_loss.item() >= 0


def test_noise_loss(mock_model):
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    loss = noise_loss(
        model=mock_model,
        x_natural=x,
        y=y
    )

    assert loss.item() >= 0
