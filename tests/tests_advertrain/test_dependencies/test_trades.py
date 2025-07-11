import torch
import pytest
from robustML.advertrain.dependencies.trades import squared_l2_norm, l2_norm, trades_loss

torch.manual_seed(0)


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 2)  # Adjust dimensions as needed

    def forward(self, x):
        return self.lin(x)


@pytest.fixture
def mock_model():
    return MockModel()


def test_squared_l2_norm():
    x = torch.randn(32, 10)
    norm = squared_l2_norm(x)

    assert torch.all(norm >= 0)


def test_l2_norm():
    x = torch.randn(32, 10)
    norm = l2_norm(x)

    assert torch.all(norm >= 0)


def test_trades_loss(mock_model):
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
    device = torch.device('cpu')

    loss = trades_loss(
        model=mock_model,
        x_natural=x,
        y=y,
        optimizer=optimizer,
        device=device
    )

    assert loss.item() >= 0
