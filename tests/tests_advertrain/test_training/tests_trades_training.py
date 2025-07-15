import torch
import pytest
from robustML.advertrain.training.trades_training import TRADESTraining

torch.manual_seed(0)


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.lin(x)


class MockOptimizer(torch.optim.Optimizer):
    # Mock optimizer for testing
    pass


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_optimizer(mock_model):
    return MockOptimizer(mock_model.parameters(), {})


@pytest.fixture
def mock_device():
    return torch.device('cpu')


@pytest.fixture
def trades_training(mock_model, mock_optimizer, mock_device):
    return TRADESTraining(
        model=mock_model,
        optimizer=mock_optimizer,
        device=mock_device,
        epsilon=0.1,
        beta=1.0,
        perturb_steps=20
    )


def test_val_batch(trades_training):
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    loss, batch_size = trades_training.val_batch(x, y, epoch=1)

    assert loss >= 0
    assert batch_size == x.size(0)
