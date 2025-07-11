import pytest
import torch
from torch.nn import Linear, Module
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from RobustML.advertrain.training.classical_training import ClassicalTraining

torch.manual_seed(0)


class SimpleModel(Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = Linear(in_features=5, out_features=2)  # Example layer

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def mock_model():
    return SimpleModel()


@pytest.fixture
def mock_optimizer(mock_model):
    return SGD(mock_model.parameters(), lr=0.001)


@pytest.fixture
def mock_loss_func():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def mock_device():
    return torch.device("cpu")  # or "cuda" if testing on GPU


@pytest.fixture
def mock_dataloader():
    x = torch.rand(10, 5)  # example data
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=2)


def test_initialization(mock_model, mock_optimizer, mock_loss_func, mock_device):
    training = ClassicalTraining(mock_model, mock_optimizer, mock_loss_func, mock_device)
    assert training.model is mock_model
    assert training.optimizer is mock_optimizer
    assert training.loss_func is mock_loss_func
    assert training.device == mock_device