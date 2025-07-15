import pytest
import torch
from torch.nn import Linear, Module
from torch.optim import SGD

from robustML.advertrain.training.adversarial_training import \
    AdversarialTraining


# Define a simple model with trainable parameters
class SimpleModel(Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = Linear(in_features=10, out_features=2)

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
    return torch.device("cpu")


@pytest.fixture
def mock_epsilon():
    return 0.1


def test_initialization(mock_model, mock_optimizer, mock_loss_func, mock_device, mock_epsilon):
    training = AdversarialTraining(mock_model, mock_optimizer, mock_loss_func, mock_device, mock_epsilon)
    assert training.model is mock_model
    assert training.optimizer is mock_optimizer
    assert training.loss_func is mock_loss_func
    assert training.device == mock_device
    assert training.epsilon == mock_epsilon