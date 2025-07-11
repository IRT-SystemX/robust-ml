import pytest
import torch
from torch.nn import Linear, Module
from torch.optim import SGD

from robustML.advertrain.dependencies.autoattack import APGDAttack
from robustML.advertrain.training.autoattack_training import AutoAttackTraining


class SimpleModel(Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def simple_optimizer(simple_model):
    return SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def simple_loss_func():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def simple_device():
    return torch.device('cpu')


def test_auto_attack_training_initialization(simple_model, simple_optimizer, simple_loss_func, simple_device):
    training = AutoAttackTraining(simple_model, simple_optimizer, simple_loss_func, simple_device, 'ce', 0.1)

    assert training.epsilon == 0.1
    assert training.apgd_loss == 'ce'
    assert isinstance(training.apgd, APGDAttack)
