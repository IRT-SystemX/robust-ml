import os

os.getcwd()

import pytest
import torch

from RobustML.advertrain.metrics import Metrics


@pytest.fixture
def sample_data():
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([1, 0])
    pred = torch.tensor([1, 0])
    loss = torch.tensor([0.5])
    return x, y, pred, loss


def test_initial_state():
    metrics = Metrics()
    assert metrics.TP == 0
    assert metrics.TN == 0
    assert metrics.FP == 0
    assert metrics.FN == 0
    assert metrics.loss == 0.0


def test_metrics_update(sample_data):
    x, y, pred, loss = sample_data
    metrics = Metrics()
    metrics.update(x, y, pred, loss)

    assert metrics.TP == 1
    assert metrics.TN == 1
    assert metrics.FP == 0
    assert metrics.FN == 0
    assert metrics.loss == loss.item() * len(x)


def test_get_metrics(sample_data):
    x, y, pred, loss = sample_data
    metrics = Metrics()
    metrics.update(x, y, pred, loss)

    accuracy, loss, precision, recall, f1_score = metrics.get_metrics()

    assert accuracy == (1 + 1) / (2 + 1e-8)
    assert precision == 1 / (1 + 1e-8)
