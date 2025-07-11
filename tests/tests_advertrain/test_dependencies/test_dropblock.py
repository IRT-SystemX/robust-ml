import pytest
import torch

from robustML.advertrain.dependencies.dropblock import (DropBlock2d,
                                                        drop_block_2d,
                                                        drop_block_fast_2d)

torch.manual_seed(0)


@pytest.fixture
def sample_input_tensor():
    return torch.randn(1, 3, 24, 24)


def test_drop_block_2d(sample_input_tensor):
    drop_prob = 0.1
    block_size = 7
    output = drop_block_2d(sample_input_tensor, drop_prob=drop_prob, block_size=block_size)

    assert output.shape == sample_input_tensor.shape


def test_drop_block_fast_2d(sample_input_tensor):
    drop_prob = 0.1
    block_size = 7
    output = drop_block_fast_2d(sample_input_tensor, drop_prob=drop_prob, block_size=block_size)

    assert output.shape == sample_input_tensor.shape


def test_DropBlock2d_initialization():
    drop_block = DropBlock2d(drop_prob=0.1, block_size=7, gamma_scale=1.0)
    assert drop_block.drop_prob == 0.1
    assert drop_block.block_size == 7
    assert drop_block.gamma_scale == 1.0


def test_DropBlock2d_forward(sample_input_tensor):
    drop_block = DropBlock2d(drop_prob=0.1, block_size=7, gamma_scale=1.0)
    drop_block.train()  # Set to training mode

    output_train = drop_block(sample_input_tensor)
    assert output_train.shape == sample_input_tensor.shape
    dropped_elements_train = (sample_input_tensor != output_train).float().mean().item()
    assert dropped_elements_train > 0

    drop_block.eval()
    output_eval = drop_block(sample_input_tensor)
    assert output_eval.shape == sample_input_tensor.shape
    assert torch.equal(output_eval, sample_input_tensor)
