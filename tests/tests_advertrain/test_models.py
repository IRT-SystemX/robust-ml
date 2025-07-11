import torch
import torch.nn as nn

from RobustML.advertrain.models import ConvNet, ResNet

torch.manual_seed(0)


def test_ConvNet_initialization():
    device = torch.device("cpu")
    model = ConvNet(device)

    assert isinstance(model.conv1, nn.Conv2d)
    assert isinstance(model.conv2_1, nn.Conv2d)
    assert isinstance(model.conv3_1, nn.Conv2d)
    assert isinstance(model.conv4_1, nn.Conv2d)
    assert isinstance(model.pooling, nn.MaxPool2d)
    assert isinstance(model.activation, nn.ReLU)
    assert isinstance(model.linear1, nn.Linear)
    assert isinstance(model.linear2, nn.Linear)
    assert isinstance(model.linear3, nn.Linear)


def test_ConvNet_forward_pass():
    device = torch.device("cpu")
    model = ConvNet(device)

    dummy_input = torch.randn(1, 3, 64, 128, device=device)

    output = model(dummy_input)

    assert output.shape == torch.Size([1, 2])


def test_ResNet_initialization():
    device = torch.device("cpu")
    model = ResNet(device)

    assert isinstance(model.conv1, nn.Conv2d)
    assert isinstance(model.conv1_bn, nn.BatchNorm2d)
    assert isinstance(model.conv2, nn.Conv2d)
    assert isinstance(model.conv2_bn, nn.BatchNorm2d)
    assert isinstance(model.conv3, nn.Conv2d)
    assert isinstance(model.conv3_drop, nn.Dropout2d)
    assert isinstance(model.conv3_bn, nn.BatchNorm2d)

    assert isinstance(model.conv4, nn.Conv2d)
    assert isinstance(model.conv4_bn, nn.BatchNorm2d)
    assert isinstance(model.conv5, nn.Conv2d)
    assert isinstance(model.conv5_bn, nn.BatchNorm2d)
    assert isinstance(model.conv6, nn.Conv2d)
    assert isinstance(model.conv6_drop, nn.Dropout2d)
    assert isinstance(model.conv6_bn, nn.BatchNorm2d)

    assert isinstance(model.conv7, nn.Conv2d)
    assert isinstance(model.conv7_bn, nn.BatchNorm2d)
    assert isinstance(model.conv8, nn.Conv2d)
    assert isinstance(model.conv8_bn, nn.BatchNorm2d)
    assert isinstance(model.conv9, nn.Conv2d)
    assert isinstance(model.conv9_drop, nn.Dropout2d)
    assert isinstance(model.conv9_bn, nn.BatchNorm2d)

    assert isinstance(model.conv10, nn.Conv2d)
    assert isinstance(model.conv10_bn, nn.BatchNorm2d)
    assert isinstance(model.conv11, nn.Conv2d)
    assert isinstance(model.conv11_bn, nn.BatchNorm2d)
    assert isinstance(model.conv12, nn.Conv2d)
    assert isinstance(model.conv12_drop, nn.Dropout2d)
    assert isinstance(model.conv12_bn, nn.BatchNorm2d)

    assert isinstance(model.fc1, nn.Linear)
    assert isinstance(model.fc1_bn, nn.BatchNorm1d)
    assert isinstance(model.fc2, nn.Linear)
