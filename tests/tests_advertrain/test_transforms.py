from torchvision.transforms import (ColorJitter, Compose, Pad, RandomAffine,
                                    RandomApply, RandomHorizontalFlip,
                                    RandomVerticalFlip, Resize, ToTensor)

from RobustML.advertrain.transforms import DataTransformations


def test_initialization():
    dt = DataTransformations(train_prob=0.7)
    assert dt.train_prob == 0.7


def test_get_train_transforms():
    dt = DataTransformations(train_prob=0.5)
    train_transforms = dt.get_train_transforms()

    assert isinstance(train_transforms, Compose)
    assert isinstance(train_transforms.transforms[0], Pad)
    assert isinstance(train_transforms.transforms[1], RandomHorizontalFlip)
    assert isinstance(train_transforms.transforms[2], RandomVerticalFlip)
    assert isinstance(train_transforms.transforms[3], RandomApply)
    assert isinstance(train_transforms.transforms[3].transforms[0], RandomAffine)
    assert isinstance(train_transforms.transforms[3].transforms[1], RandomAffine)
    assert isinstance(train_transforms.transforms[3].transforms[2], ColorJitter)
    assert train_transforms.transforms[3].p == 0.5
    assert isinstance(train_transforms.transforms[4], Resize)
    assert isinstance(train_transforms.transforms[5], ToTensor)


def test_get_test_transforms():
    dt = DataTransformations()
    test_transforms = dt.get_test_transforms()

    assert isinstance(test_transforms, Compose)
    assert isinstance(test_transforms.transforms[0], Pad)
    assert isinstance(test_transforms.transforms[1], Resize)
    assert isinstance(test_transforms.transforms[2], ToTensor)
