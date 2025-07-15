import os
import json
import random
import torch
import numpy as np
from PIL import Image
import pandas as  pd
import math

from torch.utils.data import Dataset
#from transforms import get_train_transforms, get_test_transforms

import argparse

NUM_WORKERS = 16
DEFAULT_SCENARIO = "all"
DEFAULT_USECASE = "safran"
DEFAULT_RATE = 0.1






class SDNET2018Dataset_all(Dataset):
    """Create a torch dataset from a pandas DataFrame"""

    def __init__(self, root_path, df, transform):
        """
        OK samples will have label 0, and RETOUCHE samples will have label 1.

        Args:
        * root_path (string): path to the root folders with every concrete
        * df (Pandas DataFrame): Dataset dataframe
        * transform : PyTorch transform to apply (default: None)
        """

        self.root_path = root_path
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #return x, y
        path = os.path.join(self.root_path, self.df["Filename"].iloc[idx])
        # path =  path.replace('_OK', '') if "OK" in self.df["Filename"].iloc[idx] else path
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        label = 0 if "OK" in self.df["Filename"].iloc[idx] else 1

        return image, label


def get_split(filenames, split):
    """Return train and validation split from filenames."""
    random.shuffle(filenames)
    N = int(len(filenames) * split)
    test = filenames[:N]
    train = filenames[N:]
    N_tr = int(len(train) * split)
    val = train[:N_tr]
    train = train[N_tr:]
    
    return test, train, val



def get_concrete_train_test_filenames(root_path, concrete, validation_split):
    concrete_path = os.path.join(root_path, concrete)
    cracked_path = os.path.join(concrete_path, "C"+concrete)
    uncracked_path = os.path.join(concrete_path, "U"+concrete)
    
    cracked_filenames = [concrete+"/C"+concrete+"/" + f for f in os.listdir(cracked_path)]
    uncracked_filenames = [concrete+"/U"+concrete+"/" + f +"_OK" for f in os.listdir(uncracked_path)]
    
    test_unck_filenames, train_unck_filenames, val_unck_filenames = get_split(
        uncracked_filenames, validation_split)
    test_defect_filenames, train_defect_filenames, val_defect_filenames = get_split(
        cracked_filenames, validation_split)

    

    return (
        train_unck_filenames + train_defect_filenames,
        test_unck_filenames + test_defect_filenames,
        val_unck_filenames + val_defect_filenames,
    )


def dataframe_from_filenames(filenames, concrete):
    return pd.DataFrame(
        list(
            zip(
                [concrete] * len(filenames),
                filenames,
                [0 if "OK" in name else 1 for name in filenames],
            )
        ),
        columns=["Concrete", "Filename", "Label"],
    )


def create_dataset_all(root_path, validation_split):
    """
    Create the train / test dataset using the whole labelled data.

    The datasets will be as followed :
    * Train dataset : 90% of each classes per concrete.
    * Test dataset  : 10% of each classes per concrete, including the 2 concretes ignored from the train dataset.

    These 2 special concretes will be used to assess if the model is able to generalize well to unseen welding spots.

    Return the train and test datasets as pandas DataFrame.
    """

    concretes = os.listdir(root_path)
    ignored_concretes = []

    dfs_train = []
    dfs_val = []
    dfs_test = []

    for concrete in concretes:
        trainset, testset, valset = get_concrete_train_test_filenames(
            root_path, concrete, validation_split
        )

        df = dataframe_from_filenames(testset, concrete)
        dfs_test.append(df)

        if concrete not in ignored_concretes:
            df = dataframe_from_filenames(trainset, concrete)
            dfs_train.append(df)

            df = dataframe_from_filenames(valset, concrete)
            dfs_val.append(df)
    
    print(
        f"Train : {len(trainset)}\
        |Test : {len(testset)}\
        |validation : {len(valset)}")

    df_test = pd.concat(dfs_test)
    df_train = pd.concat(dfs_train)
    df_val = pd.concat(dfs_val)

    return df_train, df_val, df_test


def create_dataset_single(root_path, validation_split):
    """
    Create the train / test dataset of each modality of the data.

    The datasets will be as followed for each modality :
    * Train dataset : 90% of each classes .
    * Test dataset  : 10% of each classes .

    
    Return the the dic containing the train, test va data of each modality.
    """

    concretes = os.listdir(root_path)
    ignored_concretes = []

    dfs = {}
    #dfs_train = []
    #dfs_val = []
    #dfs_test = []

    for concrete in concretes:
        trainset, testset, valset = get_concrete_train_test_filenames(
            root_path, concrete, validation_split
        )

        #trainset, valset = get_concrete_train_test_filenames(root_path, concrete, 0.1)

        df_test = dataframe_from_filenames(testset, concrete)
                
        df_train = dataframe_from_filenames(trainset, concrete)
        
        df_val = dataframe_from_filenames(valset, concrete)
        
        dfs[concrete] = [df_train, df_val, df_test]
    
    print(
    f"Train : {len(trainset)}\
    |Test : {len(testset)}\
    |validation : {len(valset)}")


    return dfs


def get_train_dataloader(
    root_dir,
    csv_path_train,
    csv_path_val,
    batch_size=16,
    train_transform=None,
    test_transform=None,
):
    # Load data
    df = pd.read_csv(csv_path_train)

    # Create weight sampler
    n_defects = np.sum(df["Label"])
    n_ok = len(df) - n_defects

    bins = [1 / n_ok, 1 / n_defects]
    weights = []
    for label in df["Label"]:
        weights.append(bins[label])

    weights = torch.from_numpy(np.array(weights))
    samp = torch.utils.data.sampler.WeightedRandomSampler(weights, 12000)

    # Create dataloader
    train_dataset = SDNET2018Dataset_all(
        root_dir,
        df=df,
        transform=train_transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=samp,
        num_workers=NUM_WORKERS,
    )

    df = pd.read_csv(csv_path_val)
    val_dataset = SDNET2018Dataset_all(
        root_dir,
        df=df,
        transform=test_transform,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    return train_dataloader, val_dataloader


def get_test_dataloader(root_dir, csv_path, test_transform=None):
    df = pd.read_csv(csv_path)

    dataloaders = {}

    concretes = df["Concrete"].unique()
    for concrete in concretes:
        df_concrete = df[df["Concrete"] == concrete]

        dataset = SDNET2018Dataset_all(
            root_dir,
            df=df_concrete,
            transform=test_transform,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=NUM_WORKERS,
        )

        dataloaders[concrete] = dataloader

    return dataloaders


def learning_data_generator(data_path, test_rate):

    df_train, df_val, df_test  = create_dataset_all(data_path, validation_split=test_rate)

    path = "learning_data/"
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    df_train.to_csv(path+"train_dataset_all.csv")
    df_val.to_csv(path+"val_dataset_all.csv")
    df_test.to_csv(path+"test_dataset_all.csv")
