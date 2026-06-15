# deep learning libraries
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.jit import RecursiveScriptModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets

import io
# other libraries
import os
import random


def load_data(
    path: str, batch_size: int = 128
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function loads the data from mnist dataset. All batches must
    be equal size. The division between train and val must be 0.8-0.2.

    Args:
        path: path to save the datasets.
        batch_size: batch size. Defaults to 128.

    Returns:
        tuple of three dataloaders, train, val and test in respective order.
    """

    # define transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root=path,train=True,download=True,transform=transform)
    test_data = datasets.MNIST(root=path,train=False,download=True,transform=transform)

    training_data,validation_data = random_split(train_data,[0.8,0.2])

    train = DataLoader(training_data,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)
    val = DataLoader(validation_data,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)
    test = DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=False)
    
    return (train,val,test)