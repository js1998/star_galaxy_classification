from astropy.io import fits
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from astropy.nddata.utils import Cutout2D
import glob, os
from esutil import wcsutil
from astropy.table import Table
import pandas as pd
import fitsio
from esutil import wcsutil
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from random import randint
import math
import sys
from astropy.visualization import hist
from torch.utils.data import DataLoader
from astroML.datasets import fetch_imaging_sample, fetch_sdss_S82standards

def train(config):
    train_data = PitcairnDataset(config, mode="train")
    inc = 1
    outc = 3
    model = MyUnet(inc, outc)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    tr_data_loader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=True)
    
    model.train()
    
    data_loss = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
    
    iter_idx = -1
    
    for epoch in range(config.num_epoch):
        prefix = "Training Epoch {:3d}: ".format(epoch)
        for data in tr_data_loader:
            iter_idx += 1
            x, y = data
            
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            
            logits = model.forward(x)
            loss = data_loss(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iter_idx % config.rep_intv == 0:
                print("training loss: " + loss)

def main(config):
    """The main function."""

    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)