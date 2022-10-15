from ast import Raise
from re import S
import re
import gym

import matplotlib.pyplot as plt

from citylearn.citylearn import CityLearnEnv
import numpy as np
import pandas as pd
import os

from collections import deque
import argparse 
import random
# import logger
import logging
from sys import stdout
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl

import pickle

import wandb
from pytorch_lightning.loggers import WandbLogger

import json

lookback = 5
lookfuture = 20

features_to_forecast = ['non_shiftable_load', 'solar_generation', 'electricity_pricing', 'carbon_intensity',
                                                             'hour', 'month']

hidden_feature = 64

non_shiftable_load_idx = features_to_forecast.index("non_shiftable_load")
solar_generation_idx = features_to_forecast.index("solar_generation")
electricity_pricing_idx = features_to_forecast.index("electricity_pricing")
carbon_intensity_idx = features_to_forecast.index("carbon_intensity") 

from agents.models import ModelCityLearn, ModelAfterPrediction, ModelCityLearnOptim, DatasetCityLearn

def test_model(model, test_loader):

    #model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):

            storage_random = torch.rand((x.shape[0],5))

            net_comsumption = torch.randn((x.shape[0], 5))

            print("storage_random", storage_random.shape)

            action, futur_state = model(x.float(), storage_random, net_comsumption)
            print(x.shape)
            print(y.shape)
            print(action.shape)
            print(futur_state.shape)
            break

    # we try one validation step
    model.validation_step((x, y), 0)
    #exit()

def rework_dataset(data):
    """
    Get a dataframe in input of size (nb_building * 8759, nb_feature)
    Return a list of dataframe of size (nb_building, 8760, nb_feature)
    """
    data_list = []
    
    data_list = list(data.groupby("building_id"))

    data_list = [data[1].drop("building_id", axis=1) for data in data_list]

    return data_list

def train_worldmodel(path_dataset):

    # we load the dataset
    data = pd.read_parquet(path_dataset)

    data.reset_index(inplace=True)


    data_list = rework_dataset(data)

    # we define the dataset fro validation and training
    data_train = data_list
    data_val = data_list

    # we define the dataset for training and validation
    dataset = DatasetCityLearn(data_train, lookback, lookfuture, features_to_forecast)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    dataset_val = DatasetCityLearn(data_val, lookback, lookfuture, features_to_forecast)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=False)

    # we define the model
    model = ModelCityLearnOptim(len(features_to_forecast), hidden_feature, len(features_to_forecast), lookback, lookfuture)

    # load model weight from previous training (to accelerate training)
    # model.load_state_dict(torch.load("models_checkpoint/model_world_v3.pt"))

    # model testing
    test_model(model, dataloader_val)
    
    train = True

    if train:
        os.environ['WANDB_API_KEY'] = "71d38d7113a35496e93c0cd6684b16faa4ba7554"
        wandb.init(project='citylearn', entity='forbu14')

        # init wandb logger
        wandb_logger = WandbLogger(project='citylearn', entity='forbu14')

        # we define the trainer
        trainer = pl.Trainer(max_epochs=20, logger=wandb_logger)

        # we train the model
        trainer.fit(model, dataloader, dataloader_val)

        # save model
        torch.save(model.state_dict(), 'models/model_world_v3.pt')

    return model

if __name__ == "__main__":
    
    # we train the model
    path_dataset = "data_histo/data.parquet"

    train_worldmodel(path_dataset)