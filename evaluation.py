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
from agents.models import ModelCityLearn, ModelAfterPrediction, ModelCityLearnOptim, DatasetCityLearn

lookback = 5
lookfuture = 20

features_to_forecast = ['non_shiftable_load', 'solar_generation', 'electricity_pricing', 'carbon_intensity',
                                                             'hour', 'month']

hidden_feature = 64

non_shiftable_load_idx = features_to_forecast.index("non_shiftable_load")
solar_generation_idx = features_to_forecast.index("solar_generation")
electricity_pricing_idx = features_to_forecast.index("electricity_pricing")
carbon_intensity_idx = features_to_forecast.index("carbon_intensity") 


def rework_dataset(data):
    """
    Get a dataframe in input of size (nb_building * 8759, nb_feature)
    Return a list of dataframe of size (nb_building, 8760, nb_feature)
    """
    data_list = []
    
    data_list = list(data.groupby("building_id"))

    data_list = [data[1].drop("building_id", axis=1) for data in data_list]

    return data_list

def test_model(model, test_loader):

    #model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):

            storage_random = torch.rand((x.shape[0],5))

            print("storage_random", storage_random.shape)

            action, futur_state = model(x.float(), storage_random)
            print(x.shape)
            print(y.shape)
            print(action.shape)
            print(futur_state.shape)
            break

    # we try one validation step
    model.validation_step((x, y), 0)
    #exit()


def plot_performance_validation(model, dataset_test):
    """
    In this function we will plot the performance of the model on the validation set
    Only the environment value will be plotted (not the reward)
    """

    dataloader_val = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

    model.eval()

    # we create list to store the environment values (prediction and ground truth)
    env_pred_1 = {}
    env_true_1 = {}

    env_pred_5 = {}
    env_true_5 = {}

    for idx, var in enumerate(features_to_forecast):
        
        env_pred_1[var] = []
        env_true_1[var] = []

        env_pred_5[var] = []
        env_true_5[var] = []

    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(dataloader_val):

            # get nb_building
            nb_building = x.shape[1]

            storage_random = torch.rand((x.shape[0], nb_building))
            action, futur_state = model(x.float(), storage_random)
            
            # we denormalize the futur state using the mean and std of the training set
            std_torch = torch.tensor(model.std).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            mean_torch = torch.tensor(model.mean).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # we dneormalize the futur state and the input (x) and output (y)
            futur_state = futur_state * std_torch + mean_torch
            x = x * std_torch + mean_torch
            y = y * std_torch + mean_torch

            # we register the prediction and the ground truth for 2 horizon (1 and 5)
            for idx, var in enumerate(features_to_forecast):

                env_pred_1[var].append(futur_state[0, 0, 0, idx].item())
                env_true_1[var].append(y[0, 0, 0, idx].item())

                env_pred_5[var].append(futur_state[0, 0, 4, idx].item())
                env_true_5[var].append(y[0, 0, 4, idx].item())

    # mkdir if not exist results_worldmodel
    if not os.path.exists("results_worldmodel"):
        os.makedirs("results_worldmodel")

    # we plot the results
    for idx, var in enumerate(features_to_forecast):
            
        plt.figure(figsize=(100, 10))
        plt.plot(env_pred_1[var], label="pred_1")
        plt.plot(env_true_1[var], label="true_1")
        plt.plot(env_pred_5[var], label="pred_5")
        plt.plot(env_true_5[var], label="true_5")
        plt.legend()
        plt.title(var)
        plt.show()

        # we save the figure
        plt.savefig("results_worldmodel/" + var + ".png")

def compute_performance_validation(model, dataset_test):

    dataloader_val = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)


    model.eval()

    storage_init = torch.zeros((1, 5))

    reward_tot = 0
    reward_grid = 0
    
    print("begin validation")

    # adding torch no grad
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(dataloader_val):

            action, futur_state = model(x.float(), storage_init)

            net_demand_old = x[:, :, -1,  non_shiftable_load_idx] - x[:, :, -1, solar_generation_idx] 
            
            # we denormalize the futur state using the mean and std of the training set
            std_torch = torch.tensor(model.std).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            mean_torch = torch.tensor(model.mean).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # we dneormalize the futur state and the input (x) and output (y)
            futur_state = futur_state * std_torch + mean_torch
            x = x * std_torch + mean_torch
            y = y * std_torch + mean_torch

            # we compute the REAL reward and storage_init TODO
            # we apply the action to the storage
            
            reward, storage, reward_price, reward_emission, net_demand_new = model.simulation_one_step(y[:, :, 0, :], action[:, :, 0, 0], storage_init)
            reward_tot += reward_price

            reward_grid += torch.abs(net_demand_old.sum(axis=1) - net_demand_new.sum(axis=1))

            #print(reward)

        print("reward_tot", reward_tot)
        print("reward_grid", reward_grid)

        # now we compute the reward for the baseline
        reward_baseline = 0
        reward_grid_baseline = 0
        storage_init = torch.zeros((1, 5))

        for batch_idx, (x, y) in enumerate(dataloader_val):

            action = torch.zeros((1, 5))

            net_demand_old = x[:, :, -1,  non_shiftable_load_idx] - x[:, :, -1, solar_generation_idx] 
            
            # we denormalize the futur state using the mean and std of the training set
            std_torch = torch.tensor(model.std).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            mean_torch = torch.tensor(model.mean).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # we dneormalize the futur state and the input (x) and output (y)
            futur_state = futur_state * std_torch + mean_torch
            x = x * std_torch + mean_torch
            y = y * std_torch + mean_torch

            # we compute the REAL reward and storage_init TODO
            # we apply the action to the storage

            reward, storage, reward_price, reward_emission, net_demand_new = model.simulation_one_step(y[:, :, 0, :], action, storage_init)
            reward_baseline += reward
            reward_grid_baseline += torch.abs(net_demand_old.sum(axis=1) - net_demand_new.sum(axis=1))

        print("reward_baseline", reward_baseline)
        print("reward_grid_baseline", reward_grid_baseline)


    return reward_tot.sum(), reward_baseline.sum()


def evaluation_worldmodel(path_dataset):

    # we load the dataset
    data = pd.read_parquet(path_dataset)

    data.reset_index(inplace=True)

    # normalize the data
    # save the mean and std to denormalize the data
    mean = data[features_to_forecast].mean()
    std = data[features_to_forecast].std()

    data[features_to_forecast] = (data[features_to_forecast] - mean) / std

    data_list = rework_dataset(data)

    # load mean and std from pickle
    with open("models/mean_std.pkl", "rb") as f:
        mean, std = pickle.load(f)

    # we define the dataset fro validation and training
    data_train = data_list
    data_val = data_list

    # we define the dataset for training and validation
    dataset = DatasetCityLearn(data_train, lookback, lookfuture, features_to_forecast)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    dataset_val = DatasetCityLearn(data_val, lookback, lookfuture, features_to_forecast)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=False)

    # we define the model
    model = ModelCityLearnOptim(len(features_to_forecast), hidden_feature, len(features_to_forecast), lookback, lookfuture, mean, std)

    # load model from models_checkpoint/model_world.pt
    model.load_state_dict(torch.load("models/model_world_v3.pt"))

    # model testing
    test_model(model, dataloader_val)

    plot_performance_validation(model, dataset_val)

    # TODO compute model performance on real data (reward performance)
    reward_optim, reward_free = compute_performance_validation(model, dataset_val)

    print(reward_optim)
    print(reward_free)

    performance = {"reward_optim": reward_optim.detach().numpy().item(),
             "reward_free": reward_free.detach().numpy().item(), "ratio": (reward_optim/reward_free).detach().numpy().item()}

    # save file into json
    with open('metrics/performance_v3.json', 'w') as fp:
        json.dump(performance, fp)

if __name__ == "__main__":

    evaluation_worldmodel("data_histo/data.parquet")