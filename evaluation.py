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
    # model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            storage_random = torch.rand((x.shape[0], 5))

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
    # exit()


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

            net_demand_old = x[:, :, -1, non_shiftable_load_idx] - x[:, :, -1, solar_generation_idx]

            storage_random = torch.rand((x.shape[0], nb_building))
            action, futur_state = model(x.float(), storage_random, net_demand_old)

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

        plt.legend()
        plt.title(var)
        plt.show()

        # we save the figure
        plt.savefig("results_worldmodel/" + var + "_1.png")

        plt.figure(figsize=(100, 10))
        plt.plot(env_pred_5[var], label="pred_5")
        plt.plot(env_true_5[var], label="true_5")

        plt.legend()
        plt.title(var)
        plt.show()

        # we save the figure
        plt.savefig("results_worldmodel/" + var + "_5.png")


def plot_action_list(action_list, state_list):
    # first we concatenate the action list to get a proper tensor
    action_list = torch.cat(action_list, dim=0)

    # same thing for the state list
    state_list = torch.cat(state_list, dim=0)

    # action_list is of size (nb_step, nb_building) state list is of size (nb_step, nb_building, nb_feature)
    # we map the action to the state
    action_list = action_list.unsqueeze(2)

    # we concatenate the action and the state
    action_state_list = torch.cat((state_list, action_list), dim=2)

    # now we can change the dimension to (nb_step * nb_building, nb_feature + 1)
    action_state_list = action_state_list.view(-1, action_state_list.shape[-1])

    # we can create the DataFrame with columns features_to_forecast + [actions]
    df = pd.DataFrame(action_state_list.numpy(), columns=features_to_forecast + ["actions"])

    print(df.head())

    # now we can look at the action with respect to hour
    # define figure
    fig, ax = plt.subplots(1, 1, figsize=(70, 10))
    plot = df.groupby("hour")["actions"].mean().plot()
    fig = plot.get_figure()
    # we save the plot in the results_worldmodel folder
    fig.savefig("results_worldmodel/action_hour.png")


def compute_performance_validation(model, dataset_test):
    dataloader_val = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

    model.eval()

    storage_init = torch.zeros((1, 5))

    reward_price_tot = 0
    reward_emission_tot = 0
    reward_grid_tot = 0
    action_previous = 0

    print("begin validation")
    action_done_list = []
    state_list = []

    net_demand_old = torch.zeros((1, 5))

    # adding torch no grad
    with torch.no_grad():

        for batch_idx, (x, y) in enumerate(dataloader_val):
            action, futur_state = model(x.float(), storage_init, net_demand_old)

            reward, storage, reward_price, reward_emission, net_demand_new = model.simulation_one_step(y[:, :, 0, :],
                                                                                                       action[:, :, 0,
                                                                                                       0], storage_init)
            reward_price_tot += reward_price
            reward_emission_tot += reward_emission

            reward_grid_tot += torch.abs(net_demand_old.sum(axis=1) - net_demand_new.sum(axis=1))

            storage_init = storage

            net_demand_old = net_demand_new

            # append action[:, :, 0, 0] to action_done_list in list format
            action_done_list.append(action[:, :, 0, 0])
            state_list.append(x[:, :, -1, :])

        print("reward_price", reward_price_tot)
        print("reward_emission", reward_emission_tot)
        print("reward_grid", reward_grid_tot)

        # now we can a plot of the action done
        plot_action_list(action_done_list, state_list)

        # now we compute the reward for the baseline
        reward_price_tot_baseline = 0
        reward_emission_tot_baseline = 0
        reward_grid_tot_baseline = 0
        storage_init = torch.zeros((1, 5))

        for batch_idx, (x, y) in enumerate(dataloader_val):
            action = torch.zeros((1, 5))

            net_demand_old = x[:, :, -1, non_shiftable_load_idx] - x[:, :, -1, solar_generation_idx]

            # we compute the REAL reward and storage_init TODO
            # we apply the action to the storage
            reward, storage, reward_price, reward_emission, net_demand_new = model.simulation_one_step(y[:, :, 0, :],
                                                                                                       action,
                                                                                                       storage_init)
            reward_price_tot_baseline += reward_price
            reward_emission_tot_baseline += reward_emission
            reward_grid_tot_baseline += torch.abs(net_demand_old.sum(axis=1) - net_demand_new.sum(axis=1))

            storage_init = storage

        print("reward_price_baseline", reward_price_tot_baseline)
        print("reward_emission_baseline", reward_emission_tot_baseline)
        print("reward_grid_baseline", reward_grid_tot_baseline)

    return {"reward_price": reward_price_tot.mean(), "reward_emission": reward_emission_tot.mean(),
            "reward_grid": reward_grid_tot.mean(),
            "reward_price_baseline": reward_price_tot_baseline.mean(),
            "reward_emission_baseline": reward_emission_tot_baseline.mean(),
            "reward_grid_baseline": reward_grid_tot_baseline.mean()}


def evaluation_worldmodel(path_dataset):
    # we load the dataset
    data = pd.read_parquet(path_dataset)

    data.reset_index(inplace=True)

    print(data[features_to_forecast].describe())

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
    model = ModelCityLearnOptim(len(features_to_forecast), hidden_feature, len(features_to_forecast), lookback,
                                lookfuture)

    # load model from models_checkpoint/model_world.pt
    model.load_state_dict(torch.load("models/model_world_v3.pt"))

    # model testing
    test_model(model, dataloader_val)

    # TODO compute model performance on real data (reward performance)
    rewards = compute_performance_validation(model, dataset_val)

    performance = rewards

    # we compute the different ratios
    ratio_price = performance["reward_price"] / performance["reward_price_baseline"]
    ratio_emission = performance["reward_emission"] / performance["reward_emission_baseline"]
    ratio_grid = performance["reward_grid"] / performance["reward_grid_baseline"]

    # we add them to the performance dict
    performance["ratio_price"] = ratio_price
    performance["ratio_emission"] = ratio_emission
    performance["ratio_grid"] = ratio_grid

    print(performance)

    plot_performance_validation(model, dataset_val)

    # we convert every value to float
    for key in performance.keys():
        performance[key] = float(performance[key])

    # save file into json
    with open('metrics/performance_v3.json', 'w') as fp:
        json.dump(performance, fp)


if __name__ == "__main__":
    evaluation_worldmodel("data_histo/data.parquet")
