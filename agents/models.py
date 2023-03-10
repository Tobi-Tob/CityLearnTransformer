
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

features_to_forecast = ['non_shiftable_load', 'solar_generation', 'electricity_pricing', 'carbon_intensity',
                                                             'hour', 'month']

hidden_feature = 64

non_shiftable_load_idx = features_to_forecast.index("non_shiftable_load")
solar_generation_idx = features_to_forecast.index("solar_generation")
electricity_pricing_idx = features_to_forecast.index("electricity_pricing")
carbon_intensity_idx = features_to_forecast.index("carbon_intensity") 

# we train the model with pytorch RNN or LSTM
# we define the model
class ModelCityLearn(nn.Module):
    """
    A 2 layers model with 128 hidden units in each layer
    At each layers we have a LSTM layer
    The last layer is a linear layer

    """
    def __init__(self, input_size, hidden_size, output_size, lookback, lookfuture):
        super(ModelCityLearn, self).__init__()
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        # loss function definition
        self.loss = nn.MSELoss()

        self.lookback = lookback
        self.lookfuture = lookfuture

        self.output_size = output_size

    def forward(self, input):

        # input is a tensor of shape (batch_size, nb_building, lookback, input_size)
        # we need to reshape it to (batch_size* nb_building, lookback, input_size)
        batch_size = input.shape[0]
        nb_building = input.shape[1]
        input = input.reshape(batch_size*nb_building, self.lookback, -1)
        
        # we complete the input with 0 values (lookfuture)
        input = torch.cat((input, torch.zeros(input.shape[0], self.lookfuture-1, input.shape[2])), dim = 1)

        output, _ = self.lstm1(input)

        # relu activation
        output = nn.functional.elu(output)

        output = self.linear_1(output)
        # relu 
        output = nn.functional.elu(output)

        output = self.linear(output)

        # we reshape the output to (batch_size, nb_building, lookfuture, output_size)
        output = output[:, -(self.lookfuture):, :]
        output = output.reshape(batch_size, nb_building, self.lookfuture, self.output_size)

        return output


class ModelAfterPrediction(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1):
        super(ModelAfterPrediction, self).__init__()
        self.hidden_size = hidden_size

        # first layer is a BiLSTM layer
        self.lstm1 = nn.LSTM(input_size + 2, hidden_size, batch_first = True, bidirectional = True)

        # second layer is a linear layer
        self.linear = nn.Linear(hidden_size*2, output_size)

    def forward(self, input, storage_random, netconsumption_random):
        # input is a tensor of shape (batch_size, nb_building, lookback, input_size)
        # we need to reshape it to (batch_size* nb_building, lookback, input_size)
        batch_size = input.shape[0]
        nb_building = input.shape[1]
        input = input.reshape(batch_size*nb_building, -1, input.shape[-1])

        # storage_random is (B, nb_buildings) we should transform it into (B*nb_buildings, lookfuture, 1)

        storage_random = storage_random.unsqueeze(2).unsqueeze(3).repeat(1, 1, input.shape[1], 1)
        storage_random = storage_random.reshape(batch_size*nb_building, -1, 1)

        netconsumption_random = netconsumption_random.unsqueeze(2).unsqueeze(3).repeat(1, 1, input.shape[1], 1)
        netconsumption_random = netconsumption_random.reshape(batch_size*nb_building, -1, 1)

        # we concatenate the input with the random values
        input = torch.cat((input, storage_random, netconsumption_random), dim = 2)

        output, _ = self.lstm1(input)

        # relu activation
        output = nn.functional.elu(output)

        output = self.linear(output)

        # clip output to get a result between -1 and 1
        output = torch.clamp(output, -1, 1)

        # we reshape the output to (batch_size, nb_building, lookfuture, output_size)
        output = output.reshape(batch_size, nb_building, -1, 1)

        return output

class ModelCityLearnOptim(pl.LightningModule):
    """
    In this model we will learn to optimize the action with a learn model
    And also using reward hacking.

    The idea to to forecast the futur state from the past (the past is lookback param and the future is lookfuture param)

    We train the world model directly using state loss
    We train the action model using the reward loss

    """
    def __init__(self, input_size, hidden_size, output_size, lookback = 5, lookforward = 20):
        super().__init__()

        # already trained learn model
        self.world_model = ModelCityLearn(input_size, hidden_size, output_size, lookback, lookforward)

        # we define the model that we want to train
        self.action_model = ModelAfterPrediction(input_size, hidden_size, output_size=1)

        self.lookback = lookback
        self.lookforward = lookforward

        self.gamma = 0.99

        self.loss_env = nn.MSELoss()

        self.std = torch.tensor([0.889005, 1.017158, 0.117803, 0.035369, 0.288406, 0.287340])
        self.std = self.std.unsqueeze(0).unsqueeze(0).unsqueeze(0)



    def forward(self, x, storage, past_netconsumption):
        
        x = x / self.std # scale the input

        futur_state = self.world_model(x) # return (batch_size, nb_building, lookfuture, output_size)

        # we predict the action
        action = self.action_model(futur_state.detach(), storage, past_netconsumption) # return (batch_size, nb_building, lookfuture, 1)

        return action, futur_state

    def normalize_env_loss(self, y, y_pred):

        # we normalize the loss by the std of the output
        y = y / self.std
        y_pred = y_pred / self.std

        return self.loss_env(y_pred, y)

    def step(self, batch, batch_idx):
        x, y = batch

        # get nb_building
        nb_building = x.shape[1]

        # here we should generate a random storage value
        storage_random = torch.rand((x.shape[0], nb_building))

        net_demand_old_norm = x[:, :, -1,  non_shiftable_load_idx] - x[:, :, -1, solar_generation_idx] 
        
        # we predict the action
        action, futur_state = self(x.float(), storage_random, net_demand_old_norm)

        # we compute the loss
        loss_env = self.normalize_env_loss(y.float(), futur_state)

        # loss for the first step and the 5th step
        loss_env_1 = self.loss_env(y[:, :, 0, :], futur_state[:, :, 0, :])
        loss_env_5 = self.loss_env(y[:, :, 4, :], futur_state[:, :, 4, :])

        # we compute the reward
        loss_reward_price, loss_reward_emission, loss_reward_grid = self.loss_reward(action, futur_state.detach(), storage_random, net_demand_old_norm, y)

        return loss_env, loss_env_1, loss_env_5, loss_reward_price, loss_reward_emission, loss_reward_grid

    def training_step(self, batch, batch_idx):

        loss_env, loss_env_1, loss_env_5, loss_reward_price, loss_reward_emission, loss_reward_grid = self.step(batch, batch_idx)


        loss_reward_price, loss_reward_emission, loss_reward_grid = self.normalize_rewards(loss_reward_price, loss_reward_emission, loss_reward_grid)


        self.log('train_loss_reward_emission', loss_reward_emission)
        self.log('train_loss_reward_price', loss_reward_price)
        self.log('train_loss_reward_grid', loss_reward_grid)

        # we log the two reward
        self.log('train_loss_env', loss_env)
        self.log('train_loss_env_1', loss_env_1)
        self.log('train_loss_env_5', loss_env_5)

        return self.cost_definition(loss_env, loss_reward_price, loss_reward_emission, loss_reward_grid)

    def validation_step(self, batch, batch_idx):

        loss_env, loss_env_1, loss_env_5, loss_reward_price, loss_reward_emission, loss_reward_grid = self.step(batch, batch_idx)

        loss_reward_price, loss_reward_emission, loss_reward_grid = self.normalize_rewards(loss_reward_price, loss_reward_emission, loss_reward_grid)

        self.log('val_loss_reward_emission', loss_reward_emission)
        self.log('val_loss_reward_price', loss_reward_price)
        self.log('val_loss_reward_grid', loss_reward_grid)

        # we log the two reward        
        self.log('val_loss_env', loss_env)
        self.log('val_loss_env_1', loss_env_1)
        self.log('val_loss_env_5', loss_env_5)

        return self.cost_definition(loss_env, loss_reward_price, loss_reward_emission, loss_reward_grid)


    def cost_definition(self, loss_env, loss_reward_price, loss_reward_emission, loss_reward_grid):
        return loss_env + loss_reward_price + loss_reward_emission #+ loss_reward_grid

    def normalize_rewards(self, loss_reward_price, loss_reward_emission, loss_reward_grid):
        # we normalize the reward
        loss_reward_price = loss_reward_price
        loss_reward_emission = loss_reward_emission*2
        loss_reward_grid = loss_reward_grid/10

        return loss_reward_price, loss_reward_emission, loss_reward_grid


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def loss_reward(self, action, futur_state, storage_random, net_demand_old, y):

        # get nb_building
        nb_building = futur_state.shape[1]

        # we initiate the storage with a random value between -1 and 1 for every sample
        # get batch size
        storage = storage_random

        # init the reward array of shape (batch_size, 1)
        reward_emission_tot = torch.zeros(action.shape[0], nb_building)
        reward_price_tot = torch.zeros(action.shape[0], nb_building)

        reward_grid_tot = torch.zeros(action.shape[0],)

        # we compute the reward
        for i in range(self.lookforward):
            # we compute the reward

            reward, storage, reward_price, reward_emission, net_demand_new = self.simulation_one_step(y[:, :, i, :],
                                                            action[:, :, i, 0], storage)


            # compute grid reward from net demand old and new
            reward_grid_tot += torch.abs(net_demand_old.sum(axis=1) - net_demand_new.sum(axis=1))

            reward_emission_tot += reward_emission 
            reward_price_tot += reward_price
            # update the net demand
            net_demand_old = net_demand_new

        return torch.sum(reward_price_tot), torch.sum(reward_emission_tot), torch.sum(reward_grid_tot)

    def simulation_one_step(self, futur_state, action, storage):

        # we clip the action to get a value between -1 and 1
        action = torch.clamp(action, -1, 1)

        # also for the action we cannot demand more than the space left in the battery (electrical_soc)
        # and we cannot spend more electrical energy than the one that we have
        action = torch.clamp(action, -storage*0.9, (1 - storage)/0.9)

        assert torch.all(storage >= 0), "electrical_soc must be positive"
        assert torch.all(storage <= 1), "electrical_soc must be less than 1"

        # first we compute the full demand for electrical power
        full_demand = futur_state[:, :,  non_shiftable_load_idx] - futur_state[:, :, solar_generation_idx] + action*6.4

        # net emission is 
        net_emission = full_demand

        # the full demand have to be fullfilled by the electrical grid
        reward_electricity_price = full_demand*futur_state[:, :, electricity_pricing_idx]
        reward_electricity_price = torch.clamp(reward_electricity_price, 0, 10000000)

        # we compute the reward for the carbon intensity
        reward_carbon_intensity = full_demand*futur_state[:, :, carbon_intensity_idx]
        reward_carbon_intensity = torch.clamp(reward_carbon_intensity, 0, 10000000)

        # we compute the new electrical soc value
        # now we can compute the efficiency
        eff =  0.9110

        # we compute the new electrical soc value
        electrical_soc_new = torch.where(action >= 0, storage + action*eff, storage + action/eff)
        electrical_soc_new = torch.clamp(electrical_soc_new, 0, 1)

        return reward_electricity_price + reward_carbon_intensity, electrical_soc_new, reward_electricity_price, reward_carbon_intensity, net_emission

# we define the dataset
class DatasetCityLearn(torch.utils.data.Dataset):

    def __init__(self, data_list, lookback, lookfuture, features_to_forecast):
        self.data_list = data_list
        self.lookback = lookback
        self.lookfuture = lookfuture
        self.features_to_forecast = features_to_forecast

        self.nb_building = len(data_list)

        # TODO rework dataset part to get a proper dataset that return 5 building informations
        # in format (batch_size, nb_building, lookback, features)

    def __len__(self):
        return len(self.data_list[0]) - self.lookback - self.lookfuture - 5

    def __getitem__(self, idx):
        # we get the data

        x_tot = []
        y_tot = []

        for i in range(self.nb_building):
            x = self.data_list[i][self.features_to_forecast].iloc[idx:(idx + self.lookback)].values
            y = self.data_list[i][self.features_to_forecast].iloc[(idx + self.lookback):(idx + self.lookback + self.lookfuture)].values

            x_tot.append(x)
            y_tot.append(y)

        x_tot = np.concatenate(np.expand_dims(x_tot, axis = 0))
        y_tot = np.concatenate(np.expand_dims(y_tot, axis = 0))

        # convert to np.float32
        x_tot = x_tot.astype(np.float32)
        y_tot = y_tot.astype(np.float32)

        # size of data (nb_building, lookback, features)
        return x_tot, y_tot
