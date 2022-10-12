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

# create logger
logger = logging.getLogger(__name__)

# INFO level
logger.setLevel(logging.INFO)

# create formatter
handler = logging.StreamHandler(stdout)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)

# add formatter to logger
logger.addHandler(handler)

lookback = 5
lookfuture = 20

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

    def forward(self, input):
        
        # we complete the input with 0 values (lookfuture)
        input = torch.cat((input, torch.zeros(input.shape[0], self.lookfuture-1, input.shape[2])), dim = 1)

        output, _ = self.lstm1(input)

        # relu activation
        output = nn.functional.relu(output)

        output = self.linear_1(output)
        # relu 
        output = nn.functional.relu(output)

        output = self.linear(output)

        return output[:, -(self.lookfuture):, :]


class ModelAfterPrediction(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1):
        super(ModelAfterPrediction, self).__init__()
        self.hidden_size = hidden_size

        # first layer is a BiLSTM layer
        self.lstm1 = nn.LSTM(input_size + 1, hidden_size, batch_first = True, bidirectional = True)

        # second layer is a linear layer
        self.linear = nn.Linear(hidden_size*2, output_size)

    def forward(self, input, storage_random):
        # TODO we should add a storage_random to store the random values

        # storage_random is (B,) we should transform it into (B, lookfuture, 1)
        storage_random = storage_random.unsqueeze(1).unsqueeze(2).repeat(1, input.shape[1], 1)

        # we concatenate the input with the random values
        input = torch.cat((input, storage_random), dim = 2)

        output, _ = self.lstm1(input)

        # relu activation
        output = nn.functional.relu(output)

        output = self.linear(output)

        # clip output to get a result between -1 and 1
        output = torch.clamp(output, -1, 1)

        return output

class ModelCityLearnOptim(pl.LightningModule):
    """
    In this model we will learn to optimize the action with a learn model
    And also using reward hacking.

    The idea to to forecast the futur state from the past (the past is lookback param and the future is lookfuture param)

    We train the world model directly using state loss
    We train the action model using the reward loss

    """
    def __init__(self, input_size, hidden_size, output_size, lookback = 5, lookforward = 20, mean = 0, std = 1):
        super().__init__()

        # already trained learn model
        self.world_model = ModelCityLearn(input_size, hidden_size, output_size, lookback, lookforward)

        # we define the model that we want to train
        self.action_model = ModelAfterPrediction(input_size, hidden_size, output_size=1)

        self.lookback = lookback
        self.lookforward = lookforward

        # mean and std to properly compute the reward
        self.mean = mean
        self.std = std

        self.gamma = 0.99

        self.loss_env = nn.MSELoss()

    def forward(self, x, storage):
        # apply autoregressive model
        futur_state = self.world_model(x)

        # we predict the action
        action = self.action_model(futur_state.detach(), storage)

        return action, futur_state

    def training_step(self, batch, batch_idx):

        x, y = batch

        # here we should generate a random storage value
        storage_random = torch.rand((x.shape[0],))

        # we predict the action
        action, futur_state = self(x.float(), storage_random)

        # we compute the loss
        loss_env = self.loss_env(y.float(), futur_state)

        # loss for the first step and the 5th step
        loss_env_1 = self.loss_env(y[:, 0, :], futur_state[:, 0, :])
        loss_env_5 = self.loss_env(y[:, 4, :], futur_state[:, 4, :])

        # we compute the reward
        loss_reward = self.loss_reward(action, futur_state.detach(), storage_random)
        self.log('train_loss_reward', loss_reward)

        # we log the two reward
        self.log('train_loss_env', loss_env)
        self.log('train_loss_env_1', loss_env_1)
        self.log('train_loss_env_5', loss_env_5)

        return loss_env + loss_reward

    def validation_step(self, batch, batch_idx):

        x, y = batch

        # here we should generate a random storage value
        storage_random = torch.rand((x.shape[0],))

        # we predict the action
        action, futur_state = self(x.float(), storage_random)

        # we compute the loss
        loss_env = self.loss_env(y.float(), futur_state)

        # loss for the first step and the 5th step
        loss_env_1 = self.loss_env(y[:, 0, :], futur_state[:, 0, :])
        loss_env_5 = self.loss_env(y[:, 4, :], futur_state[:, 4, :])

        # we compute the reward
        loss_reward = self.loss_reward(action, futur_state.detach(), storage_random)

        self.log('val_loss_reward', loss_reward)

        # we log the two reward        
        self.log('val_loss_env', loss_env)
        self.log('val_loss_env_1', loss_env_1)
        self.log('val_loss_env_5', loss_env_5)

        return loss_env + loss_reward

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def loss_reward(self, action, futur_state, storage_random):

        # we initiate the storage with a random value between -1 and 1 for every sample
        # get batch size
        storage = storage_random

        # init the reward array of shape (batch_size, 1)
        reward_tot = torch.zeros(action.shape[0],)

        # we denormalize the futur state using the mean and std of the training set
        std_torch = torch.tensor(self.std).float().unsqueeze(0).unsqueeze(0)
        mean_torch = torch.tensor(self.mean).float().unsqueeze(0).unsqueeze(0)

        futur_state = futur_state * std_torch + mean_torch

        # we compute the reward
        for i in range(self.lookforward):
            # we compute the reward

            reward, storage = self.simulation_one_step(futur_state[:, i, :], action[:, i, 0], storage)

            reward_tot = reward_tot + reward * self.gamma**i

        return torch.mean(reward_tot)

    def simulation_one_step(self, futur_state, action, storage):

        # we clip the action to get a value between -1 and 1
        action = torch.clamp(action, -1, 1)

        # also for the action we cannot demand more than the space left in the battery (electrical_soc)
        # and we cannot spend more electrical energy than the one that we have
        action = torch.clamp(action, -storage*0.9, (1 - storage)/0.9)

        assert torch.all(storage >= 0), "electrical_soc must be positive"
        assert torch.all(storage <= 1), "electrical_soc must be less than 1"

        # first we compute the full demand for electrical power
        full_demand = futur_state[:, non_shiftable_load_idx] - futur_state[:, solar_generation_idx] + action*6.4

        # the full demand have to be fullfilled by the electrical grid
        reward_electricity_price = full_demand*futur_state[:, electricity_pricing_idx]
        reward_electricity_price = torch.clamp(reward_electricity_price, 0, 10000000)

        # we compute the reward for the carbon intensity
        reward_carbon_intensity = full_demand*futur_state[:, carbon_intensity_idx]
        reward_carbon_intensity = torch.clamp(reward_carbon_intensity, 0, 10000000)

        # we compute the new electrical soc value
        # now we can compute the efficiency
        eff =  0.9110

        # we compute the new electrical soc value
        electrical_soc_new = torch.where(action >= 0, storage + action*eff, storage + action/eff)
        electrical_soc_new = torch.clamp(electrical_soc_new, 0, 1)

        return reward_electricity_price + reward_carbon_intensity*1.5, electrical_soc_new

# we define the dataset
class DatasetCityLearn(torch.utils.data.Dataset):

    def __init__(self, data, lookback, lookfuture, features_to_forecast):
        self.data = data
        self.lookback = lookback
        self.lookfuture = lookfuture
        self.features_to_forecast = features_to_forecast

    def __len__(self):
        return len(self.data) - self.lookback - self.lookfuture

    def __getitem__(self, idx):
        # we get the data
        x = self.data[self.features_to_forecast].iloc[idx:(idx + self.lookback)].values
        y = self.data[self.features_to_forecast].iloc[(idx + self.lookback):(idx + self.lookback + self.lookfuture)].values
        return x, y

def test_model(model, test_loader):

    #model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):

            storage_random = torch.rand((x.shape[0],))

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

            storage_random = torch.rand((x.shape[0],))
            action, futur_state = model(x.float(), storage_random)
            
            # we denormalize the futur state using the mean and std of the training set
            std_torch = torch.tensor(model.std).float().unsqueeze(0).unsqueeze(0)
            mean_torch = torch.tensor(model.mean).float().unsqueeze(0).unsqueeze(0)

            # we dneormalize the futur state and the input (x) and output (y)
            futur_state = futur_state * std_torch + mean_torch
            x = x * std_torch + mean_torch
            y = y * std_torch + mean_torch

            # we register the prediction and the ground truth for 2 horizon (1 and 5)
            for idx, var in enumerate(features_to_forecast):

                env_pred_1[var].append(futur_state[0, 0, idx].item())
                env_true_1[var].append(y[0, 0, idx].item())

                env_pred_5[var].append(futur_state[0, 4, idx].item())
                env_true_5[var].append(y[0, 4, idx].item())

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

    storage_init = torch.tensor([0.])

    reward_tot = 0
    
    # adding torch no grad
    with torch.no_grad():
        
        for batch_idx, (x, y) in enumerate(dataloader_val):

            
            action, futur_state = model(x.float(), storage_init)

            
            # we denormalize the futur state using the mean and std of the training set
            std_torch = torch.tensor(model.std).float().unsqueeze(0).unsqueeze(0)
            mean_torch = torch.tensor(model.mean).float().unsqueeze(0).unsqueeze(0)

            # we dneormalize the futur state and the input (x) and output (y)
            futur_state = futur_state * std_torch + mean_torch
            x = x * std_torch + mean_torch
            y = y * std_torch + mean_torch

            # we compute the REAL reward and storage_init TODO
            # we apply the action to the storage
            reward, storage_init = model.simulation_one_step(y[:, 0, :], action[:, 0, 0], storage_init)
            reward_tot += reward

            #print(reward)

        print("reward_tot", reward_tot)

        # now we compute the reward for the baseline
        reward_baseline = 0
        storage_init = torch.tensor([0.])

        for batch_idx, (x, y) in enumerate(dataloader_val):

            action = torch.tensor([0.])
            
            # we denormalize the futur state using the mean and std of the training set
            std_torch = torch.tensor(model.std).float().unsqueeze(0).unsqueeze(0)
            mean_torch = torch.tensor(model.mean).float().unsqueeze(0).unsqueeze(0)

            # we dneormalize the futur state and the input (x) and output (y)
            futur_state = futur_state * std_torch + mean_torch
            x = x * std_torch + mean_torch
            y = y * std_torch + mean_torch

            # we compute the REAL reward and storage_init TODO
            # we apply the action to the storage
            reward, storage_init = model.simulation_one_step(y[:, 0, :], action, storage_init)
            reward_baseline += reward

        print("reward_baseline", reward_baseline)


    return reward_tot, reward_baseline

        
def train_worldmodel(path_dataset):

    # we load the dataset
    data = pd.read_parquet(path_dataset)

    data.reset_index(inplace=True)

    # normalize the data
    # save the mean and std to denormalize the data
    mean = data[features_to_forecast].mean()
    std = data[features_to_forecast].std()

    # save to pickle
    with open("models/mean_std.pkl", "wb") as f:
        pickle.dump([mean, std], f)

    data[features_to_forecast] = (data[features_to_forecast] - mean) / std

    # we define the dataset fro validation and training
    data_train = data
    data_val = data.iloc[-8759:]

    # we define the dataset for training and validation
    dataset = DatasetCityLearn(data_train, lookback, lookfuture, features_to_forecast)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    dataset_val = DatasetCityLearn(data_val, lookback, lookfuture, features_to_forecast)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False)

    # we define the model
    model = ModelCityLearnOptim(len(features_to_forecast), hidden_feature, len(features_to_forecast), lookback, lookfuture, mean, std)

    # load model from models_checkpoint/model_world.pt
    #model.load_state_dict(torch.load("models_checkpoint/model_world.pt"))

    # model testing
    test_model(model, dataloader_val)
    
    train = True

    if train:
        os.environ['WANDB_API_KEY'] = "71d38d7113a35496e93c0cd6684b16faa4ba7554"
        wandb.init(project='citylearn', entity='forbu14')

        # init wandb logger
        wandb_logger = WandbLogger(project='citylearn', entity='forbu14')

        # we define the trainer
        trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

        # we train the model
        trainer.fit(model, dataloader, dataloader_val)

        # save model
        torch.save(model.state_dict(), 'models/model_world.pt')

    # plot performance after training
    plot_performance_validation(model, dataset_val)

    # TODO compute model performance on real data (reward performance)
    reward_optim, reward_free = compute_performance_validation(model, dataset_val)

    performance = {"reward_optim": list(reward_optim.detach().numpy()),
             "reward_free": list(reward_free.detach().numpy()), "ratio": list((reward_optim/reward_free).detach().numpy())}

    # save file into json
    with open('metrics/performance.json', 'w') as fp:
        json.dump(performance, fp)

    return model

if __name__ == "__main__":
    
    # we train the model
    path_dataset = "data.parquet"

    train_worldmodel(path_dataset)