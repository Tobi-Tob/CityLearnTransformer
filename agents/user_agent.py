from agents.random_agent import RandomAgent
from agents.rbc_agent import BasicRBCAgent

import pickle

from collections import deque
import os
import sys

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import optax

import torch
import torch.nn as nn
import pytorch_lightning as pl

import time

###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
# UserAgent = BasicRBCAgent

str_to_int_mapping = {
    "month"  :   0,
"day_type"  :    1,
"hour"  :    2,
"outdoor_dry_bulb_temperature"  :  3,
"outdoor_dry_bulb_temperature_predicted_6h"  :    4,
"outdoor_dry_bulb_temperature_predicted_12h"  :    5,
"outdoor_dry_bulb_temperature_predicted_24h"  :    6,
"outdoor_relative_humidity"  :   7,
"outdoor_relative_humidity_predicted_6h"  :   8,
"outdoor_relative_humidity_predicted_12h"  :    9,
"outdoor_relative_humidity_predicted_24h"  :  10,
"diffuse_solar_irradiance"  :    11,
"diffuse_solar_irradiance_predicted_6h"  :   12,
"diffuse_solar_irradiance_predicted_12h"  :   13,
"diffuse_solar_irradiance_predicted_24h"  :   14,
"direct_solar_irradiance"  : 15,
"direct_solar_irradiance_predicted_6h"  :   16,
"direct_solar_irradiance_predicted_12h"  :  17,
"direct_solar_irradiance_predicted_24h"  :  18,
"carbon_intensity"  :  19,
"non_shiftable_load"  :  20,
"solar_generation"  :  21,
"electrical_storage_soc"  : 22,
"net_electricity_consumption"  :  23,
"electricity_pricing"  :   24,
"electricity_pricing_predicted_6h"  : 25,
"electricity_pricing_predicted_12h"  :  26,
"electricity_pricing_predicted_24h"  : 27,
}

int_to_str_mapping = {v: k for k, v in str_to_int_mapping.items()}


features_to_forecast = ['non_shiftable_load', 'solar_generation', 'electricity_pricing', 'carbon_intensity',
                                                             'hour', 'month']

hidden_feature = 64
lookback = 5
lookfuture = 20


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

            reward_tot = reward_tot + reward

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

        return reward_electricity_price + reward_carbon_intensity, electrical_soc_new



class UserAgent:
    def __init__(self):

        self.action_space = {}

        # we init a history of information specific to the agent
        self.history = {}

        self.features_to_forecast = features_to_forecast

        # we load the mean and std
        with open('models/mean_std.pkl', 'rb') as f:
            self.mean, self.std = pickle.load(f)

        # we define the model
        self.model = ModelCityLearnOptim(len(features_to_forecast), hidden_feature, len(features_to_forecast),
                                                             lookback, lookfuture, self.mean, self.std)

        # load model from models_checkpoint/model_world.pt
        self.model.load_state_dict(torch.load("models_checkpoint/model_world.pt"))


    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

        # we init a history of information specific to the agent
        self.history[agent_id] = {var : deque(maxlen=lookback) for var in self.features_to_forecast}

    def adding_observation(self, observation, agent_id):
        """Get observation return action"""

        # add observation to the history
        for var in self.features_to_forecast:
            self.history[agent_id][var].append(observation[var])
    
    def get_model_prediction(self, agent_id, storage_current):
        """Get observation return action"""

        # we get the last 24*7 observations for every variables using history variable
        # we need to reshape the data to be able to use the model
        data = {var : list(self.history[agent_id][var]) for var in self.features_to_forecast}

        # we convert the data to a dataframe
        data = pd.DataFrame(data, columns=self.features_to_forecast)

        data_pt = torch.tensor(data.to_numpy())

        # put into format (1, lookback, len(features_to_forecast))
        data_pt = data_pt.unsqueeze(0)

        std_torch = torch.tensor(self.model.std).float().unsqueeze(0).unsqueeze(0)
        mean_torch = torch.tensor(self.model.mean).float().unsqueeze(0).unsqueeze(0)

        # we normalize the data
        data_pt = (data_pt - mean_torch) / std_torch

        action, futur_state = self.model(data_pt.float(), torch.tensor([storage_current]).float())

        return action[0, 0, 0].detach().numpy().item()

    def augment_observations(self, observation, agent_id):

        observation_dict = {}

        for idx, obs in enumerate(observation):
            observation_dict[int_to_str_mapping[idx]] = obs

        return observation_dict
    
    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        # we augment the observation with the crude electricity consumption and the index pricing
        observation = self.augment_observations(observation, agent_id)

        current_soc = observation['electrical_storage_soc']

        # add observation to the history
        self.adding_observation(observation, agent_id)

        # if there is less than lookback observations, we return 0 as action
        if len(self.history[agent_id]['non_shiftable_load']) < lookback:
            return [0.]
        else:

            # we get the prediction
            action_final = self.get_model_prediction(agent_id, current_soc)

            #print("action final", action_final)

            return [float(action_final)]
