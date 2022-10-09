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

import pickle

import wandb
from pytorch_lightning.loggers import WandbLogger

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



def jax_gym_simulation(env_variable, electrical_soc, action, params):
    """
    this is a simple jax function that take env variable, electrical soc and action
    and return a reward and a futur electrical soc
    
    Parameters:
    -----------
    env_variable: dict
        the env variable that we want to use to predict the futur reward
    electrical_soc: float
        the electrical soc of the building
    action: float
        the action that we want to apply to the building
    
    Returns:
    --------
    reward: float
        the reward that we get from the action
    electrical_soc: float
        the futur electrical soc of the building
    """

    # we make a correction on the action demand 
    action = jnp.clip(action, -1, 1)

    # also for the action we cannot demand more than the space left in the battery (electrical_soc)
    # and we cannot spend more electrical energy than the one that we have
    action = jnp.clip(action, -electrical_soc*0.9, (1 - electrical_soc)/0.9)

    assert electrical_soc >= 0, "electrical_soc must be positive"
    assert electrical_soc <= 1, "electrical_soc must be less than 1"

    # first we compute the full demand for electrical power
    full_demand = env_variable['non_shiftable_load'] - env_variable['solar_generation'] + action*params['soc_max']

    # the full demand have to be fullfilled by the electrical grid
    reward_electricity_price = jnp.clip(full_demand*env_variable['electricity_pricing'], 0, 1000000)

    # we compute the reward for the carbon intensity
    reward_carbon_intensity = jnp.clip(full_demand*env_variable['carbon_intensity'], 0, 1000000)

    # we compute the new electrical soc value
    # now we can compute the efficiency
    eff =  0.9110#get_efficiency(action*params['soc_max'])

    #print(f"eff: {eff}")

    # we compute the new electrical soc value
    if action >= 0:
      electrical_soc_new = min(electrical_soc + (action*eff), 1)
    else:
      electrical_soc_new = max(electrical_soc + (action/eff), 0)

    return reward_electricity_price + reward_carbon_intensity, electrical_soc_new

# we train the model with pytorch RNN or LSTM
# we define the model
class ModelCityLearn(pl.LightningModule):
    """
    A 2 layers model with 128 hidden units in each layer
    At each layers we have a LSTM layer
    The last layer is a linear layer

    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelCityLearn, self).__init__()
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        # loss function definition
        self.loss = nn.MSELoss()

    def forward(self, input, hidden=None):
        # reshape input (B, L, C) -> (B, C, L)
        output, _ = self.lstm1(input)

        # relu activation
        output = nn.functional.relu(output)

        output = self.linear_1(output)
        # relu 
        output = nn.functional.relu(output)

        output = self.linear(output)

        # we take the last value of the sequence
        output = output[:, -1, :]

        return output

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss(y_hat, y.float())

        # logs train_loss
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x.float())
        loss = self.loss(y_hat, y.float())

        # logs val_loss
        self.log('val_loss', loss)

        return loss

# we define the dataset
class DatasetCityLearn(torch.utils.data.Dataset):

    def __init__(self, data, lookback, features_to_forecast):
        self.data = data
        self.lookback = lookback
        self.features_to_forecast = features_to_forecast

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        # we get the data
        x = self.data[self.features_to_forecast].iloc[idx:(idx + self.lookback)].values
        y = self.data[self.features_to_forecast].iloc[idx + self.lookback].values
        return x, y

def autoregressive_model(model, x, horizon=20, lookback_official=5):
    """
    This function is used to make a prediction on the autoregressive model
    
    Args:
        model (torch.nn.Module): the model
        x (torch.Tensor): the input (B, L, C) where B is the batch size, L is the lookback and C is the number of features
        horizon (int): the horizon of the prediction
    """
    # we get the shape of the input
    batch_size, lookback, n_features = x.shape

    # if lookback is not equal to the lookback of the model then we return the last x values
    if lookback != lookback_official:

        return x[:, -1, :].detach().numpy()
    else:

        prediction_future = []

        for i in range(horizon):
            y_hat = model(x.float())

            prediction_future.append(y_hat)

            # we add the prediction to the input
            x = torch.cat([x, y_hat.unsqueeze(1)], dim=1)
            x = x[:, 1:, :]

        pred_futur = torch.cat(prediction_future, dim=0)

        return pred_futur.detach().numpy()


def full_simulation(actions_all, data, episode_size, episode_start, soc_init=0.):
    """
    This is a full simulation (whole year) of the jax_gym_simulation function
    """

    # we create the params
    params = {'soc_max': 6.4, 'nominal_power' : 5., 'soc_nominal' : 5./6.4}

    # we create the initial electrical soc
    electrical_soc = soc_init

    # we create the reward array
    rewards_tot = 0

    # we loop over the data
    for i in range(episode_start, episode_start + episode_size):

        # we get the env variable
        env_variable = data.iloc[i].to_dict()

        # we get the action
        actions = actions_all[i]

        # we compute the reward and the futur electrical soc
        reward, electrical_soc = jax_gym_simulation(env_variable, electrical_soc, actions, params)

        # we append the reward
        rewards_tot += reward

    #print("rewards_tot: ", rewards_tot)

    return rewards_tot

def optimize_action_imagination(prediction_futur, current_soc):

    lookforward = 20
    nb_iter = 10

    actions_all = jnp.zeros((lookforward,))

    # we create the optimizer
    optimizer = optax.adam(learning_rate=0.05)

    # we create the initial state of the optimizer
    state = optimizer.init(actions_all)

    for i in range(nb_iter):


        # we compute the loss and gradient (with respect to the actions)
        loss, grad = jax.value_and_grad(full_simulation)(actions_all, prediction_futur, lookforward, 0, soc_init=current_soc)

        # we update the state
        updates, state = optimizer.update(grad, state)

        # we update the actions
        actions_all = optax.apply_updates(actions_all, updates)

        # clip actions_all between -1 and 1
        actions_all = jnp.clip(actions_all, -1, 1)

        print("loss: ", loss)

    return actions_all[0]
  

features_to_forecast = ['non_shiftable_load', 'solar_generation', 'electricity_pricing', 'carbon_intensity',
                                                             'hour', 'month']

hidden_feature = 64
lookback = 5
lookforward = 20

class UserAgent:
    def __init__(self):

        self.action_space = {}

        # we init a history of information specific to the agent
        self.history = {}

        self.features_to_forecast = features_to_forecast

        # here we can load the model
        self.model = ModelCityLearn(len(features_to_forecast), hidden_feature, len(features_to_forecast))

        # we load the model from .pt file
        self.model.load_state_dict(torch.load('models/model_rnn.pt'))

        self.model.eval()

        # we load the mean and std
        with open('models/mean_std.pkl', 'rb') as f:
            self.mean, self.std = pickle.load(f)

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

        # we init a history of information specific to the agent
        self.history[agent_id] = {var : deque(maxlen=lookback) for var in self.features_to_forecast}

    def adding_observation(self, observation, agent_id):
        """Get observation return action"""

        # add observation to the history
        for var in self.features_to_forecast:
            self.history[agent_id][var].append(observation[var])
    
    def get_model_prediction(self, agent_id):
        """Get observation return action"""

        # we get the last 24*7 observations for every variables using history variable
        # we need to reshape the data to be able to use the model
        data = {var : list(self.history[agent_id][var]) for var in self.features_to_forecast}

        # we convert the data to a dataframe
        data = pd.DataFrame(data, columns=self.features_to_forecast)

        # we normalize the data
        data = (data - self.mean) / self.std

        # we convert the data to a tensor
        x = torch.tensor(data.values).unsqueeze(0)

        prediction_futur = autoregressive_model(self.model, x, lookforward)

        df_pred = pd.DataFrame(prediction_futur, columns=features_to_forecast)

        # we denormalize the data
        df_pred = df_pred * self.std + self.mean

        df_pred["solar_generation"][df_pred["solar_generation"] < 0] = 0
        
        return df_pred

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
            df_pred = self.get_model_prediction(agent_id)

            # we compute the action
            action = optimize_action_imagination(df_pred, current_soc)

            return [action]
