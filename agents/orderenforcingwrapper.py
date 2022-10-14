from gym.spaces import Box

from rewards.user_reward import UserReward

from collections import deque
import torch
import torch.nn as nn
import pytorch_lightning as pl

import pickle

def dict_to_action_space(aspace_dict):
    return Box(
                low = aspace_dict["low"],
                high = aspace_dict["high"],
                dtype = aspace_dict["dtype"],
              )


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


import json
from .models import ModelCityLearn, ModelAfterPrediction, ModelCityLearnOptim, DatasetCityLearn

lookback = 5
lookfuture = 20

features_to_forecast = ['non_shiftable_load', 'solar_generation', 'electricity_pricing', 'carbon_intensity',
                                                             'hour', 'month']

hidden_feature = 64

non_shiftable_load_idx = features_to_forecast.index("non_shiftable_load")
solar_generation_idx = features_to_forecast.index("solar_generation")
electricity_pricing_idx = features_to_forecast.index("electricity_pricing")
carbon_intensity_idx = features_to_forecast.index("carbon_intensity") 
hour_idx = features_to_forecast.index("hour")
month_idx = features_to_forecast.index("month")

features_index = [str_to_int_mapping[feature] for feature in features_to_forecast]

class OrderEnforcingAgent:
    """
    Emulates order enforcing wrapper in Pettingzoo for easy integration
    Calls each agent step with agent in a loop and returns the action
    """
    def __init__(self):
        self.num_buildings = None
        self.action_space = None

        # now we load mean and std from models_checkpoint/mean_std.pkl
        with open("models_checkpoint/mean_std.pkl", "rb") as f:
            self.mean, self.std = pickle.load(f)


        # here we load the model
        self.model = ModelCityLearnOptim(len(features_to_forecast), hidden_feature,
                        len(features_to_forecast), lookback, lookfuture, self.mean, self.std)

        self.model.load_state_dict(torch.load("models_checkpoint/model_world_v3.pt"))


    
    def register_reset(self, observation):
        """Get the first observation after env.reset, return action""" 
        action_space = observation["action_space"]
        self.action_space = [dict_to_action_space(asd) for asd in action_space]
        obs = observation["observation"]
        self.num_buildings = len(obs)

        # observation history
        self.observation_history = [deque(maxlen=lookback) for _ in range(self.num_buildings)]

        return self.compute_action(obs)
    
    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def update_observation_history(self, observation):
        for i in range(self.num_buildings):

            # we rework observation[i] to get a list of features_to_forecast
            obs_building = [observation[i][str_to_int_mapping[feature]] for feature in features_to_forecast]

            self.observation_history[i].append(obs_building)

    def retrieve_observation_history(self):
        """
        This function get the observation history
        and create an array of shape (num_buildings, lookback, len(features_to_forecast))
        of torch.tensor type
        """

        obs_history = torch.zeros((self.num_buildings, lookback, len(features_to_forecast)))

        for i in range(self.num_buildings):
    
            # get an obs_lookback_building by retrieving the right index of
            # self.observation_history[i][j] corresponding to the features_to_forecast
            obs_lookback_building = list(self.observation_history[i])

            obs_history[i, :, :] = torch.Tensor(obs_lookback_building)

        return obs_history

    def get_storage_value(self, observation):
        """
        This function get the storage value
        """
        storage_value = torch.zeros((self.num_buildings, 1))

        for i in range(self.num_buildings):
            storage_value[i] = observation[i][str_to_int_mapping["electrical_storage_soc"]]

        return storage_value

    def compute_action(self, observation):
        """
        Inputs: 
            observation - List of observations from the env
        Returns:
            actions - List of actions in the same order as the observations

        You can change this function as needed
        please make sure the actions are in same order as the observations

        Reward preprocesing - You can use your custom reward function here
        please specify your reward function in agents/user_agent.py

        """
        assert self.num_buildings is not None
        rewards = UserReward(agent_count=len(observation),observation=observation).calculate()

        # here we update the observation list
        self.update_observation_history(observation)

        # we check if we have enough observation to make a prediction
        if len(self.observation_history[0]) < lookback:
            return [self.action_space[i].sample() for i in range(self.num_buildings)]

        # here we retrieve the observation history
        obs_history = self.retrieve_observation_history()
        obs_history = obs_history.unsqueeze(0)

        # we get the two other variables net_demand_old and storage value
        net_demand_old = obs_history[:, :, -1,  non_shiftable_load_idx] - obs_history[:, :, -1, solar_generation_idx] 

        # we get the storage value from the last observation
        storage_value = self.get_storage_value(observation)

        with torch.no_grad():
            # here we get the action from the model
            
            # we normalize obs_history
            std_torch = torch.tensor(self.std).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            mean_torch = torch.tensor(self.mean).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

            obs_history = (obs_history - mean_torch) / std_torch

            net_demand_old = obs_history[:, :, -1,  non_shiftable_load_idx] - obs_history[:, :, -1, solar_generation_idx] 


            actions, futur_state = self.model(obs_history, net_demand_old, storage_value)

            actions = actions[:, :, 0, 0]

        # actions is of shape (1, num_buildings, 1) transform it to (num_buildings)
        actions = actions.squeeze(0)

        # here we clip the actions to be between -1 and 1
        actions = torch.clamp(actions, -1, 1)

        # now we write the actions to have a form of type [[0.], [0.]]
        actions = actions.unsqueeze(1).tolist()

        return actions