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


class Constants:
    episodes = 3
    schema_path = '/home/aicrowd/data/citylearn_challenge_2022_phase_1/schema.json'
    variables_to_forecast = ['solar_generation', 'non_shiftable_load', 'electricity_pricing', 'carbon_intensity', "electricity_consumption_crude",
                                     'hour', 'month']

    additional_variable = ['hour', "month"]


# create env from citylearn
env = CityLearnEnv(schema=Constants.schema_path)

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict

## env wrapper for stable baselines
class EnvCityGym(gym.Env):
    """
    Env wrapper coming from the gym library.
    """
    def __init__(self, env):
        self.env = env

        # get the number of buildings
        self.num_buildings = len(env.action_space)
        print("num_buildings: ", self.num_buildings)

        self.action_space = gym.spaces.Box(low=np.array([-0.2]), high=np.array([0.2]), dtype=np.float32)

        self.observation_space = gym.spaces.MultiDiscrete(np.array([25, 13]))

    def reset(self):
        obs_dict = env_reset(self.env)
        obs = self.env.reset()

        observation = [o for o in obs]

        return observation

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        obs, reward, done, info = self.env.step(action)

        observation = [o for o in obs]

        return observation, reward, done, info
        
    def render(self, mode='human'):
        return self.env.render(mode)




def env_run_without_action(actions_all=None):
    """
    This function is used to run the environment without applying any action.
    and return the dataset
    """
    # create env from citylearn
    env = CityLearnEnv(schema=Constants.schema_path)

    # get the number of buildings
    num_buildings = len(env.action_space)
    print("num_buildings: ", num_buildings)

    # create env wrapper
    env = EnvCityGym(env)

    # reset the environment
    obs = env.reset()

    infos = []

    for id_building in range(num_buildings):
        # run the environment
        obs = env.reset()

        for i in range(8759):

            info_tmp = env.env.buildings[id_building].observations.copy()

            if actions_all is not None:

                action = [[actions_all[i + 8759 * b]] for b in range(num_buildings)]

            else:
                # we get the action
                action = np.zeros((5, )) # 5 is the number of buildings

                # reshape action into form like [[0], [0], [0], [0], [0]]
                action = [[a] for a in action]

            #print(action)

            obs, reward, done, info = env.step(action)

            info_tmp['reward'] = reward[id_building]
            info_tmp['building_id'] = id_building
            infos.append(info_tmp)

            if done:
                obs = env.reset()

    # create the data
    data_pd = {}

    for info in infos:
        for i, v in info.items():
            try:
                data_pd[i].append(v)
            except:
                data_pd[i] = [v]

    data = pd.DataFrame(infos)

    return data

if __name__ == "__main__":

    # data generation
    data = env_run_without_action()

    # save the data into the data_histo folder into parquet format
    data.to_parquet("data_histo/data.parquet")

