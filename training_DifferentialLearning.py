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

import jax
import jax.numpy as jnp
import optax

import joblib
import xgboost as xgb

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

def get_efficiency(energy):

    power_efficiency_curve = np.array([[0.  , 0.3 , 0.7 , 0.8 , 1.  ],
        [0.83, 0.83, 0.9 , 0.9 , 0.85]])

    nominal_power = 5.
    efficiency_scaling = 0.5

    # Calculating the maximum power rate at which the battery can be charged or discharged
    energy_normalized = np.abs(energy)/nominal_power
    idx = max(0, np.argmax(energy_normalized <= power_efficiency_curve[0]) - 1)
    efficiency = power_efficiency_curve[1][idx]\
        + (energy_normalized - power_efficiency_curve[0][idx]
        )*(power_efficiency_curve[1][idx + 1] - power_efficiency_curve[1][idx]
        )/(power_efficiency_curve[0][idx + 1] - power_efficiency_curve[0][idx])
    efficiency = efficiency**efficiency_scaling

    return efficiency


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
    eff = get_efficiency(action*params['soc_max'])

    # we compute the new electrical soc value
    if action >= 0:
      electrical_soc_new = min(electrical_soc + (action*eff), 1)
    else:
      electrical_soc_new = max(electrical_soc + (action/eff), 0)

    return reward_electricity_price + reward_carbon_intensity, electrical_soc_new

def full_simulation(actions_all, data, episode_size, episode_start):
    """
    This is a full simulation (whole year) of the jax_gym_simulation function
    """

    # we create the params
    params = {'soc_max': 6.4, 'nominal_power' : 5., 'soc_nominal' : 5/6.4}

    # we create the initial electrical soc
    electrical_soc = 0.0

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

    print("rewards_tot: ", rewards_tot)

    return rewards_tot

def optimize():
    """
    This function is used to optimize the actions using a differential approach
    """

    # if data not saved, we create it
    if not os.path.exists("data.parquet"):

        # we load the data
        data = env_run_without_action()

        # save data to parquet file
        data.to_parquet('data.parquet')
    
    else:
        data = pd.read_parquet('data.parquet')

        data = data.reset_index(drop=True)

    tot_len = len(data) # 8759*5 = 43795

    episode_size = 2000
    
    # we start episodes between 0 and 43795 at every 2000 steps
    episodes_start_list = [i for i in range(0, tot_len, episode_size)]

    # we create the initial actions
    actions_all = np.zeros((tot_len,))

    # we create the optimizer
    optimizer = optax.adam(learning_rate=0.05)

    # we create the initial state of the optimizer
    state = optimizer.init(actions_all)

    for episode_start in episodes_start_list:
        
        print("episode_start: ", episode_start)

        # we loop over the optimization
        for i in range(20):

            # compute exact episode_size
            if episode_start + episode_size > tot_len:
                episode_size = tot_len - episode_start

            # we compute the loss and gradient (with respect to the actions)
            loss, grad = jax.value_and_grad(full_simulation)(actions_all, data, episode_size, episode_start)

            # we update the state
            updates, state = optimizer.update(grad, state)

            # we update the actions
            actions_all = optax.apply_updates(actions_all, updates)

            # clip actions_all between -1 and 1
            actions_all = np.clip(actions_all, -1, 1)

            print("loss: ", loss)
        
        print(actions_all.shape)

    return actions_all, data

def learn_optimal_actions(actions_all, data):
    """
    This function is used to learn the optimal actions
    """

    # we create the features
    features = ['non_shiftable_load', 'solar_generation', 'electricity_pricing', 'carbon_intensity',
                                                             'hour', 'month', 'electrical_storage_soc'] # simple features

    data = env_run_without_action(actions_all=actions_all)

    data_learn = data.copy()
    data_learn['action_optimal'] = actions_all

    # we create the target
    data_learn['action_optimal_t1'] = data_learn['action_optimal'].shift(-1)

    # we drop the last row
    data_learn = data_learn[:-1]

    # we create the model (xgboost)
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 50)

    # we fit the model
    model.fit(data_learn[features], data_learn['action_optimal_t1'])

    # we save the model
    joblib.dump(model, 'models/model_calibration_optimal.joblib')

    return model

def test_jaxgym_env_raw_optimal_action(actions_all, data, episode_size, episode_start):

    return full_simulation(actions_all, data, episode_size, episode_start)


def testing_raw_optimal_action(actions_all):

    # create env from citylearn
    env = CityLearnEnv(schema=Constants.schema_path)
    env = EnvCityGym(env)

    nb_buildings = len(env.env.buildings)

    # reset the environment
    obs = env.reset()

    reward_tot = 0

    # run the environment
    for i in range(8759):

        # we get the action
        try:
            actions = [[actions_all[i + 1 + b*8759]] for b in range(nb_buildings)]
        except:
            actions = [[0] for b in range(nb_buildings)]

        # we get the observation
        obs, reward, done, info = env.step(actions)
        reward_tot += reward[0]

    # compute the final performance
    global_reward = sum(env.env.evaluate()[:2])/2

    print("Computing the reward with the actions (evaluation): ", global_reward)
    print("Computing the reward with the actions (sum reward): ", reward_tot)

def testing_model_performance_gym(model):
    """
    We test the model using the gym env setup
    """
    features = ['non_shiftable_load', 'solar_generation', 'electricity_pricing', 'carbon_intensity',
                                                             'hour', 'month', 'electrical_storage_soc']
    # create env from citylearn
    env = CityLearnEnv(schema=Constants.schema_path)
    env = EnvCityGym(env)

    nb_buildings = len(env.env.buildings)

    # reset the environment
    obs = env.reset()

    # run the environment
    for i in range(8759):

        # we get the 5 observation
        observations = [env.env.buildings[b].observations for b in range(nb_buildings)]

        # we create the dataframe
        df = pd.DataFrame(observations)

        df = df[features]

        # we get the action
        action = model.predict(df.values.reshape(nb_buildings, -1))

        # reshape the action
        action = [[act] for act in action]

        # we step in the environment
        obs, reward, done, info = env.step(action)

    # compute the final performance
    global_reward = sum(env.env.evaluate()[:2])/2

    print("global_reward with the learned actions: ", global_reward)

    return global_reward

def retrieve_train_eval():
    """
    This function is used to retrieve the historical data from the environment.
    """

    # use argparse
    parser = argparse.ArgumentParser()

    # retrieve training mode
    parser.add_argument('--train', help='train the model', default="true")

    # retrieve optimization mode
    parser.add_argument('--optimize', help='optimize the actions', default="true")

    # get training mode
    args = parser.parse_args()

    # get training mode from args
    train = args.train
    
    logger.info("Testing first differential optimization")

    # if we want to optimize the actions
    if args.optimize == "true":
        actions_all, data = optimize()

        data['action_optimal'] = actions_all

        # we save the data
        data.to_parquet('data_optimal.parquet')

    else:

        # we load the data
        data = pd.read_parquet('data_optimal.parquet')

        # we load the actions
        actions_all = data['action_optimal'].to_numpy()

        print(data)

    print(data.describe())

    logger.info("Testing the raw optimal actions according to jaxgym")

    # we test the raw optimal actions according to the jax optimization
    reward_optim = test_jaxgym_env_raw_optimal_action(actions_all, data, 8759, 0)

    actions_nooptim = np.zeros((8759*5,))
    reward_nooptim = test_jaxgym_env_raw_optimal_action(actions_nooptim, data, 8759, 0)


    print("reward_optim: ", reward_optim)
    print("reward_nooptim: ", reward_nooptim)

    # test optimal actions
    testing_raw_optimal_action(actions_all)
    testing_raw_optimal_action(actions_nooptim)

    logger.info("Comparaison between jax gym and real gym env")

    # construct dataset of optimal actions and learn model
    if train == "true":
        
        logger.info("Learning optimal actions")

        model = learn_optimal_actions(actions_all, data)

    # now we want to test the model
    logger.info("Testing the model")

    testing_model_performance_gym(model)    

    logger.info("End of the test")

    
if __name__ == "__main__":
    retrieve_train_eval()
    

