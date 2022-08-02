"""
This script is used to train a policy to control the energy consumption of a city in order to minimize the cost of energy and emissions.
We train a policy using the CityLearn environment to simulate the energy consumption of a city and the emission
also we train the model using a genetic algorithm.
"""

from re import S
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import supersuit as ss
import gym


from citylearn.citylearn import CityLearnEnv
import numpy as np

class Constants:
    episodes = 3
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

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
        observation = [obs[0][2], obs[0][0]]

        return observation

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        action = np.array(action)
        action = action.reshape(1, -1)
        action = action.repeat(self.num_buildings, axis=0)
        obs, reward, done, info = self.env.step(action)

        reward = sum(reward)

        
        # we retrieve the building information with the hour and month from obs dict
        observation = [obs[0][2], obs[0][0]]

        return observation, reward, done, info
        
    def render(self, mode='human'):
        return self.env.render(mode)

# training with PPO and stable baselines
model = PPO(MlpPolicy, env=EnvCityGym(env), verbose=1)
model.learn(total_timesteps=1000000)
model.save("ppo_citylearn")









    