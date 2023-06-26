import pathlib
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as st

from utils import init_environment
from agents.random_agent import RandomAgent
from agents.one_action_agent import OneActionAgent
from agents.rbc_agent import BasicRBCAgent, RBCAgent1, RBCAgent2
from agents.orderenforcingwrapper import OrderEnforcingAgent

"""
This file is used to plot data
"""


class Constants:
    episodes = 1
    state_dim = 28  # size of state space
    action_dim = 1  # size of action space
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    buildings_to_use = [1, 2, 3, 4, 5]

    env = init_environment(buildings_to_use)

    # agent = RandomAgent()
    agent = OneActionAgent([0])
    # agent = BasicRBCAgent()
    # agent = RBCAgent1()
    # agent = RBCAgent2()
    # agent = OrderEnforcingAgent()


str_to_int_mapping = {
    "month": 0,
    "day_type": 1,
    "hour": 2,
    "outdoor_dry_bulb_temperature": 3,
    "outdoor_dry_bulb_temperature_predicted_6h": 4,
    "outdoor_dry_bulb_temperature_predicted_12h": 5,
    "outdoor_dry_bulb_temperature_predicted_24h": 6,
    "outdoor_relative_humidity": 7,
    "outdoor_relative_humidity_predicted_6h": 8,
    "outdoor_relative_humidity_predicted_12h": 9,
    "outdoor_relative_humidity_predicted_24h": 10,
    "diffuse_solar_irradiance": 11,  # Sonneneinstrahlung (aus verschiedenen Richtungen gestreut durch Nebel/Wolken)
    "diffuse_solar_irradiance_predicted_6h": 12,
    "diffuse_solar_irradiance_predicted_12h": 13,
    "diffuse_solar_irradiance_predicted_24h": 14,
    "direct_solar_irradiance": 15,  # Sonneneinstrahlung (direkt)
    "direct_solar_irradiance_predicted_6h": 16,
    "direct_solar_irradiance_predicted_12h": 17,
    "direct_solar_irradiance_predicted_24h": 18,
    "carbon_intensity": 19,  # current carbon intensity of the power grid
    "non_shiftable_load": 20,  # electricity currently consumed by electrical devices in kWh
    "solar_generation": 21,
    "electrical_storage_soc": 22,  # state of charge of the electrical storage device
    "net_electricity_consumption": 23,  # current net electricity consumption of the building
    "electricity_pricing": 24,
    "electricity_pricing_predicted_6h": 25,
    "electricity_pricing_predicted_12h": 26,
    "electricity_pricing_predicted_24h": 27,
}


def action_space_to_dict(aspace):
    """ Only for box space """
    return {"high": aspace.high,
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
                "observation": observations}
    return obs_dict


def run_env_and_visualize():
    env = init_environment([1, 2, 3, 4, 5])
    agent = Constants.agent

    obs_dict = env_reset(env)
    # print(obs_dict['building_info'])

    actions = agent.register_reset(obs_dict)

    t = 0
    y = []
    y_min = []
    y_max = []
    building_to_visualize = 0  # None to visualize mean of all buildings
    data_to_visualize = str_to_int_mapping["solar_generation"]

    while True:
        observations, reward, done, _ = env.step(actions)
        if building_to_visualize is None:
            y_all_buildings = [row[data_to_visualize] for row in observations]
            y_t_min, y_t_max = min(y_all_buildings), max(y_all_buildings)
            y.append(y_all_buildings)
            y_min.append(y_t_min)
            y_max.append(y_t_max)
        else:
            y.append(observations[building_to_visualize][data_to_visualize])

        t += 1
        if done:
            break
        else:
            actions = agent.compute_action(observations)
        if t % 1000 == 0:
            print(f"t: {t}")

    x = range(t)

    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    plt.xticks([1, 745, 1465, 2209, 2929, 3673, 4417, 5089, 5833, 6553, 7297, 8017],
               ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'])
    if building_to_visualize is None:
        mean_y = np.mean(y, axis=1)
        ax.plot(x, mean_y, '-', color='g')
        plt.fill_between(np.arange(len(mean_y)), y_min, y_max, alpha=0.5)
    else:
        ax.plot(x, y, '-', color='g')
    ax.set_xlabel('Time step [hourly]')
    ax.set_ylabel('Solar Generation [W/kW]')
    plt.show()


if __name__ == '__main__':
    run_env_and_visualize()
