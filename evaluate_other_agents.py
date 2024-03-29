import pathlib
import sys
import numpy as np
import time

from utils import init_environment
from agents.random_agent import RandomAgent
from agents.one_action_agent import OneActionAgent
from agents.rbc_agent import BasicRBCAgent, RBCAgent1, RBCAgent2
from agents.orderenforcingwrapper import OrderEnforcingAgent

"""
This file is used to evaluate agents on the CityLearn environment
"""


class Constants:
    file_to_save = '_results.txt'
    episodes = 1
    state_dim = 28  # size of state space
    action_dim = 1  # size of action space
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    buildings_to_use = [1, 2, 3, 4, 5]

    env = init_environment(buildings_to_use)

    # agent = RandomAgent()
    # agent = OneActionAgent([0])
    # agent = BasicRBCAgent()
    # agent = RBCAgent1()
    agent = RBCAgent2()
    # agent = OrderEnforcingAgent()

    print_interactions = False


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


def evaluate():
    print("========================= Start Evaluation ========================")

    env = Constants.env
    obs_dict = env_reset(env)

    agent = Constants.agent
    print("==> Model:", agent.__class__.__name__)

    if isinstance(agent, OneActionAgent):
        print("Action to perform:", agent.action_to_perform)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    amount_buildings = len(env.buildings)
    print("Amount of buildings:", amount_buildings)
    episode_return = np.zeros(amount_buildings)
    episode_metrics = []
    interrupted = False
    error = 0

    original_stdout = sys.stdout
    with open(Constants.file_to_save, 'w') as f:
        sys.stdout = f
        print("==> Model:", agent.__class__.__name__)
        if isinstance(agent, OneActionAgent):
            print("Action to perform:", agent.action_to_perform)
        start_timestep = env.schema['simulation_start_time_step']
        end_timestep = env.schema['simulation_end_time_step']
        print("Environment simulation from", start_timestep, "to", end_timestep)
        print("Buildings used:", Constants.buildings_to_use)
        sys.stdout = original_stdout
        try:
            while True:

                observations, reward, done, _ = env.step(actions)
                if Constants.print_interactions and 0 <= observations[0][2] <= 24:
                    print(num_steps, "Hour", observations[0][2])
                    action_array = np.array([actions[0][0], actions[1][0], actions[2][0], actions[3][0], actions[4][0]])
                    print("Action", action_array)
                    print("Reward", reward)
                    # solar_generation_array = np.array([observations[0][21], observations[1][21], observations[2][21], observations[3][21], observations[4][21]])
                    # load_array = np.array([observations[0][20], observations[1][20], observations[2][20], observations[3][20], observations[4][20]])
                    # solar_generation_surplus_array = solar_generation_array - load_array
                    net_electricity_array = np.array([observations[0][23], observations[1][23], observations[2][23], observations[3][23], observations[4][23]])
                    print("Consume", net_electricity_array)
                    storage_array = np.array([observations[0][22], observations[1][22], observations[2][22], observations[3][22], observations[4][22]])
                    print("Storage", storage_array)
                    # pricing_array = np.array([observations[0][24], observations[1][24], observations[2][24], observations[3][24], observations[4][24]])
                    # print("Pricing", storage_array)
                    # error_array = np.absolute((action_array - pricing_array)**1)
                    # print("Error", error_array)
                    # error += sum(error_array)

                episode_return += reward
                if done:
                    episodes_completed += 1
                    metrics_t = env.evaluate()
                    metrics = {"price_cost": metrics_t[0]*100, "emmision_cost": metrics_t[1]*100, "grid_cost": metrics_t[2]*100}
                    if np.any(np.isnan(metrics_t)):
                        raise ValueError("Episode metrics are nan, please contant organizers")
                    episode_metrics.append(metrics)
                    print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )
                    sys.stdout = f
                    print(episodes_completed, episode_return)
                    print("Reward mean:", sum(episode_return) / amount_buildings)
                    sys.stdout = original_stdout

                    obs_dict = env_reset(env)
                    episode_return = np.zeros(amount_buildings)

                    step_start = time.perf_counter()
                    actions = agent.register_reset(obs_dict)
                    agent_time_elapsed += time.perf_counter() - step_start
                else:
                    step_start = time.perf_counter()
                    actions = agent.compute_action(observations)
                    agent_time_elapsed += time.perf_counter() - step_start

                num_steps += 1
                if num_steps % 1000 == 0:
                    print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

                if episodes_completed >= Constants.episodes:
                    break
        except KeyboardInterrupt:
            print("========================= Stopping Evaluation ==========================")
            interrupted = True

        if not interrupted:
            print("========================= Evaluation Done ========================")

        print(f"Total time taken by agent: {agent_time_elapsed}s")
        sys.stdout = f
        print(f"Total time taken by agent: {agent_time_elapsed}s")
        print("Total number of steps:", num_steps)
        if len(episode_metrics) > 0:
            price_cost = np.mean([e['price_cost'] for e in episode_metrics])
            emission_cost = np.mean([e['emmision_cost'] for e in episode_metrics])
            grid_cost = np.mean([e['grid_cost'] for e in episode_metrics])
            print("Average Price Cost:", price_cost)
            print("Average Emission Cost:", emission_cost)
            print("Average Grid Cost:", grid_cost)
            print("==> Score:", (price_cost + emission_cost + grid_cost) / 3)
            sys.stdout = original_stdout
            print("==> Score:", (price_cost + emission_cost + grid_cost) / 3)
        print("Evaluation saved in:", str(pathlib.Path(__file__).parent.resolve()) + '/' + Constants.file_to_save)

        # print(error)


if __name__ == '__main__':
    evaluate()
