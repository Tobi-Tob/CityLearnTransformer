import pathlib
import sys

import numpy as np
import time
from citylearn.citylearn import CityLearnEnv

from agents.random_agent import RandomAgent
from agents.one_action_agent import OneActionAgent
from agents.rbc_agent import BasicRBCAgent, BetterRBCAgent
from agents.orderenforcingwrapper import OrderEnforcingAgent

"""
This file is used to evaluate agents on the CityLearn environment
"""


class Constants:
    file_to_save = 'evaluation_results.txt'
    episodes = 1
    state_dim = 28  # size of state space
    action_dim = 1  # size of action space
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'
    env = CityLearnEnv(schema_path)

    # agent = RandomAgent()
    # agent = OneActionAgent([0])
    # agent = BasicRBCAgent()
    agent = BetterRBCAgent()
    # agent = OrderEnforcingAgent()


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

    agent = Constants.agent
    print("==> Model:", agent)

    if isinstance(agent, OneActionAgent):
        print("Action to perform:", agent.action_to_perform)

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    amount_buildings = len(env.buildings)
    episode_return = np.zeros(amount_buildings)
    episode_metrics = []
    interrupted = False

    original_stdout = sys.stdout
    with open(Constants.file_to_save, 'w') as f:
        sys.stdout = f
        print("==> Model:", agent)
        if isinstance(agent, OneActionAgent):
            print("Action to perform:", agent.action_to_perform)
        start_timestep = env.schema['simulation_start_time_step']
        end_timestep = env.schema['simulation_end_time_step']
        print("Environment simulation from", start_timestep, "to", end_timestep)
        sys.stdout = original_stdout
        try:
            while True:

                observations, reward, done, _ = env.step(actions)

                episode_return += reward
                if done:
                    episodes_completed += 1
                    metrics_t = env.evaluate()
                    metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1], "grid_cost": metrics_t[2]}
                    if np.any(np.isnan(metrics_t)):
                        raise ValueError("Episode metrics are nan, please contant organizers")
                    episode_metrics.append(metrics)
                    print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )
                    sys.stdout = f
                    print(episodes_completed, episode_return)
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

        sys.stdout = f
        print("Total number of steps:", num_steps)
        if len(episode_metrics) > 0:
            price_cost = np.mean([e['price_cost'] for e in episode_metrics])
            emission_cost = np.mean([e['emmision_cost'] for e in episode_metrics])
            grid_cost = np.mean([e['grid_cost'] for e in episode_metrics])
            print("Average Price Cost:", price_cost)
            print("Average Emission Cost:", emission_cost)
            print("Average Grid Cost:", grid_cost)
            print("==>", (price_cost + emission_cost + grid_cost) / 3)
            sys.stdout = original_stdout
            print("==>", (price_cost + emission_cost + grid_cost) / 3)
        print(f"Total time taken by agent: {agent_time_elapsed}s")
        print("Evaluation saved in:", str(pathlib.Path(__file__).parent.resolve()) + '/' + Constants.file_to_save)


if __name__ == '__main__':
    evaluate()
