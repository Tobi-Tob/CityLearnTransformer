import os
import warnings

import numpy as np
import pickle
import time

from agents.one_action_agent import OneActionAgent
from agents.orderenforcingwrapper import OrderEnforcingAgent
from agents.random_agent import RandomAgent
from agents.rbc_agent import BasicRBCAgent, BetterRBCAgent

from utils import init_environment

"""
This file is used to generate offline data for a decision transformer.
Data is saved as pickle file.
Data structure:
list(
    dict(
        "observations": nparray(nparray(np.float32)),
        "next_observations": nparray(nparray(np.float32)),
        "actions": nparray(nparray(np.float32)),
        "rewards": nparray(np.oat32),
        "terminals": nparray(np.bool_)
        )
    )
"""


class Constants:
    file_prefix = "f"
    sequence_length = 50  # should be divisor of environment simulation steps
    episodes = 10
    state_dim = 28
    action_dim = 1

    buildings_to_use = [1, 2, 3, 4, 5]

    env = init_environment(buildings_to_use)

    # agent = RandomAgent()
    # agent = OneActionAgent([0])
    # agent = BasicRBCAgent()
    # agent = BetterRBCAgent()
    agent = OrderEnforcingAgent()

    print_sequences = False


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


def generate_data():
    print("========================= Start Data Collection ========================")

    env = Constants.env

    agent = Constants.agent

    dataset = []
    observation_data = []
    next_observation_data = []
    action_data = []
    reward_data = []
    done_data = []
    amount_buildings = len(env.buildings)
    print("==> Model:", agent.__class__.__name__)
    print("Amount of buildings:", amount_buildings)
    print("Buildings used:", Constants.buildings_to_use)
    start_timestep = env.schema['simulation_start_time_step']
    end_timestep = env.schema['simulation_end_time_step']
    print("Environment simulation from", start_timestep, "to", end_timestep)

    obs_dict = env_reset(env)
    observations = obs_dict["observation"]

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    sequences_completed = 0
    current_step_total = 0
    current_step_in_sequence = 0
    sequences_return = np.zeros(amount_buildings)
    interrupted = False
    episode_metrics = []

    try:
        while True:
            current_step_in_sequence += 1
            current_step_total += 1
            next_observations, reward, done, _ = env.step(actions)
            # ACTION [-1,1] attempts to decrease or increase the electricity stored in the battery by an amount
            # equivalent to action times its maximum capacity
            # Save environment interactions:
            observation_data.append(observations)
            next_observation_data.append(next_observations)
            action_data.append(actions)
            reward_data.append(reward)
            sequences_return += reward
            done_data.append(False)  # always False

            observations = next_observations  # observations of next time step

            if current_step_in_sequence >= Constants.sequence_length:  # Sequence completed
                current_step_in_sequence = 0
                sequences_completed += 1

                for bi in range(amount_buildings):
                    obs_building_i = np.zeros((Constants.sequence_length, Constants.state_dim), dtype=np.float32)
                    n_obs_building_i = np.zeros((Constants.sequence_length, Constants.state_dim), dtype=np.float32)
                    acts_building_i = np.zeros((Constants.sequence_length, Constants.action_dim), dtype=np.float32)
                    rwds_building_i = np.zeros(Constants.sequence_length, dtype=np.float32)
                    dones_building_i = np.zeros(Constants.sequence_length, dtype=np.bool_)
                    for ti in range(Constants.sequence_length):
                        obs_building_i[ti] = np.array(observation_data[ti][bi])
                        n_obs_building_i[ti] = np.array(next_observation_data[ti][bi])
                        acts_building_i[ti] = np.array(action_data[ti][bi])
                        rwds_building_i[ti] = reward_data[ti][bi]
                        dones_building_i[ti] = done_data[ti]

                    dict_building_i = {
                        "observations": obs_building_i,
                        "next_observations": n_obs_building_i,
                        "actions": acts_building_i,
                        "rewards": rwds_building_i,
                        "terminals": dones_building_i
                    }
                    dataset.append(dict_building_i)

                if Constants.print_sequences:
                    print("Sequence completed:", sequences_completed)
                    print("Sequence Return:", sequences_return)
                observation_data = []
                next_observation_data = []
                action_data = []
                reward_data = []
                done_data = []
                sequences_return = np.zeros(len(env.buildings))

            if done:
                episodes_completed += 1

                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1], "grid_cost": metrics_t[2]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contant organizers")
                episode_metrics.append(metrics)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

                obs_dict = env_reset(env)
                observations = obs_dict["observation"]

                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter() - step_start
            else:
                step_start = time.perf_counter()
                actions = agent.compute_action(next_observations)
                agent_time_elapsed += time.perf_counter() - step_start

            if current_step_total % 1000 == 0:
                print(f"Num Steps: {current_step_total}, Num episodes: {episodes_completed}")

            if episodes_completed >= Constants.episodes:
                break
    except KeyboardInterrupt:
        print("========================= Stopping Generation ==========================")
        interrupted = True

    if not interrupted:
        print("========================= Generation Completed =========================")

    print(f"Total time taken by agent: {agent_time_elapsed}s")
    print("Total number of steps:", current_step_total)
    if len(episode_metrics) > 0:
        price_cost = np.mean([e['price_cost'] for e in episode_metrics])
        emission_cost = np.mean([e['emmision_cost'] for e in episode_metrics])
        grid_cost = np.mean([e['grid_cost'] for e in episode_metrics])
        print("Average Price Cost:", price_cost)
        print("Average Emission Cost:", emission_cost)
        print("Average Grid Cost:", grid_cost)
        print("==> Score:", (price_cost + emission_cost + grid_cost) / 3)

    print("========================= Writing Data File ============================")

    longest_sequence_length = 0
    shortest_sequence_length = float('inf')
    for data in dataset:
        if len(data["observations"]) > longest_sequence_length:
            longest_sequence_length = len(data["observations"])
        if len(data["observations"]) < shortest_sequence_length:
            shortest_sequence_length = len(data["observations"])

    print("Amount Of Sequences: ", len(dataset))
    print("Longest Sequence: ", longest_sequence_length)
    print("Shortest Sequence: ", shortest_sequence_length)

    test = len(dataset) - (amount_buildings * sequences_completed)
    if test != 0:
        warnings.warn(str(len(dataset)) + " != " + str(amount_buildings) + "*" + str(sequences_completed))
    total_values = (2 * Constants.state_dim + Constants.action_dim + 2) * longest_sequence_length * len(dataset)

    print("Total values to store: ", total_values)

    ''' Format: [SEQUENCE_LENGTH] x [AMOUNT_BUILDINGS] x [AMOUNT_SEQUENCES] '''
    file_info = "_" + str(longest_sequence_length) + "x" + str(amount_buildings) + "x" + str(sequences_completed)
    file_extension = ".pkl"
    file_name = Constants.file_prefix + file_info + file_extension
    file_path = "./data/" + file_name

    # create or overwrite pickle file
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)

    print("========================= Writing Completed ============================")
    file_size = os.stat(file_path).st_size
    if file_size > 1e+6:
        string_byte = "(" + str(round(file_size / 1e+6)) + " MB)"
    else:
        string_byte = "(" + str(round(file_size / 1e+3)) + " kB)"
    print("==> Data saved in", file_name, string_byte)


if __name__ == '__main__':
    generate_data()
