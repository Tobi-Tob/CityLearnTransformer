import math
import os
import pathlib
import warnings

import numpy as np
import time
import torch
from MyDecisionTransformer import MyDecisionTransformer
from citylearn.citylearn import CityLearnEnv
import sys

from utils import init_environment

"""
This file is used to evaluate a decision transformer loaded form https://huggingface.co/TobiTob/model_name
"""


class Constants:
    file_to_save = '_results.txt'
    """Environment Constants"""
    episodes = 1  # amount of environment resets
    state_dim = 28  # size of state space
    action_dim = 1  # size of action space

    #  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    buildings_to_use = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    env = init_environment(buildings_to_use)

    """Model Constants"""

    load_model = "TobiTob/decision_transformer_random_230"
    TARGET_RETURN = -9000
    scale = 1000.0  # normalization for rewards/returns
    trained_sequence_length = 230
    force_download = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # mean and std computed from training dataset these are available in the model card for each model.
    state_mean = np.array(
        [6.5249427917620135, 3.9993135011441647, 12.49771167048055, 16.825446250727847, 16.82580094184701,
         16.828741445148562, 16.828180804459944, 72.99828375286042, 73.0012585812357, 72.99359267734553,
         73.00102974828376, 208.00308924485125, 208.0545766590389, 208.2866132723112, 207.94530892448512,
         201.11270022883295, 201.12254004576658, 201.15926773455377, 200.98135011441647, 0.15644143459640733,
         1.064985996011147, 0.6985259305737326, 0.5077011139382186, 0.49160775358187564, 0.27306979145658644,
         0.27306979145658644, 0.27306979145658644, 0.27306979145658644])
    state_std = np.array([3.4517203419362, 2.000572882797276, 6.924445762360648, 3.5581132080274425, 3.558410500805662,
                          3.563460666518717, 3.562742154586059, 16.491663737463313, 16.493405084016068,
                          16.495564654312346, 16.49694264406781, 292.5675403707197, 292.54446787504037,
                          292.792528944882, 292.55912445362566, 296.2258939141665, 296.2202986371211, 296.2051386297462,
                          296.1393568330303, 0.03533480921331586, 0.8881741764856719, 1.0167875215772866,
                          0.40298540713872427, 2.6547702496117305, 0.11773822184102951, 0.11773822184102949,
                          0.1177382218410294, 0.11773822184102911])

    start_timestep = env.schema['simulation_start_time_step']
    end_timestep = env.schema['simulation_end_time_step']
    total_time_steps = end_timestep - start_timestep


def preprocess_states(state_list_of_lists, amount_buildings):
    for bi in range(amount_buildings):
        for si in range(Constants.state_dim):
            state_list_of_lists[bi][si] = (state_list_of_lists[bi][si] - Constants.state_mean[si]) / \
                                          Constants.state_std[si]

    return state_list_of_lists


def calc_sequence_target_return(return_to_go_list, num_steps_in_episode):
    timesteps_left = Constants.total_time_steps - num_steps_in_episode
    target_returns_for_next_sequence = []
    for bi in range(len(return_to_go_list)):
        required_reward_per_timestep = return_to_go_list[bi] / timesteps_left
        if timesteps_left < Constants.trained_sequence_length:
            target_returns_for_next_sequence.append(required_reward_per_timestep * timesteps_left / Constants.scale)
        else:
            target_returns_for_next_sequence.append(
                required_reward_per_timestep * Constants.trained_sequence_length / Constants.scale)
    return target_returns_for_next_sequence


def evaluate():
    print("========================= Start Evaluation ========================")
    # Check current working directory.
    retval = os.getcwd()
    print("Current working directory %s" % retval)

    env = Constants.env

    agent = MyDecisionTransformer(load_from=Constants.load_model, force_download=Constants.force_download,
                                  device=Constants.device)
    print("Using device:", Constants.device)
    print("==> Model:", Constants.load_model)

    context_length = agent.model.config.max_length
    amount_buildings = len(env.buildings)

    print("Target Return:", Constants.TARGET_RETURN)
    print("Context Length:", context_length)
    print("Trained Sequence Length:", Constants.trained_sequence_length)
    start_timestep = env.schema['simulation_start_time_step']
    end_timestep = env.schema['simulation_end_time_step']
    print("Environment simulation from", start_timestep, "to", end_timestep)
    print("Amount of buildings:", amount_buildings)

    warnings.warn("Correct mean and std?")

    episodes_completed = 0
    sequences_completed = 0
    num_steps_total = 0
    num_steps_in_episode = 0
    num_steps_in_sequence = 0
    agent_time_elapsed = 0
    episode_metrics = []

    return_to_go_list = [Constants.TARGET_RETURN] * amount_buildings
    target_returns_for_next_sequence = calc_sequence_target_return(return_to_go_list, num_steps_in_episode)

    # Initialize Tensors
    state_list_of_lists = env.reset()
    state_list_of_lists = preprocess_states(state_list_of_lists, amount_buildings)

    state_list_of_tensors = []
    target_return_list_of_tensors = []
    action_list_of_tensors = []
    reward_list_of_tensors = []

    episode_return = np.zeros(amount_buildings)

    for bi in range(amount_buildings):
        state_bi = torch.from_numpy(np.array(state_list_of_lists[bi])).reshape(1, Constants.state_dim).to(
            device=Constants.device,
            dtype=torch.float32)
        target_return_bi = torch.tensor(target_returns_for_next_sequence[bi], device=Constants.device,
                                        dtype=torch.float32).reshape(1, 1)
        action_bi = torch.zeros((0, Constants.action_dim), device=Constants.device, dtype=torch.float32)
        reward_bi = torch.zeros(0, device=Constants.device, dtype=torch.float32)

        state_list_of_tensors.append(state_bi)
        target_return_list_of_tensors.append(target_return_bi)
        action_list_of_tensors.append(action_bi)
        reward_list_of_tensors.append(reward_bi)

    timesteps = torch.tensor(0, device=Constants.device, dtype=torch.long).reshape(1, 1)
    # print(state_list_of_tensors) Liste mit 5 Tensoren, jeder Tensor enth??lt einen State s der L??nge 28
    # print(action_list_of_tensors) Liste mit 5 leeren Tensoren mit size (0,1)
    # print(reward_list_of_tensors) Liste mit 5 leeren Tensoren ohne size
    # print(target_return_list_of_tensors) Liste mit 5 leeren Tensoren, jeder Tensor enth??lt den target_return / scale
    # print(timesteps) enth??lt einen Tensor mit 0: tensor([[0]])

    original_stdout = sys.stdout
    with open(Constants.file_to_save, 'w') as f:
        sys.stdout = f
        print("==> Model:", Constants.load_model)
        print("Target Return:", Constants.TARGET_RETURN)
        print("Context Length:", context_length)
        print("Trained Sequence Length:", Constants.trained_sequence_length)
        start_timestep = env.schema['simulation_start_time_step']
        end_timestep = env.schema['simulation_end_time_step']
        print("Environment simulation from", start_timestep, "to", end_timestep)
        print("Buildings used:", Constants.buildings_to_use)
        sys.stdout = original_stdout
        if end_timestep - start_timestep >= 4096:
            warnings.warn("Evaluation steps are over 4096")

        while True:
            if num_steps_in_sequence >= Constants.trained_sequence_length:  # if Sequence complete
                sequences_completed += 1
                num_steps_in_sequence = 0

                target_returns_for_next_sequence = calc_sequence_target_return(return_to_go_list, num_steps_in_episode)

                #  Reset History and only save last state
                last_state_list_of_tensor = []
                target_return_list_of_tensors = []
                action_list_of_tensors = []
                reward_list_of_tensors = []

                for bi in range(amount_buildings):
                    state_bi = state_list_of_tensors[bi][-1:]
                    target_return_bi = torch.tensor(target_returns_for_next_sequence[bi], device=Constants.device,
                                                    dtype=torch.float32).reshape(1, 1)
                    action_bi = torch.zeros((0, Constants.action_dim), device=Constants.device, dtype=torch.float32)
                    reward_bi = torch.zeros(0, device=Constants.device, dtype=torch.float32)

                    last_state_list_of_tensor.append(state_bi)
                    target_return_list_of_tensors.append(target_return_bi)
                    action_list_of_tensors.append(action_bi)
                    reward_list_of_tensors.append(reward_bi)

                state_list_of_tensors = last_state_list_of_tensor

                timesteps = torch.tensor(0, device=Constants.device, dtype=torch.long).reshape(1, 1)

            next_actions = []
            for bi in range(amount_buildings):
                action_list_of_tensors[bi] = torch.cat(
                    [action_list_of_tensors[bi], torch.zeros((1, Constants.action_dim), device=Constants.device)],
                    dim=0)
                reward_list_of_tensors[bi] = torch.cat(
                    [reward_list_of_tensors[bi], torch.zeros(1, device=Constants.device)])

                # get actions for all buildings
                step_start = time.perf_counter()
                action_bi = agent.get_action(
                    state_list_of_tensors[bi],
                    action_list_of_tensors[bi],
                    reward_list_of_tensors[bi],
                    target_return_list_of_tensors[bi],
                    timesteps,
                )
                agent_time_elapsed += time.perf_counter() - step_start

                action_list_of_tensors[bi][-1] = action_bi
                action_bi = action_bi.detach().cpu().numpy()
                next_actions.append(action_bi)

            # Interaction with the environment
            state_list_of_lists, reward_list_of_lists, done, _ = env.step(next_actions)
            state_list_of_lists = preprocess_states(state_list_of_lists, amount_buildings)
            episode_return += reward_list_of_lists

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

                # new Initialization and env Reset
                num_steps_in_episode = 0
                num_steps_in_sequence = 0

                return_to_go_list = [Constants.TARGET_RETURN] * amount_buildings
                target_returns_for_next_sequence = calc_sequence_target_return(return_to_go_list, num_steps_in_episode)

                episode_return = np.zeros(amount_buildings)
                state_list_of_lists = env.reset()
                state_list_of_lists = preprocess_states(state_list_of_lists, amount_buildings)

                state_list_of_tensors = []
                target_return_list_of_tensors = []
                action_list_of_tensors = []
                reward_list_of_tensors = []

                for bi in range(amount_buildings):
                    state_bi = torch.from_numpy(np.array(state_list_of_lists[bi])).reshape(1, Constants.state_dim).to(
                        device=Constants.device, dtype=torch.float32)
                    target_return_bi = torch.tensor(target_returns_for_next_sequence[bi], device=Constants.device,
                                                    dtype=torch.float32).reshape(1, 1)
                    action_bi = torch.zeros((0, Constants.action_dim), device=Constants.device, dtype=torch.float32)
                    reward_bi = torch.zeros(0, device=Constants.device, dtype=torch.float32)

                    state_list_of_tensors.append(state_bi)
                    target_return_list_of_tensors.append(target_return_bi)
                    action_list_of_tensors.append(action_bi)
                    reward_list_of_tensors.append(reward_bi)

                timesteps = torch.tensor(0, device=Constants.device, dtype=torch.long).reshape(1, 1)

            else:
                # Process data for next step
                for bi in range(amount_buildings):
                    cur_state = torch.from_numpy(np.array(state_list_of_lists[bi])).to(device=Constants.device).reshape(
                        1,
                        Constants.state_dim)
                    state_list_of_tensors[bi] = torch.cat([state_list_of_tensors[bi], cur_state], dim=0)
                    reward_list_of_tensors[bi][-1] = reward_list_of_lists[bi]

                    pred_return = target_return_list_of_tensors[bi][0, -1] - (
                                reward_list_of_lists[bi] / Constants.scale)
                    target_return_list_of_tensors[bi] = torch.cat(
                        [target_return_list_of_tensors[bi], pred_return.reshape(1, 1)], dim=1)
                    return_to_go_list[bi] = return_to_go_list[bi] - reward_list_of_lists[bi]

                timesteps = torch.cat(
                    [timesteps,
                     torch.ones((1, 1), device=Constants.device, dtype=torch.long) * (num_steps_in_sequence + 1)],
                    dim=1)

                if timesteps.size(dim=1) > context_length:
                    # Store only the last values according to context_length
                    timesteps = timesteps[:, -context_length:]
                    for bi in range(amount_buildings):
                        state_list_of_tensors[bi] = state_list_of_tensors[bi][-context_length:]
                        action_list_of_tensors[bi] = action_list_of_tensors[bi][-context_length:]
                        reward_list_of_tensors[bi] = reward_list_of_tensors[bi][-context_length:]
                        target_return_list_of_tensors[bi] = target_return_list_of_tensors[bi][:, -context_length:]

                num_steps_in_episode += 1
                num_steps_in_sequence += 1

            num_steps_total += 1

            if num_steps_total % 100 == 0:
                print(f"Num Steps: {num_steps_total}, Num episodes: {episodes_completed}")

            if episodes_completed >= Constants.episodes:
                break

        print("========================= Evaluation Done ========================")
        print(f"Total time taken by agent: {agent_time_elapsed}s")
        sys.stdout = f
        print(f"Total time taken by agent: {agent_time_elapsed}s")
        print("Total number of steps:", num_steps_total)
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


if __name__ == '__main__':
    evaluate()
