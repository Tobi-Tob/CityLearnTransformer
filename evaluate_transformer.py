import os
import pathlib

import numpy as np
import time
import torch
from MyDecisionTransformer import MyDecisionTransformer
from citylearn.citylearn import CityLearnEnv
import sys

"""
This file is used to evaluate a decision transformer loaded form https://huggingface.co/TobiTob/model_name
"""


class Constants:
    file_to_save = 'evaluation_results.txt'
    """Environment Constants"""
    episodes = 1  # amount of environment resets
    state_dim = 28  # size of state space
    action_dim = 1  # size of action space
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

    """Model Constants"""
    load_model = "TobiTob/decision_transformer_2"
    force_download = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TARGET_RETURN = -2500  # vllt Vector aus 5 Werten
    # mean and std computed from training dataset these are available in the model card for each model.

    state_mean = np.array(
        [6.525973284621532, 3.9928073981048064, 12.498801233017467, 16.836990550577212, 16.837287388159297,
         16.83684213167729, 16.837161803003287, 73.00388172165772, 73.00331088023746, 73.00445256307798,
         73.00331088023746, 208.30597100125584, 208.30597100125584, 208.20287704075807, 208.30597100125584,
         201.25448110514898, 201.25448110514898, 201.16189062678387, 201.25448110514898, 0.15652765849893777,
         1.0663012570140091, 0.6994348432433195, 0.5023924181838172, 0.49339119658209996, 0.2731373418679261,
         0.2731373418679261, 0.2731373418679261, 0.2731373418679261])
    state_std = np.array(
        [3.448045414453991, 2.0032677368929734, 6.921673394725967, 3.564552828057008, 3.5647828974724476,
         3.5643565817901974, 3.564711987899257, 16.480221141108398, 16.480030755727572, 16.480238315742053,
         16.480030755727565, 292.79094956097464, 292.79094956097464, 292.70528837855596, 292.79094956097543,
         296.18549714910006, 296.18549714910023, 296.1216266457902, 296.18549714910006, 0.035369600587780235,
         0.8889958578862672, 1.0171468928300462, 0.40202104980478576, 2.6674362928093682, 0.11780233435944305,
         0.11780233435944333, 0.11780233435944351, 0.11780233435944402])


def preprocess_states(state_list_of_lists, amount_buildings):
    for bi in range(amount_buildings):
        for si in range(Constants.state_dim):
            state_list_of_lists[bi][si] = (state_list_of_lists[bi][si] - Constants.state_mean[si]) / \
                                          Constants.state_std[si]

    return state_list_of_lists


def evaluate():
    print("========================= Start Evaluation ========================")
    # Check current working directory.
    retval = os.getcwd()
    print("Current working directory %s" % retval)

    env = CityLearnEnv(schema=Constants.schema_path)

    agent = MyDecisionTransformer(load_from=Constants.load_model, force_download=Constants.force_download,
                                  device=Constants.device)
    print("Using device:", Constants.device)
    start_timestep = env.schema['simulation_start_time_step']
    end_timestep = env.schema['simulation_end_time_step']
    print("Environment simulation from", start_timestep, "to", end_timestep)

    context_length = agent.model.config.max_length
    amount_buildings = len(env.buildings)

    scale = 1000.0  # normalization for rewards/returns
    target_return = Constants.TARGET_RETURN / scale

    # Initialize Tensors
    episode_return = np.zeros(amount_buildings)
    state_list_of_lists = env.reset()
    state_list_of_lists = preprocess_states(state_list_of_lists, amount_buildings)

    state_list_of_tensors = []
    target_return_list_of_tensors = []
    action_list_of_tensors = []
    reward_list_of_tensors = []

    for bi in range(amount_buildings):
        state_bi = torch.from_numpy(np.array(state_list_of_lists[bi])).reshape(1, Constants.state_dim).to(
            device=Constants.device,
            dtype=torch.float32)
        target_return_bi = torch.tensor(target_return, device=Constants.device, dtype=torch.float32).reshape(1, 1)
        action_bi = torch.zeros((0, Constants.action_dim), device=Constants.device, dtype=torch.float32)
        reward_bi = torch.zeros(0, device=Constants.device, dtype=torch.float32)

        state_list_of_tensors.append(state_bi)
        target_return_list_of_tensors.append(target_return_bi)
        action_list_of_tensors.append(action_bi)
        reward_list_of_tensors.append(reward_bi)

    timesteps = torch.tensor(0, device=Constants.device, dtype=torch.long).reshape(1, 1)
    # print(state_list_of_tensors) Liste mit 5 Tensoren, jeder Tensor enthält einen State s der Länge 28
    # print(action_list_of_tensors) Liste mit 5 leeren Tensoren mit size (0,1)
    # print(reward_list_of_tensors) Liste mit 5 leeren Tensoren ohne size
    # print(target_return_list_of_tensors) Liste mit 5 leeren Tensoren, jeder Tensor enthält den target_return / scale
    # print(timesteps) enthält einen Tensor mit 0: tensor([[0]])

    episodes_completed = 0
    num_steps = 0
    t = 0
    agent_time_elapsed = 0
    episode_metrics = []

    original_stdout = sys.stdout
    with open(Constants.file_to_save, 'w') as f:
        sys.stdout = f
        print("==> Model:", Constants.load_model)
        print("Target Return:", Constants.TARGET_RETURN)
        print("Context Length:", context_length)
        sys.stdout = original_stdout

        while True:

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
                t = 0
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
                    target_return_bi = torch.tensor(target_return, device=Constants.device,
                                                    dtype=torch.float32).reshape(1,
                                                                                 1)
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

                    pred_return = target_return_list_of_tensors[bi][0, -1] - (reward_list_of_lists[bi] / scale)
                    target_return_list_of_tensors[bi] = torch.cat(
                        [target_return_list_of_tensors[bi], pred_return.reshape(1, 1)], dim=1)

                    episode_return[bi] += reward_list_of_lists[bi]

                timesteps = torch.cat(
                    [timesteps, torch.ones((1, 1), device=Constants.device, dtype=torch.long) * (t + 1)],
                    dim=1)

                if timesteps.size(dim=1) > context_length:
                    # Store only the last values according to context_length
                    timesteps = timesteps[:, -context_length:]
                    for bi in range(amount_buildings):
                        state_list_of_tensors[bi] = state_list_of_tensors[bi][-context_length:]
                        action_list_of_tensors[bi] = action_list_of_tensors[bi][-context_length:]
                        reward_list_of_tensors[bi] = reward_list_of_tensors[bi][-context_length:]
                        target_return_list_of_tensors[bi] = target_return_list_of_tensors[bi][:, -context_length:]

            num_steps += 1
            t += 1
            if num_steps % 100 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= Constants.episodes:
                break

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
        print(f"Total time taken by agent: {agent_time_elapsed}s")
        sys.stdout = original_stdout
        print("Evaluation saved in:", str(pathlib.Path(__file__).parent.resolve()) + '/' + Constants.file_to_save)


if __name__ == '__main__':
    evaluate()
