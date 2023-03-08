import os
import pathlib

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
    load_model = "TobiTob/decision_transformer_2"
    TARGET_RETURN = -300
    force_download = False
    device = "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # mean and std computed from training dataset these are available in the model card for each model.
    state_mean = np.array(
        [6.519444444444445, 3.9837962962962963, 12.5, 16.78500002964779, 16.78491901104097, 16.785196788774595,
         16.7854977140824, 72.89907407407408, 72.90567129629629, 72.909375, 72.91342592592592, 207.3190972222222,
         207.3190972222222, 207.18541666666667, 207.23611111111111, 201.11863425925927, 201.11863425925927,
         200.80648148148148, 200.88761574074073, 0.15636648599282596, 1.0591688615113237, 0.6963716355719771,
         0.29117993655536184, 0.39915770157282743, 0.27310532142960087, 0.27310532142960087, 0.27310532142960087,
         0.27310532142960087])
    state_std = np.array(
        [3.4712575323780417, 2.001555126539623, 6.922187552431729, 3.553894201620696, 3.553811950901204,
         3.5540391337133648, 3.5546125114989646, 16.542013984734275, 16.546533736927085, 16.54789739417334,
         16.548964694872215, 291.88390011273555, 291.8839001127356, 291.7552778101503, 291.83391263987227,
         296.4150072818235, 296.4150072818235, 296.26064904211574, 296.30532672046087, 0.03537502595557663,
         0.8835211260856368, 1.0154967670063433, 0.3233198693428126, 0.9206463117416827, 0.11787932777153287,
         0.11787932777153277, 0.11787932777153269, 0.11787932777153291])


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

    env = Constants.env

    agent = MyDecisionTransformer(load_from=Constants.load_model, force_download=Constants.force_download,
                                  device=Constants.device)
    print("Using device:", Constants.device)
    print("==> Model:", Constants.load_model)

    context_length = agent.model.config.max_length
    amount_buildings = len(env.buildings)

    print("Target Return:", Constants.TARGET_RETURN)
    print("Context Length:", context_length)
    print("Amount of buildings:", amount_buildings)

    scale = 1000.0  # normalization for rewards/returns
    target_return = Constants.TARGET_RETURN / scale

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
        start_timestep = env.schema['simulation_start_time_step']
        end_timestep = env.schema['simulation_end_time_step']
        print("Environment simulation from", start_timestep, "to", end_timestep)
        print("Buildings used:", Constants.buildings_to_use)
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


if __name__ == '__main__':
    evaluate()
