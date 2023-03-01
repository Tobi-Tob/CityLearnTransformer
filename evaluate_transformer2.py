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

    state_mean = state_mean.astype(np.float32)
    state_std = state_std.astype(np.float32)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)


def process_data_from_env(all_states, all_rewards, building_to_evaluate):
    state_as_array = np.array(all_states[building_to_evaluate])
    cur_state_as_tensor = torch.from_numpy(state_as_array).to(device=Constants.device).reshape(1, Constants.state_dim)
    cur_state = (cur_state_as_tensor - Constants.state_mean) / Constants.state_std

    cur_reward = all_rewards[building_to_evaluate]

    return cur_state, cur_reward


def evaluate():
    building_to_evaluate = 0
    device = Constants.device

    scale = 1000.0  # normalization for rewards/returns TODO experimerntiere damit
    TARGET_RETURN = Constants.TARGET_RETURN / scale

    state_dim = Constants.state_dim
    act_dim = Constants.action_dim

    agent = MyDecisionTransformer(load_from=Constants.load_model, force_download=Constants.force_download,
                                  device=device)

    env = CityLearnEnv(schema=Constants.schema_path)

    max_ep_len = 1

    episode_return, episode_length = 0, 0

    all_states = env.reset()
    # Bearbeite env Output
    state_as_array = np.array(all_states[building_to_evaluate])
    # print(state_as_array)
    state_as_tensor = torch.from_numpy(state_as_array).reshape(1, state_dim).to(device=device,
                                                                                dtype=torch.float32)  # TODO vllt geht auch .from_list
    # print(state_as_tensor)
    states = (state_as_tensor - Constants.state_mean) / Constants.state_std
    # print(states)

    target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)  # Todo return s
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)  # erweitert actions an der letzten Stelle um [0]
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])  # erweitert rewards an der letzten Stelle um 0

        action = agent.get_action(
            states,
            actions,
            rewards,
            target_return,
            timesteps,
        )
        actions[-1] = action  # schreibt neue action an der letzten Stelle (anstelle von [0]) hinein
        action = action.detach().cpu().numpy()  # convert to ndarray

        all_actions = [action, [0], [0], [0], [0]]

        all_states, all_rewards, done, _ = env.step(all_actions)

        # Bearbeite env Output
        cur_state, cur_reward = process_data_from_env(all_states, all_rewards, building_to_evaluate)

        states = torch.cat([states, cur_state], dim=0)  # hinten an hängen

        rewards[-1] = cur_reward  # schreibt neuen reward an der letzten Stelle (anstelle von 0) hinein

        pred_return = target_return[0, -1] - (cur_reward / scale)  # verringere target_return
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)  # hinten an hängen
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)  # hinten an hängen TODO brauch man die?

        episode_return += cur_reward
        episode_length += 1

        if done:
            break


if __name__ == '__main__':
    evaluate()
