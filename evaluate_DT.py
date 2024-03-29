import os
import pathlib
import numpy as np
import time
import torch
from MyDecisionTransformer import MyDecisionTransformer
import sys

from utils import init_environment

"""
This file is used to evaluate a decision transformer loaded form https://huggingface.co/TobiTob/model_name
"""


class Constants:
    """Environment Constants"""
    episodes = 1  # amount of environment resets
    state_dim = 28  # size of state space
    action_dim = 1  # size of action space

    """Model Constants"""
    scale = 1000.0  # normalization for rewards/returns
    force_download = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(DT_model, buildings_to_use, TR, evaluation_interval, simulation_start_end=None, file_to_save='_results.txt'):
    r"""Evaluates a DT model in the environment

            Parameters
            ----------
            DT_model: str
                The model id of a pretrained model hosted inside a model repo on huggingface,
                example: 'TobiTob/decision_transformer_fr_24'.
            buildings_to_use: str or list
                String to define which buildings are used in the environment.
                One of: "train", "validation", "test"
                List to define manually which buildings to use, example: [1, 2, 4, 15]
            TR: int
                Target Return, hyperparameter of a DT model.
            evaluation_interval: int
                The simulation is split into intervals of length evaluation_interval.
                For a new interval, a partition of the TR is calculated and given to the model.
                Also, the state, action, reward history is reset for each interval.
                Large Intervals may cause memory problems.
            simulation_start_end: list[int]
                List to define start and end time step of the simulation, example: [0,8759].
            file_to_save: str
                Name of a file to save the results in, example: '_results.txt'.

    """
    print("========================= Start Evaluation ========================")
    # Check current working directory.
    retval = os.getcwd()
    print("Current working directory %s" % retval)

    if buildings_to_use == "train":
        buildings_to_use = [1, 2, 3, 4, 5]
    elif buildings_to_use == "validation":
        buildings_to_use = [6, 7, 8, 9, 10]
    elif buildings_to_use == "test":
        buildings_to_use = [11, 12, 13, 14, 15, 16, 17]

    env_params = dict(state_dim=Constants.state_dim, action_dim=Constants.action_dim, buildings_to_use=buildings_to_use,
                      simulation_start_end=simulation_start_end)
    agent_params = dict(load_from=DT_model, force_download=Constants.force_download, device=Constants.device, target_return=TR,
                        scale_rewards=Constants.scale)

    env = init_environment(**env_params)

    agent = MyDecisionTransformer(**env_params, **agent_params)

    start_timestep = env.schema['simulation_start_time_step']
    end_timestep = env.schema['simulation_end_time_step']
    total_time_steps = end_timestep - start_timestep

    print("Using device:", Constants.device)
    print("==> Model:", DT_model)

    context_length = agent.model.config.max_length
    amount_buildings = len(env.buildings)

    print("Target Return:", agent.target_return)
    print("Context Length:", context_length)
    print("Evaluation Interval Length:", evaluation_interval)
    start_timestep = env.schema['simulation_start_time_step']
    end_timestep = env.schema['simulation_end_time_step']
    print("Environment simulation from", start_timestep, "to", end_timestep)
    print("Amount of buildings:", amount_buildings)

    episodes_completed = 0
    sequences_completed = 0
    num_steps_total = 0
    num_steps_in_episode = 0
    num_steps_in_sequence = 0
    agent_time_elapsed = 0
    episode_metrics = []

    return_to_go_list = [agent.target_return] * amount_buildings
    target_returns_for_next_sequence = agent.calc_sequence_target_return(return_to_go_list, num_steps_in_episode, evaluation_interval,
                                                                         total_time_steps)

    # Initialize Tensors
    state_list_of_lists = env.reset()
    state_list_of_lists = agent.preprocess_states(state_list_of_lists, amount_buildings)

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
    # print(state_list_of_tensors) Liste mit 5 Tensoren, jeder Tensor enthält einen State s der Länge 28
    # print(action_list_of_tensors) Liste mit 5 leeren Tensoren mit size (0,1)
    # print(reward_list_of_tensors) Liste mit 5 leeren Tensoren ohne size
    # print(target_return_list_of_tensors) Liste mit 5 leeren Tensoren, jeder Tensor enthält den target_return / scale
    # print(timesteps) enthält einen Tensor mit 0: tensor([[0]])

    original_stdout = sys.stdout
    with open(file_to_save, 'w') as f:
        sys.stdout = f
        print("==> Model:", DT_model)
        print("Target Return:", agent.target_return)
        print("Context Length:", context_length)
        print("Evaluation Interval Length:", evaluation_interval)
        start_timestep = env.schema['simulation_start_time_step']
        end_timestep = env.schema['simulation_end_time_step']
        print("Environment simulation from", start_timestep, "to", end_timestep)
        print("Buildings used:", buildings_to_use)
        sys.stdout = original_stdout

        while True:
            if num_steps_in_sequence >= evaluation_interval:  # if Sequence complete
                sequences_completed += 1
                num_steps_in_sequence = 0

                target_returns_for_next_sequence = agent.calc_sequence_target_return(return_to_go_list, num_steps_in_episode, evaluation_interval,
                                                                                     total_time_steps)

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
            state_list_of_lists = agent.preprocess_states(state_list_of_lists, amount_buildings)
            episode_return += reward_list_of_lists

            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0] * 100, "emmision_cost": metrics_t[1] * 100, "grid_cost": metrics_t[2] * 100}
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

                return_to_go_list = [agent.target_return] * amount_buildings
                target_returns_for_next_sequence = agent.calc_sequence_target_return(return_to_go_list, num_steps_in_episode, evaluation_interval,
                                                                                     total_time_steps)

                episode_return = np.zeros(amount_buildings)
                state_list_of_lists = env.reset()
                state_list_of_lists = agent.preprocess_states(state_list_of_lists, amount_buildings)

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

            if num_steps_total % 500 == 0:
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
        print("Evaluation saved in:", str(pathlib.Path(__file__).parent.resolve()) + '/' + file_to_save)


if __name__ == '__main__':
    evaluate(DT_model="TobiTob/decision_transformer_merged3", buildings_to_use="validation",
             TR=-9000, evaluation_interval=168, simulation_start_end=[0, 87])

"""
Run in console:

from evaluate_DT import evaluate
evaluate("TobiTob/decision_transformer_fr_24", "validation", -9000, 24)
"""
