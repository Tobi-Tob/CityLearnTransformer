import os
import numpy as np
import torch

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = '1'
from transformers import DecisionTransformerModel
from huggingface_hub import hf_hub_download


class MyDecisionTransformer:
    def __init__(self, load_from, force_download, device, state_dim, action_dim, target_return, scale_rewards, **kwargs):

        # load the model
        model, info = DecisionTransformerModel.from_pretrained(load_from, force_download=force_download, output_loading_info=True)
        self.model = model.to(device)
        self.device = device
        # load mean and std from training process
        state_mean = hf_hub_download(repo_id=load_from, filename="state_mean.npy", force_download=force_download)
        state_std = hf_hub_download(repo_id=load_from, filename="state_std.npy", force_download=force_download)
        self.state_mean = np.load(state_mean)
        self.state_std = np.load(state_std)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_return = target_return
        self.scale_rewards = scale_rewards
        # print("Loading Model Info:", info)

    def preprocess_states(self, state_list_of_lists, amount_buildings):
        for bi in range(amount_buildings):
            for si in range(self.state_dim):  # TODO more efficient?
                state_list_of_lists[bi][si] = (state_list_of_lists[bi][si] - self.state_mean[si]) / self.state_std[si]

        return state_list_of_lists

    def calc_sequence_target_return(self, return_to_go_list, num_steps_in_episode, evaluation_interval, total_time_steps):
        timesteps_left = total_time_steps - num_steps_in_episode
        target_returns_for_next_sequence = []
        for bi in range(len(return_to_go_list)):
            required_reward_per_timestep = return_to_go_list[bi] / timesteps_left
            if timesteps_left < evaluation_interval:
                target_returns_for_next_sequence.append(required_reward_per_timestep * timesteps_left / self.scale_rewards)
            else:
                target_returns_for_next_sequence.append(
                    required_reward_per_timestep * evaluation_interval / self.scale_rewards)
        return target_returns_for_next_sequence

    # Function that gets an action from the model using autoregressive prediction with a window of the previous 20
    # time steps.
    def get_action(self, states, actions, rewards, returns_to_go, timesteps):
        # This implementation does not condition on past rewards

        states = states.reshape(1, -1, self.model.config.state_dim).to(device=self.device)
        actions = actions.reshape(1, -1, self.model.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -self.model.config.max_length:]
        actions = actions[:, -self.model.config.max_length:]
        returns_to_go = returns_to_go[:, -self.model.config.max_length:]
        timesteps = timesteps[:, -self.model.config.max_length:]
        padding = self.model.config.max_length - states.shape[1]
        # pad all tokens to sequence length
        attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1).to(device=self.device)
        states = torch.cat([torch.zeros((1, padding, self.model.config.state_dim)).to(device=self.device), states], dim=1).float()
        actions = torch.cat([torch.zeros((1, padding, self.model.config.act_dim)).to(device=self.device), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padding, 1)).to(device=self.device), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long).to(device=self.device), timesteps], dim=1)

        state_preds, action_preds, return_preds = self.model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1]
