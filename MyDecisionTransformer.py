import torch
from transformers import DecisionTransformerModel


class MyDecisionTransformer:
    def __init__(self, load_from, force_download, device):

        # here we load the model
        model, info = DecisionTransformerModel.from_pretrained(load_from, force_download=force_download, output_loading_info=True)
        self.model = model.to(device)
        self.device = device
        # print("Loading Model Info:", info)

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



