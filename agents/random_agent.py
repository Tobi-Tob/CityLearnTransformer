import random

import numpy as np


class RandomAgent:

    def __init__(self):
        self.action_dim = 1

    def register_reset(self, obs_dict):
        """Get the first observation after env reset, return action"""
        obs = obs_dict["observation"]

        return self.compute_action(obs)

    def compute_action(self, obs):
        """Get observation return action"""
        return self.random_policy(obs)

    def random_policy(self, obs):
        actions = []
        for bi in range(len(obs)):
            action_i = np.zeros(self.action_dim)
            for ai in range(self.action_dim):
                action_i[ai] = random.uniform(-1, 1)

            actions.append(action_i)

        return actions
