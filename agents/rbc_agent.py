import numpy as np


class BasicRBCAgent:

    def __init__(self):
        self.action_dim = 1

    def register_reset(self, obs_dict):
        """Get the first observation after env reset, return action"""
        obs = obs_dict["observation"]

        return self.compute_action(obs)

    def compute_action(self, obs):
        """Get observation return action"""
        return self.basic_rbc_policy(obs)

    def basic_rbc_policy(self, observation):
        """
        Simple rule based policy based on day or nighttime
        """
        hour = observation[0][2]  # Hour index is 2 for all observations

        action = 0.0  # Default value
        if 9 <= hour <= 21:
            # Daytime: release stored energy
            action = -0.08
        elif (1 <= hour <= 8) or (22 <= hour <= 24):
            # Early nighttime: store DHW and/or cooling energy
            action = 0.091

        actions = []
        for bi in range(len(observation)):
            actions.append(np.array([action]))

        return actions


class BetterRBCAgent:

    def __init__(self):
        self.action_dim = 1

    def register_reset(self, obs_dict):
        """Get the first observation after env reset, return action"""
        obs = obs_dict["observation"]

        return self.compute_action(obs)

    def compute_action(self, obs):
        """Get observation return action"""
        return self.better_rbc_policy(obs)

    def better_rbc_policy(self, observation):
        """
        Rule based policy depending on daytime, solar_generation and electrical_storage.
        Parameters determined by manual search.
        """
        actions = []
        hour = observation[0][2]

        alpha = 0.076
        beta = 0.201
        gamma = 0.77

        for bi in range(len(observation)):
            solar_generation = observation[bi][21]
            electrical_storage = observation[bi][22]

            # Daytime: load battery with regard to current solar generation
            if 7 <= hour <= 16:
                action = alpha * solar_generation

            # Evening and night: try to use stored energy with regard to current storage level
            else:
                action = - beta * electrical_storage

            #  slow down positive actions near storage limit
            if action > 0:
                if electrical_storage > 0.99:
                    action = 0
                elif electrical_storage > 0.7:
                    action = gamma * action

            #  clip actions to [1,-1]
            if action > 1:
                action = 1
            if action < -1:
                action = -1

            actions.append(np.array([action]))

        return actions
