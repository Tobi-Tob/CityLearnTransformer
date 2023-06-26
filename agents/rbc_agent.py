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


class RBCAgent1:

    def __init__(self):
        self.action_dim = 1

    def register_reset(self, obs_dict):
        """Get the first observation after env reset, return action"""
        obs = obs_dict["observation"]

        return self.compute_action(obs)

    def compute_action(self, obs):
        """Get observation return action"""
        return self.rbc_policy_1(obs)

    def rbc_policy_1(self, observation):
        """
        Rule based policy depending on daytime, solar_generation and electrical_storage.
        Parameters determined by manual search.
        """
        actions = []
        hour = observation[0][2]

        # Parameters:
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

            # Slow down positive actions near storage limit
            if action > 0:
                if electrical_storage > 0.99:
                    action = 0
                elif electrical_storage > 0.7:
                    action = gamma * action

            # Clip actions to [1,-1]
            if action > 1:
                action = 1
            if action < -1:
                action = -1

            actions.append(np.array([action]))

        return actions


class RBCAgent2:

    def __init__(self):
        self.action_dim = 1
        self.building_annual_demand = None

    def register_reset(self, obs_dict):
        """Get the first observation after env reset, return action"""
        obs = obs_dict["observation"]
        building_info = obs_dict["building_info"]
        self.building_annual_demand = [building['annual_nonshiftable_electrical_demand'] for building in building_info]

        return self.compute_action(obs)

    def compute_action(self, obs):
        """Get observation return action"""
        actions = []
        for b in range(len(obs)):
            action = self.rbc_policy_2(obs[b], b)
            actions.append(action)
        return actions

    def rbc_policy_2(self, building_observation, building_id):
        """
        Rule based policy depending on daytime, solar_generation and electrical_storage.
        Parameters determined by manual search.
        """
        # Parameters:
        annual_factor = 0.93
        solar_factor = 0.141
        carbon_weight = 5.3
        pricing_weight = 2.53
        price_upper_threshold = 1.52
        sell_factor = 0.237
        saturation_threshold = 0.5
        saturation = 0.55

        action = 0
        solar_generation = building_observation[21]
        electrical_storage = building_observation[22]
        carbon_intensity = building_observation[19]
        electricity_pricing = building_observation[24]
        annual_nonshiftable_electrical_demand = self.building_annual_demand[building_id]

        non_shiftable_load_predicted = annual_factor * annual_nonshiftable_electrical_demand / 8760
        solar_generation_surplus_predicted = solar_generation - non_shiftable_load_predicted

        if solar_generation_surplus_predicted >= 0:
            action = solar_factor * solar_generation_surplus_predicted
        price = carbon_weight * carbon_intensity + pricing_weight * electricity_pricing

        if price >= price_upper_threshold:  # Expensive time, make it more unlikely to buy energy
            action -= sell_factor * (price - price_upper_threshold)

        # Slow down positive actions near storage limit
        if action > 0 and electrical_storage > saturation_threshold:
            action = saturation * action

        # Clip actions to [1,-1]
        if action > 1:
            action = 1
        if action < -1:
            action = -1

        return np.array([action])
