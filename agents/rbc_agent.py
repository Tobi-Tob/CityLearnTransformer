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


class RBCAgent2:

    def __init__(self):
        self.action_dim = 1
        self.electricity_pricing = [None, None, None, None, None]
        self.diffuse_solar_prediction = [None, None, None, None, None]
        self.building_solar_power = None
        self.building_annual_demand = None

    def register_reset(self, obs_dict):
        """Get the first observation after env reset, return action"""
        obs = obs_dict["observation"]
        building_info = obs_dict["building_info"]
        self.electricity_pricing = [None, None, None, None, None]
        self.diffuse_solar_prediction = [None, None, None, None, None]
        self.building_solar_power = [building['solar_power'] for building in building_info]
        self.building_annual_demand = [building['annual_nonshiftable_electrical_demand'] for building in building_info]

        return self.compute_action(obs)

    def compute_action(self, obs):
        """Get observation return action"""
        self.electricity_pricing.append(obs[0][25])
        electricity_pricing_predicted = self.electricity_pricing.pop(0)
        if electricity_pricing_predicted is None:
            electricity_pricing_predicted = obs[0][24]
        self.diffuse_solar_prediction.append(obs[0][12])
        diffuse_solar_predicted = self.diffuse_solar_prediction.pop(0)
        if diffuse_solar_predicted is None:
            diffuse_solar_predicted = obs[0][11]

        actions = []
        for b in range(len(obs)):
            action = self.rbc_policy_2(obs[b], b, diffuse_solar_predicted, electricity_pricing_predicted)
            actions.append(action)
        return actions

    def rbc_policy_2(self, building_observation, building_id, diffuse_solar_predicted, electricity_pricing_predicted):
        """
        Rule based policy depending on daytime, solar_generation and electrical_storage.
        Parameters determined by manual search.
        """
        alpha = 1.1
        beta = 0.00075  # ohne ist besser
        gamma = 0.5
        delta = 0.2
        epsilon = 4.5
        zeta = 2.4
        eta = 0.7
        theta = 0.5
        lota = 1.8

        action = 0
        hour = building_observation[2]
        day_type = building_observation[1]
        month = building_observation[0]
        solar_generation = building_observation[21]
        electrical_storage = building_observation[22]
        carbon_intensity = building_observation[19]
        peak_hour_of_solar_generation = 13 if (4 <= month <= 9) else 12

        solar_power = self.building_solar_power[building_id]
        annual_nonshiftable_electrical_demand = self.building_annual_demand[building_id]

        daytime_non_shiftable_load_predicted = alpha * annual_nonshiftable_electrical_demand / 8760
        solar_generation_predicted = ((beta * solar_power * diffuse_solar_predicted) + solar_generation) / 2
        solar_generation_surplus_predicted = solar_generation_predicted - daytime_non_shiftable_load_predicted

        if peak_hour_of_solar_generation - 3 <= hour <= peak_hour_of_solar_generation + 3:
            if solar_generation_surplus_predicted >= gamma:
                action = delta * solar_generation_surplus_predicted
        assert(0 <= 1-electrical_storage <= 1)
        price = epsilon * carbon_intensity + zeta * electricity_pricing_predicted

        if price + electrical_storage <= eta:  # Favorable time to buy electricity and battery low
            action += theta * (eta - price)
        if price >= lota:  # Expensive time, make it more unlikely to buy energy (maybe need prediction of price)
            action -= theta * (price - lota)

        return np.array([action])
