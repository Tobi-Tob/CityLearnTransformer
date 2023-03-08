from typing import List

import numpy as np
from citylearn.cost_function import CostFunction
from citylearn.reward_function import RewardFunction
from rewards.get_reward import get_reward


class UserReward(RewardFunction):
    def __init__(self, agent_count,
                 observation: List[dict] = None,
                 **kwargs):

        # calculate rewards
        electricity_consumption_index = 23
        carbon_emission_index = 19
        electricity_pricing_index = 24
        agent_count = agent_count

        self.electricity_consumption_history = []  # List[float]
        self.max_history_length = 2
        # nec_mean_5_buildings = 4.367397951030437  # mean of all positive electricity consumptions of reference agent
        # nec_std_5_buildings = 2.4662300511545916  # standard deviation of all positive electricity consumptions of reference agent
        # self.nec_mean = nec_mean_5_buildings / 5
        # self.nec_std = nec_std_5_buildings / 5  # mean nec of reference agent 1 building

        if observation is None:
            electricity_consumption = None
            carbon_emission = None
            electricity_price = None
        else:
            electricity_consumption = [o[electricity_consumption_index] for o in observation]
            carbon_emission = [o[carbon_emission_index] * o[electricity_consumption_index] for o in observation]
            electricity_price = [o[electricity_pricing_index] * o[electricity_consumption_index] for o in observation]

        super().__init__(agent_count=agent_count,
                         electricity_consumption=electricity_consumption,
                         carbon_emission=carbon_emission,
                         electricity_price=electricity_price)

    def calculate(self) -> List[float]:

        self.electricity_consumption_history.append(self.electricity_consumption)
        self.electricity_consumption_history = self.electricity_consumption_history[-self.max_history_length:]

        return self.get_reward2(self.electricity_consumption,
                                self.carbon_emission,
                                self.electricity_price,
                                list(range(self.agent_count)))

    def get_reward2(self, electricity_consumption: List[float], carbon_emission: List[float],
                    electricity_price: List[float],
                    agent_ids: List[int]) -> List[float]:

        alpha = 4.98866624
        price_cost = - alpha * np.array(electricity_price).clip(min=0)

        beta = 9.46691265
        emission_cost = - beta * np.array(carbon_emission).clip(min=0)

        gamma = 1.96800773
        ramping_cost = np.zeros(len(agent_ids))  # Costs when the building has fluctuations in electricity consumption
        if len(self.electricity_consumption_history) >= 2:
            ramping_cost = - gamma * abs(
                np.array(self.electricity_consumption_history[-1]) - np.array(self.electricity_consumption_history[-2]))

        delta = 0.92670507
        load_factor_cost = - delta * np.array(electricity_consumption).clip(min=0) ** 2

        reward = 1 / 3 * price_cost + 1 / 3 * emission_cost + 1 / 6 * ramping_cost + 1 / 6 * load_factor_cost
        return reward

    def get_reward1(self, electricity_consumption: List[float], carbon_emission: List[float],
                    electricity_price: List[float],
                    agent_ids: List[int]) -> List[float]:

        price_cost = - np.array(electricity_price).clip(min=0)
        emission_cost = - np.array(carbon_emission).clip(min=0)

        reward = price_cost + emission_cost
        return reward
