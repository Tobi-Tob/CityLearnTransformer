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
        """CityLearn Challenge reward calculation.

        Notes
        -----
        This function is called internally in the environment's :meth:`citylearn.CityLearnEnv.step` function.
        """

        self.electricity_consumption_history.append(self.electricity_consumption)
        self.electricity_consumption_history = self.electricity_consumption_history[-self.max_history_length:]

        return self.get_reward(self.electricity_consumption,
                               self.carbon_emission,
                               self.electricity_price,
                               list(range(self.agent_count)))

    def get_reward(self, electricity_consumption: List[float], carbon_emission: List[float],
                   electricity_price: List[float],
                   agent_ids: List[int]) -> List[float]:

        carbon_emission = np.array(carbon_emission).clip(min=0)
        electricity_price = np.array(electricity_price).clip(min=0)
        reward = (carbon_emission + electricity_price) * -1

        # reward = (np.array(electricity_consumption) * -1).clip(max=0).tolist()

        # reward = CostFunction.ramping(self.net_electricity_consumption_history)[-1]

        ramping_cost = np.zeros(len(agent_ids))  # Costs when the building has fluctuations in electricity consumption
        if len(self.electricity_consumption_history) >= 2:
            ramping_cost = abs(np.array(self.electricity_consumption_history[-1]) - np.array(self.electricity_consumption_history[-2]))

        load_factor_cost = np.zeros(len(agent_ids))


        return reward
