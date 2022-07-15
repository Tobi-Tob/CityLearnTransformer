##################################################################
#####                DO NOT EDIT THIS MODULE!                #####
##################################################################

from typing import List
from citylearn.reward_function import RewardFunction
from rewards.get_reward import get_reward

class UserReward(RewardFunction):
    def __init__(self, agent_count: int, electricity_consumption: List[float] = None, carbon_emission: List[float] = None, electricity_price: List[float] = None, **kwargs):
        super().__init__(agent_count=agent_count, electricity_consumption=electricity_consumption, carbon_emission=carbon_emission, electricity_price=electricity_price, **kwargs)

    def calculate(self) -> List[float]:
        """CityLearn Challenge reward calculation.

        Notes
        -----
        This function is called internally in the environment's :meth:`citylearn.CityLearnEnv.step` function.
        """

        return get_reward(self.electricity_consumption, self.carbon_emission, self.electricity_price)