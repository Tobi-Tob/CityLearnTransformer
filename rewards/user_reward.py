##################################################################
#####                DO NOT EDIT THIS MODULE!                #####
##################################################################

from typing import List
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
        agent_count=agent_count

        if observation is None:
            electricity_consumption = None
            carbon_emission = None
            electricity_price = None
        else:
            electricity_consumption=[o[electricity_consumption_index] for o in observation]
            carbon_emission=[o[carbon_emission_index]*o[electricity_consumption_index] for o in observation]
            electricity_price=[o[electricity_pricing_index]*o[electricity_consumption_index] for o in observation]

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

        return get_reward(self.electricity_consumption, 
                          self.carbon_emission, 
                          self.electricity_price, 
                          list(range(self.agent_count)))