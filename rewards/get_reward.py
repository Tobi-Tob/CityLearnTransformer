from typing import List
import numpy as np

###########################################################################
#####                Specify your reward function here                #####
###########################################################################

def get_reward(electricity_consumption: List[float], carbon_emission: List[float], electricity_price: List[float], agent_ids: List[int]) -> List[float]:
        """CityLearn Challenge user reward calculation.

        Parameters
        ----------
        electricity_consumption: List[float]
            List of each building's/total district electricity consumption in [kWh].
        carbon_emission: List[float]
            List of each building's/total district carbon emissions in [kg_co2].
        electricity_price: List[float]
            List of each building's/total district electricity price in [$].
        agent_ids: List[int]
            List of agent IDs matching the ordering in `electricity_consumption`, `carbon_emission` and `electricity_price`.

        Returns
        -------
        rewards: List[float]
            Agent(s) reward(s) where the length of returned list is either = 1 (central agent controlling all buildings) 
            or = number of buildings (independent agent for each building).
        """

        # *********** BEGIN EDIT ***********
        # Replace with custom reward calculation
        carbon_emission = np.array(carbon_emission).clip(min=0)
        electricity_price = np.array(electricity_price).clip(min=0)
        reward = (carbon_emission + electricity_price)*-1
        # ************** END ***************
        
        return reward