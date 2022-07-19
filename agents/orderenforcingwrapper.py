from gym.spaces import Box
from agents.user_agent import UserAgent
import sys

def dict_to_action_space(aspace_dict):
    return Box(
                low = aspace_dict["low"],
                high = aspace_dict["high"],
                dtype = aspace_dict["dtype"],
              )


class OrderEnforcingAgent:
    """
    TRY NOT TO CHANGE THIS
    
    Emulates order enforcing wrapper in Pettingzoo for easy integration
    Calls each agent step with agent in a loop and returns the action
    """
    def __init__(self):
        self.num_buildings = None
        self.agent = UserAgent()
        self.action_space = None
    
    def register_reset(self, observation):
        """Get the first observation after env.reset, return action""" 
        action_space = observation["action_space"]
        self.action_space = [dict_to_action_space(asd) for asd in action_space]
        obs = observation["observation"]
        self.num_buildings = len(obs)

        actions = []
        for agent_id in range(self.num_buildings):
            action_space = self.action_space[agent_id]
            actions.append(self.agent.register_reset(obs[agent_id], action_space, agent_id))
        return actions
    
    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def compute_action(self, observation):
        """Get observation return action"""
        assert self.num_buildings is not None
        actions = []
        for agent_id in range(self.num_buildings):
            actions.append(self.agent.compute_action(observation[agent_id], agent_id))
        return actions