import numpy as np

def rbc_policy(observation, action_space):
    """
    Simple rule based policy based on day or night time
    """
    hour = observation[2] # Hour index is 2 for all observations
    
    action = 0.0 # Default value

    multiplier = 0.6
    # Daytime: release stored energy  2*0.08 + 0.1*7 + 0.09

    if hour >= 7 and hour <= 11:
        action = -0.05 * multiplier 
    elif hour >= 12 and hour <= 15:
        action = -0.05 * multiplier
    elif hour >= 16 and hour <= 18:
        action = -0.11 * multiplier
    elif hour >= 19 and hour <= 22:
        action = -0.06 * multiplier 
    
    # Early nightime: store DHW and/or cooling energy
    if hour >= 23 and hour <= 24:
        action = 0.085 * multiplier 
    elif hour >= 1 and hour <= 6:
        action = 0.1383 * multiplier 


    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action

class BasicRBCAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """
    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return rbc_policy(observation, self.action_space[agent_id])