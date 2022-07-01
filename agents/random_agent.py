def random_policy(observation, action_space):
    return action_space.sample()

class RandomAgent:
    
    def __init__(self):
        self.action_space = {}
    
    def register_reset(self, observation, action_space, agent_id):
        """Get the first observation after env.reset, return action"""
        self.action_space[agent_id] = action_space
        return random_policy(observation, action_space)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return random_policy(observation, self.action_space[agent_id])

    