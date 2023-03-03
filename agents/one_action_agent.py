class OneActionAgent:

    def __init__(self, action_to_perform):
        self.action_dim = len(action_to_perform)
        self.action_to_perform = action_to_perform  # Array of length action_dim

    def register_reset(self, obs_dict):
        """Get the first observation after env reset, return action"""
        obs = obs_dict["observation"]

        return self.compute_action(obs)

    def compute_action(self, obs):
        """Get observation return action"""
        return self.one_action_policy(obs)

    def one_action_policy(self, obs):
        actions = []
        for bi in range(len(obs)):
            actions.append(self.action_to_perform)

        return actions
