from agents.random_agent import RandomAgent
from agents.rbc_agent import BasicRBCAgent

import xgboost as xgb 
import pickle
from collections import deque
import os
import sys

import numpy as np

###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
# UserAgent = BasicRBCAgent

def simple_rule_optim(observation_t):
    """
    This is a simple rule to optimize the energy consumption

    Basicly the rule is to store the energy when the net load is negative (energy gain due to solar)
    Spend it after 16h when the net load is positive (energy consumption)

    """
    # get the temperature
    action = 0

    if observation_t['electricity_consumption_crude'] < 0:
        action = action - observation_t['electricity_consumption_crude']/6.4

    if (observation_t['hour'] >= 16 or observation_t['hour'] <= 7) and observation_t['electricity_consumption_crude'] > 0:
              action = action - observation_t['electricity_consumption_crude']/6.4

    return action

class UserAgent:
    def __init__(self):

        self.action_space = {}

        # we init a history of information specific to the agent
        self.history = {}

        self.variables_building = ['solar_generation', 'non_shiftable_load', 'electricity_pricing', 'carbon_intensity', "electricity_consumption_crude",
                                     'hour']

        # we load all the models
        self.models = {}

        self.lookback_info = 24

        # we list the models available in the folder models/ repository
        models = os.listdir("./models/")

        # we load all the models
        for model in models:
            
            # change the name of the model for the key in the dictionary
            key = model.split(".")[0]

            self.models[key] = pickle.load(open("models/" + model, 'rb'))

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

        # we init a history of information specific to the agent
        self.history[agent_id] = {var : deque(maxlen=24*7) for var in self.variables_building}

    def adding_observation(self, observation, agent_id):
        """Get observation return action"""

        # add observation to the history
        for var in range(len(self.variables_building)):
            self.history[agent_id][var].append(observation[var])
    
    def get_model_prediction(self, agent_id):
        """Get observation return action"""

        # we get the last 24*7 observations for every variables using history variable
        # we need to reshape the data to be able to use the model
        data = {var : np.array(self.history[agent_id][var]).reshape(1, -1) for var in self.variables_building}

        prediction_t = {}

        # data is shape (nb_variables, nb_observation)
        # now we can make the prediction for each variable
        for var in self.variables_building:
            
            # we change a little bit the input to be able to use the model (the model take (1, lookback) as input)
            if data[var].shape[1] >= self.lookback_info:
                input_data = data[var][:, -self.lookback_info:]
            else:
                input_data = np.zeros((self.lookback_info, ))*np.nan
                input_data[-data[var].shape[1]:] = data[var]

            prediction_t[var] = self.models[var].predict(data[var])[0]

        # we get the model prediction
        return prediction_t
    
    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        # add observation to the history
        self.adding_observation(observation, agent_id)

        # get model prediction
        prediction_t = self.get_model_prediction(agent_id)

        # now we can compute the action with rule based behaviour
        action = simple_rule_optim(prediction_t)

        return action