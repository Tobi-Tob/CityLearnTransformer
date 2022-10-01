from agents.random_agent import RandomAgent
from agents.rbc_agent import BasicRBCAgent

import xgboost as xgb 
import pickle
from joblib import dump, load
from collections import deque
import os
import sys

import numpy as np

str_to_int_mapping = {
    "month"  :   0,
"day_type"  :    1,
"hour"  :    2,
"outdoor_dry_bulb_temperature"  :  3,
"outdoor_dry_bulb_temperature_predicted_6h"  :    4,
"outdoor_dry_bulb_temperature_predicted_12h"  :    5,
"outdoor_dry_bulb_temperature_predicted_24h"  :    6,
"outdoor_relative_humidity"  :   7,
"outdoor_relative_humidity_predicted_6h"  :   8,
"outdoor_relative_humidity_predicted_12h"  :    9,
"outdoor_relative_humidity_predicted_24h"  :  10,
"diffuse_solar_irradiance"  :    11,
"diffuse_solar_irradiance_predicted_6h"  :   12,
"diffuse_solar_irradiance_predicted_12h"  :   13,
"diffuse_solar_irradiance_predicted_24h"  :   14,
"direct_solar_irradiance"  : 15,
"direct_solar_irradiance_predicted_6h"  :   16,
"direct_solar_irradiance_predicted_12h"  :  17,
"direct_solar_irradiance_predicted_24h"  :  18,
"carbon_intensity"  :  19,
"non_shiftable_load"  :  20,
"solar_generation"  :  21,
"electrical_storage_soc"  : 22,
"net_electricity_consumption"  :  23,
"electricity_pricing"  :   24,
"electricity_pricing_predicted_6h"  : 25,
"electricity_pricing_predicted_12h"  :  26,
"electricity_pricing_predicted_24h"  : 27,
}

int_to_str_mapping = {v: k for k, v in str_to_int_mapping.items()}

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
        models = os.listdir("/home/aicrowd/models/")

        # we load all the models
        for model in models:
            
            # change the name of the model for the key in the dictionary
            key = model.split(".")[0]

            self.models[key] = pickle.load(open("/home/aicrowd/models/" + model, 'rb'))

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

        # we init a history of information specific to the agent
        self.history[agent_id] = {var : deque(maxlen=24*7) for var in self.variables_building}

    def adding_observation(self, observation, agent_id):
        """Get observation return action"""

        # add observation to the history
        for var in self.variables_building:
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

            prediction_t[var] = self.models[var].predict(input_data.reshape((1, -1)))[0]

            # custom correction for the hour
            if var == "hour":
                # we add 1h to the data hour last
                prediction_t[var] = (data[var][-1] + 1) % 24

        # we get the model prediction
        return prediction_t

    def augment_observations(self, observation, agent_id):

        observation_dict = {}

        for idx, obs in enumerate(observation):
            observation_dict[int_to_str_mapping[idx]] = obs

        # adding crude electricity consumption and index pricing in observation
        observation_dict['electricity_consumption_crude'] = observation_dict['non_shiftable_load'] - observation_dict['solar_generation']
        observation_dict['index_pricing'] = observation_dict['electricity_pricing'] + observation_dict['carbon_intensity']

        return observation_dict
    
    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        # we augment the observation with the crude electricity consumption and the index pricing
        observation = self.augment_observations(observation, agent_id)

        # add observation to the history
        self.adding_observation(observation, agent_id)

        # get model prediction
        prediction_t = self.get_model_prediction(agent_id)

        # now we can compute the action with rule based behaviour
        action = simple_rule_optim(prediction_t)

        return [action]