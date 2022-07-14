# Reward Function

CityLearn allows for custom reward function design. The CityLearn Challenge 2022 provides an interface for custom user reward function. 

Participants are to edit the `get_reward()` function in [get_reward.py](get_reward.py). Three observations from the environment are provided for the reward calculation and they include:

1. `electricity_consumption`: List of each building's/total district electricity consumption in [kWh].
2. `carbon_emission`: List of each building's/total district carbon emissions in [kg_co2].
3. `electricity_price`: List of each building's/total district electricity price in [$].

By default, the reward function defined in `get_reward()` is:
$$
\textrm{reward}_n = \textrm{min}(-G_n, 0) + \textrm{min}(-C_n, 0)
$$

Where `G_n` and `C_n` are respectively the `carbon_emission` and `electricity_price` of the building(s) controlled by agent `n`.

__Note__ that `get_reward()` function must return a `list` whose length is equal to the number of agents in the environment i.e. I the environment uses a single agent, the length of the list is equal to 1 else the length is equal to the number of buildings in the environment.

Do not edit [user_reward.py](user_reward.py) module!

