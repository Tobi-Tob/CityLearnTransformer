# Add your agents here

Refer to the random agent (`random_agent.py`) and rule based agent (`rbc_agent.py`) examples and create your agents in the same format

## What's OrderEnforicingWrapper?

This is a reference wrapper to allow compatibilty with pettingzoo baselines. You can change the `compute_action` function if you want. Please make sure the order of actions matches the observations.

## Agent ID

To make things compatible with PettingZoo, a reference wrapper is provided that provides observations for each building (referred by agent id).

Add your agent code in a way such that the actions returned are conditioned on the `agent_id`. Note that different buildings can have different action spaces.

