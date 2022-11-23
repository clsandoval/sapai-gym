from sapai import *
from random import choice
from sapai_gym import SuperAutoPetsEnv

from typing import Dict, List

def ai_agent(env: SuperAutoPetsEnv, actions: Dict[int,any]) ->int:
    """
    Uses a trained agent to produce the next action
    :param player_to_act: Player to choose action for
    :param actions: Available actions
    :return: Action to play
    """
    obs = env._encode_state()
    action, _states = model.predict(obs,  deterministic=True)
    return action