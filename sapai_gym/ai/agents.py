from sapai import *
from sb3_contrib.common.maskable.utils import get_action_masks
from random import choice
from sapai_gym import SuperAutoPetsEnv

from typing import Dict, List

def ai_agent(env: SuperAutoPetsEnv, actions: Dict[int,any], model) ->int:
    """
    Uses a trained agent to produce the next action
    :param player_to_act: Player to choose action for
    :param actions: Available actions
    :return: Action to play
    """
    obs = env._encode_state()
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
    return 0