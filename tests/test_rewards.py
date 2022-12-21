from unittest import TestCase
from sapai import Team
from sapai_gym.opponent_gen.opponent_generators import *
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


class TestRewards(TestCase):
    def test_base_reward(self):
        random_generator = Generator(model=None, strategy=random_opp_generator)
        env = SuperAutoPetsEnv(random_generator, valid_actions_only=True)
        obs = env.reset()
        model = MaskablePPO.load("1.zip", env)
        # Predict outcome with model
        action_masks = get_action_masks(env)
        action, _states = model.predict(
            obs, action_masks=action_masks, deterministic=True
        )

        obs, reward, done, info = env.step(action)
        if done:
            num_games += 1
            obs = env.reset()
        env.close()
