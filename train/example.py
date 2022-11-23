from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sapai_gym import *
from sapai_gym.opponent_gen.opponent_generators import *

opponent_generator = random_opp_generator


def train_with_masks():
    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)

    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=1, warn=False)
    obs = env.reset()
    
    num_games = 0
    while num_games < 100:
        # Predict outcome with model
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

        obs, reward, done, info = env.step(action)
        if done:
            num_games += 1
            obs = env.reset()
    env.close()

train_with_masks()