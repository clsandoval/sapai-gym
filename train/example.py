from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sapai_gym import *
from sapai_gym.opponent_gen.opponent_generators import *


def train_with_masks():
    opponent_generator = Generator(model=None, strategy=biggest_numbers_horizontal_opp_generator)
    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)

    #model = MaskablePPO("MlpPolicy", env, verbose=1,device='cuda' )
    model = MaskablePPO.load("sapai_ppo", env)
    for i in range(1000):
        try:
            model.learn(total_timesteps=100000)
        except:
            continue
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=.5, warn=False)
    obs = env.reset()
    model.save("sapai_ppo")
    
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


def train_with_masks_against_ai():

    random_generator = Generator(model=None, strategy=random_opp_generator)

    env_opp = SuperAutoPetsEnv(random_generator, valid_actions_only=True)
    model_opp = MaskablePPO.load("train//sapai_ppo", env_opp)

    opponent_generator = Generator(model = model_opp, strategy=agent_opp_generator)

    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=1, warn=False)
    obs = env.reset()
    model.save("sapai_ppo")
    
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