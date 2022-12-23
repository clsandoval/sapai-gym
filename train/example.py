from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sapai_gym import *
from sapai_gym.opponent_gen.opponent_generators import *
import wandb

wandb.init(project="sap-rl")


def train_with_masks():
    opponent_generator = Generator(
        model=None, strategy=biggest_numbers_horizontal_opp_generator
    )
    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)

    # model = MaskablePPO("MlpPolicy", env, verbose=1, device="cuda")
    model = MaskablePPO.load("1", env)
    for i in range(10000):
        try:
            model.learn(total_timesteps=50000)
            model.save("2")
            policy_evaluation = evaluate_policy(
                model, env, n_eval_episodes=20, reward_threshold=0.5, warn=False
            )
            wandb.log({"avg_reward": policy_evaluation[0]}, step=i)
        except:
            continue
    env.close()


def train_with_masks_against_ai():
    random_generator = Generator(model=None, strategy=random_opp_generator)
    for i in range(10000):
        try:
            env_opp = SuperAutoPetsEnv(random_generator, valid_actions_only=True)
            model_opp = MaskablePPO.load("train//sapai_ppo", env_opp)
            opponent_generator = Generator(
                model=model_opp, strategy=agent_opp_generator
            )
            env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
            model = MaskablePPO.load("sapai_ppo", env)
            model.learn(total_timesteps=100000)
            model.save("sapai_ppo")
        except:
            continue
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=1, warn=False)
    obs = env.reset()
    env.close()


def test_engine():
    opponent_generator = Generator(
        model=None, strategy=biggest_numbers_horizontal_opp_generator
    )
    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
    model = MaskablePPO.load("sapai_ppo", env)
    model.learn(total_timesteps=1000000)

    model.save("sapai_ppo")


train_with_masks()
