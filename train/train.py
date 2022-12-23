from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sapai_gym import *
from sapai_gym.opponent_gen.opponent_generators import *
from pathlib import Path
import os
import argparse
import wandb


MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/")


def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init():
        opponent_generator = Generator(
            model=None, strategy=biggest_numbers_horizontal_opp_generator
        )
        env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def main(args):
    """Trains an agent with league training"""
    wandb.init(project="sap-rl")
    model_evals = [0 for i in range(20)]
    for i in range(10000):
        try:
            model_path = os.path.join(MODEL_DIR, "{}".format(i % 20))

            # train agent against top 20 recent performing agents
            # initialize self play environments with ai generators
            random_generator = Generator(model=None, strategy=random_opp_generator)
            env_opp = SuperAutoPetsEnv(random_generator, valid_actions_only=True)
            model_opp = MaskablePPO.load(model_path, env_opp)
            opponent_generator = Generator(
                model=model_opp, strategy=agent_opp_generator
            )

            # initialize main agent and train with 50k steps
            env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
            model = MaskablePPO.load(os.path.join(MODEL_DIR, "current"), env)
            model.learn(total_timesteps=50000)
            policy_evaluation = evaluate_policy(
                model, env, n_eval_episodes=20, reward_threshold=0.5, warn=False
            )
            eval = policy_evaluation[0]
            wandb.log({"avg_reward": eval}, step=i)

            # decide what agents to keep and drop from top 20 list
            model_evals.append(eval)
            model_evals.sort()
            model_evals = model_evals[1:]
            for val, idx in enumerate(model_evals):
                if val == eval:
                    print("Replacing model{} with an eval score of {}".format(idx, val))
                    model.save(model_path)
            model.save(os.path.join(MODEL_DIR, "current"))
        except:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
