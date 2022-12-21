from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sapai_gym import *
from sapai_gym.opponent_gen.opponent_generators import *
from pathlib import Path
import argparse
import wandb


def main(args):
    """Trains an agent with league training"""
    wandb.init(project="sap-rl")
    opponent_generator = Generator(
        model=None,
        strategy=biggest_numbers_horizontal_opp_generator,
    )
    opponent_generator = Generator(
        model=None, strategy=biggest_numbers_horizontal_opp_generator
    )
    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)

    model = MaskablePPO("MlpPolicy", env, verbose=1, device="cuda")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    main(parser.parse_args())
