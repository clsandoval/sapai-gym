from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sapai_gym import *
from sapai_gym.opponent_gen.opponent_generators import *
from pathlib import Path
import argparse


def main(args):
    """Trains an agent with league training"""
    opponent_generator = Generator(
        model=None,
        strategy=biggest_numbers_horizontal_opp_generator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    main(parser.parse_args())
