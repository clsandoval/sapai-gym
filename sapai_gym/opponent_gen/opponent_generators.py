from sb3_contrib import MaskablePPO

from sapai import Player, Team
from sapai_gym.ai import baselines, agents
from sapai_gym import SuperAutoPetsEnv

# TODO : Wrap the ai to create a generator


def _do_store_phase(env: SuperAutoPetsEnv, ai, model = None):
    env.player.start_turn()

    while True:
        actions = env._avail_actions()
        if model:
            chosen_action = ai(env, actions, model)
        else:
            chosen_action = ai(env, actions)
            
        env.resolve_action(chosen_action)

        if SuperAutoPetsEnv._get_action_name(actions[chosen_action]) == "end_turn":
            return

def opp_generator(num_turns, ai, model = None):
    opps = list()
    env = SuperAutoPetsEnv(None, valid_actions_only=True, manual_battles=True)
    while env.player.turn <= num_turns:
        _do_store_phase(env, ai, model)
        opps.append(Team.from_state(env.player.team.state))
    return opps


def random_opp_generator(num_turns):
    return opp_generator(num_turns, baselines.random_agent)


def biggest_numbers_horizontal_opp_generator(num_turns):
    return opp_generator(num_turns, baselines.biggest_numbers_horizontal_scaling_agent)


def agent_opp_generator(num_turns, model):
    return opp_generator(num_turns, agents.ai_agent, model )

class Generator():
    def __init__(self, model=None, strategy=None) -> None:
        self.model = model
        self.strategy = strategy

    def generate(self,num_turns):
        if self.model:
            return self.strategy(num_turns, self.model)
        return self.strategy(num_turns)