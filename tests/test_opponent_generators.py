from unittest import TestCase
from sapai import Team
from sapai_gym.opponent_gen.opponent_generators import *
class TestOpponentGenerators(TestCase):
    def test_random_generator(self):
        generator = Generator(strategy=random_opp_generator)
        opponents = generator.generate(25)

    def test_biggest_numbers_horizontal_opp_generator(self):
        generator = Generator(strategy=biggest_numbers_horizontal_opp_generator)
        opponents = generator.generate(25)
        scores = [self.map_team_to_total_attack_and_health(team) for team in opponents]
        sorted_scores = sorted(scores)

        # Check that the team is always getting stronger
        self.assertEqual(scores, sorted_scores)

    def test_agent_opponent_generator(self):
        random_generator = Generator(model=None, strategy=random_opp_generator)
        env_opp = SuperAutoPetsEnv(random_generator, valid_actions_only=True)
        model_opp = MaskablePPO.load("train//sapai_ppo", env_opp)

        generator = Generator(model = model_opp, strategy=agent_opp_generator)
        opponents = generator.generate(25)
        for t in opponents:
            print(t)

    @staticmethod
    def map_team_to_total_attack_and_health(team: Team):
        total = 0
        for pet_slot in team:
            pet = pet_slot.pet
            if pet.name != "pet-none":
                total += pet._attack
                total += pet._health
        return total
