import numpy


def count_tiers(player):
    reward = 0
    for slot in player.teams.slots:
        if type(slot).__name__ == "Pet":
            reward += slot.tier
    return reward / 25


def count_evolutions(player):
    reward = 0
    for slot in player.teams.slots:
        if type(slot).__name__ == "Pet":
            reward += slot.level
    return reward / 15


def base_reward(player):
    return player.wins / 10


def evolution_augmented_reward(player):
    normalized_wins = player.wins / 10
    normalized_evolutions = count_evolutions(player)
    return normalized_evolutions + normalized_wins
