import numpy


def base_reward(player):
    return player.wins / 10


def evolution_tier_augmented_reward(player):
    # rewards the agent for evolving pets, scales the reward by the pet tier and is normalized by the maximum amount ( 6 tiers, 3 levels, 5 pets)
    tier_evolution_reward = 0
    for slot in player.team.slots:
        if (
            slot.obj.name != "pet-none"
            and str(slot.obj.tier).isdigit()
            and str(slot.obj.level).isdigit()
        ):
            tier_evolution_reward += int(slot.obj.tier) * int(slot.obj.level)
    tier_evolution_reward = tier_evolution_reward / 90
    normalized_wins = player.wins / 10
    return tier_evolution_reward + normalized_wins
