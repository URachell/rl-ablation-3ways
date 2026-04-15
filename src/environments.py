import numpy as np
import random
import collections

# ═══════════════════════════════════════════════════════════
#  Environment Difficulty Factory
# ═══════════════════════════════════════════════════════════

DIFFICULTY_CONFIG = {
    'easy': {
        'num_agents': 8,
        'grid_size': (35, 17),
        'num_obstacles': 40,
        'max_energy': 130.0,
        'max_steps': 250
    },
    'medium': {
        'num_agents': 12,
        'grid_size': (50, 25),
        'num_obstacles': 60,
        'max_energy': 150.0,
        'max_steps': 300
    },
    'hard': {
        'num_agents': 16,
        'grid_size': (70, 35),
        'num_obstacles': 100,
        'max_energy': 180.0,
        'max_steps': 350
    },
    'extreme': {
        'num_agents': 20,
        'grid_size': (100, 50),
        'num_obstacles': 150,
        'max_energy': 200.0,
        'max_steps': 400
    }
}


def create_env(difficulty='easy', use_frame_stack=True):
    """Factory function to create environment based on difficulty"""
    from train import MultiAgentEnergyEnv
    
    config = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG['easy'])
    return MultiAgentEnergyEnv(
        num_agents=config['num_agents'],
        use_frame_stack=use_frame_stack,
        grid_size=config['grid_size'],
        max_energy=config['max_energy'],
        max_steps=config['max_steps']
    )
