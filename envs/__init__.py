from gymnasium.envs.registration import register
from .bite_env import BiteEnv

register(
    id='BiteOptimizer-v0',
    entry_point='envs.bite_env:BiteEnv',
)
