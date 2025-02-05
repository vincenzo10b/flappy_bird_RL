from gymnasium.envs.registration import register
from envs.flappy_env_simple import FlappyBirdEnv

register(
    id="flappy/FlappyBird-v0",
    entry_point="flappy.envs:FlappyBirdEnv",
)