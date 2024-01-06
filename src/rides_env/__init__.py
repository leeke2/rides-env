from gymnasium.envs.registration import register

from .env import RidesEnv

register(id="Rides-v0", entry_point="rides_env:RidesEnv")
