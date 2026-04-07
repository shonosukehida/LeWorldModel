from gymnasium.envs.registration import register

register(
    id="swm/FrankaPush-v0",
    entry_point="env.franka.gym_env:FrankaPushEnv",
)