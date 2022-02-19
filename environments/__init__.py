from gym.envs.registration import register


register(
    id="DeepMindWallRunner-v0",
    entry_point="environments.wall_runner:DeepMindWallRunner"
)
