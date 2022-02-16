import gym
import numpy as np
import typing as t

from dataclasses import dataclass
from dm_control.locomotion.examples import basic_cmu_2019


@dataclass
class MultiObservation:
    features: np.ndarray
    camera: np.ndarray


class DeepMindWallRunner(gym.Env):
    def __init__(self):
        self.env = basic_cmu_2019.cmu_humanoid_run_walls()

    def reset(self):
        time_step = self.env.reset()
        observation = self.process_observations(time_step.observation)

        return observation

    def step(self, action: np.ndarray):
        time_step = self.env.step(action)

        done = time_step.last()
        reward = time_step.reward
        observation = self.process_observations(time_step.observation)

        return observation, reward, done, None

    def process_observations(self, obs: t.List[np.ndarray]):
        filtered_obs = [
            obs["walker/appendages_pos"],
            obs["walker/body_height"][np.newaxis, ...],
            obs["walker/end_effectors_pos"],
            obs["walker/joints_pos"],
            obs["walker/joints_vel"],
            obs["walker/sensors_accelerometer"],
            obs["walker/sensors_force"],
            obs["walker/sensors_gyro"],
            obs["walker/sensors_torque"],
            obs["walker/sensors_touch"],
            obs["walker/sensors_velocimeter"],
            obs["walker/world_zaxis"],
        ]

        return MultiObservation(
            np.concatenate(filtered_obs),
            obs["walker/egocentric_camera"],
        )

    def render(self):
        pass
