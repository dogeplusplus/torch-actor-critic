import pytest
import numpy as np

from environments.wall_runner import DeepMindWallRunner


@pytest.fixture
def environment():
    env = DeepMindWallRunner()
    return env


def test_reset(environment):
    state = environment.reset()

    assert state.features.size == 168
    assert state.camera.shape == (64, 64, 3)


def test_step(environment):
    _ = environment.reset()
    action = np.random.random((56,))
    state, reward, done, _ = environment.step(action)

    assert state.features.size == 168
    assert state.camera.shape == (64, 64, 3)

    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_render(environment):
    # Check it runs without crashing
    environment.render()
