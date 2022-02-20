import pytest
import numpy as np

from environments.wall_runner import DeepMindWallRunner


@pytest.fixture
def environment():
    env = DeepMindWallRunner()
    return env


def test_reset(environment):
    state = environment.reset()

    assert len(state.features) == 168
    assert state.frame.size() == (3, 64, 64)


def test_step(environment):
    _ = environment.reset()
    action = np.random.random((56,))
    state, reward, done, _ = environment.step(action)

    assert len(state.features) == 168
    assert state.frame.size() == (3, 64, 64)

    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_render(environment):
    # Check it runs without crashing
    environment.render()
