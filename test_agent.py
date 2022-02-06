import gym
import tqdm
import torch
import mlflow
import logging
import torch.nn as nn

from pathlib import Path
from itertools import count
from torch import FloatTensor
from argparse import ArgumentParser, Namespace

from sac.utils import WelfordVarianceEstimate, StateNormalizer, Identity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_agent(
    actor: nn.Module,
    env: gym.Env,
    episodes: int,
    normalizer: StateNormalizer,
    deterministic: bool = True,
    render: bool = True
):
    for e in range(episodes):
        done = False
        state = env.reset()
        state = normalizer.normalize_state(state, update=False)

        for _ in tqdm.tqdm(count(), desc=f"Epoch {e}"):
            action, _ = actor(FloatTensor(state), deterministic=deterministic)
            state, _, done, _ = env.step(action.detach().numpy())
            state = normalizer.normalize_state(state, update=False)

            if render:
                env.render()

            if done:
                break


def parse_arguments() -> Namespace:
    parser = ArgumentParser("Soft Actor-Critic trainer for MuJoCo.")
    parser.add_argument("--run", type=str, help="Path to pre-existing mlflow run")
    parser.add_argument("--episodes", type=int, default=100, help="Number of test episodes")
    parser.add_argument("--headless", action="store_false", dest="render", help="Flag to disable rendering")
    parser.add_argument("--random", action="store_false", dest="deterministic", help="Stochastic policy")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    env = gym.make("HalfCheetah-v2")
    artifact_path = Path("mlruns", "0", args.run, "artifacts")
    auxiliaries_path = artifact_path / "auxiliaries" / "state_dict.pth"
    auxiliaries = torch.load(auxiliaries_path)

    try:
        normalizer = WelfordVarianceEstimate()
        normalizer.load_state(auxiliaries)
    except KeyError as e:
        logging.info(f"Welford parameter {str(e)} not detected. No normalization will be applied")
        normalizer = Identity()

    actor_path = artifact_path / "actor"
    actor = mlflow.pytorch.load_model(actor_path)
    actor.eval()

    test_agent(actor, env, args.episodes, normalizer, args.deterministic, args.render)


if __name__ == "__main__":
    main()
