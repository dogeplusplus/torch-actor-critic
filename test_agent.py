import gym
import tqdm
import mlflow
import torch.nn as nn

from pathlib import Path
from itertools import count
from torch import FloatTensor
from argparse import ArgumentParser, Namespace


def test_agent(
    actor: nn.Module,
    env: gym.Env,
    episodes: int,
    deterministic: bool = True,
    render: bool = True
):

    for e in range(episodes):
        done = False
        state = env.reset()
        for _ in tqdm.tqdm(count(), desc=f"Epoch {e}"):
            action, _ = actor(FloatTensor(state), deterministic=deterministic)
            state, _, done, _ = env.step(action.detach().numpy())

            if render:
                env.render()

            if done:
                break


def parse_arguments() -> Namespace:
    parser = ArgumentParser("Soft Actor-Critic trainer for MuJoCo.")
    parser.add_argument("--run", type=str, help="Path to pre-existing mlflow run")
    parser.add_argument("--episodes", type=int, default=100, help="Number of test episodes")
    parser.add_argument("--render", type=bool, default=True, help="Flag to enable rendering")
    parser.add_argument("--deterministic", type=bool, default=True, help="Deterministic policy")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    env = gym.make("Humanoid-v3")
    actor_path = Path("mlruns", "0", args.run, "artifacts", "actor")
    actor = mlflow.pytorch.load_model(actor_path)
    test_agent(actor, env, args.episodes, args.deterministic, args.render)


if __name__ == "__main__":
    main()
