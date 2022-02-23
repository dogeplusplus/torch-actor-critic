import gym
import tqdm
import mlflow
import logging
import torch.nn as nn

from pathlib import Path
from itertools import count
from torch import FloatTensor
from mlflow.tracking import MlflowClient
from argparse import ArgumentParser, Namespace


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_agent(
    actor: nn.Module,
    env: gym.Env,
    episodes: int,
    deterministic: bool = True,
    render: bool = True
):
    for e in range(episodes):
        metrics = {
            "ep_ret": 0,
            "ep_len": 0
        }
        done = False
        state = env.reset()

        pbar = tqdm.tqdm(count(), desc=f"Epoch {e}")

        for _ in pbar:
            action, _ = actor(FloatTensor(state), deterministic=deterministic)
            state, reward, done, _ = env.step(action.detach().numpy())
            metrics["ep_len"] += 1
            metrics["ep_ret"] += reward

            pbar.set_postfix(metrics)

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
    artifact_path = Path("mlruns", "0", args.run, "artifacts")

    client = MlflowClient()
    run = client.get_run(args.run)
    sac_params = run.data.params

    # Default env to humanoid to account for legacy experiments where not recorded
    environment = sac_params.get("environment", "Humanoid-v2")
    env = gym.make(environment)

    actor_path = artifact_path / "actor"
    actor = mlflow.pytorch.load_model(actor_path)
    actor.eval()

    run_agent(actor, env, args.episodes, args.deterministic, args.render)


if __name__ == "__main__":
    main()
