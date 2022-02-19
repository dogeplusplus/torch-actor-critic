import gym
import torch
import mlflow
import logging
import typing as t
import torch.nn as nn
import torch.optim as optim


from pathlib import Path
from contextlib import nullcontext
from argparse import ArgumentParser, Namespace

from buffer.replay_buffer import ReplayBuffer
from buffer.visual_replay_buffer import VisualReplayBuffer
from networks.linear import Actor, DoubleCritic
from networks.convolutional import VisualActor, VisualDoubleCritic
from sac.mpi import (
    mpi_fork,
    proc_id,
)

from sac.algorithm import SAC


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def agent_configuration(environment: str) -> t.Tuple[nn.Module, nn.Module, t.Any]:
    if environment == "DeepMindWallRunner-v0":
        return (VisualActor, VisualDoubleCritic, VisualReplayBuffer)
    else:
        return (Actor, DoubleCritic, ReplayBuffer)


def parse_arguments() -> Namespace:
    parser = ArgumentParser("Soft Actor-Critic trainer for MuJoCo.")
    parser.add_argument("--run", type=str, default=None, help="Path to pre-existing mlflow run")
    parser.add_argument("--experiment", default="Default", help="Mlflow experiment name")
    parser.add_argument("--disable-logging", dest="logging", action="store_false", help="Turn off logging")
    parser.add_argument("--render", dest="render", action="store_true", help="Enable environment rendering")
    parser.add_argument("--environment", default="Humanoid-v2", help="Environment to use")
    parser.add_argument("--cpus", default=1, help="Number of cpu cores to use.")
    parser.set_defaults(logging=True)
    parser.set_defaults(render=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    environment = args.environment
    env = gym.make(environment)
    env._max_episode_steps = 5000

    torch.set_num_threads(2)
    mpi_fork(args.cpus)

    run_id = args.run
    mlflow.set_experiment(args.experiment)

    if run_id is not None:
        artifacts_path = Path("mlruns", "0", run_id, "artifacts")
        actor = mlflow.pytorch.load_model(artifacts_path / "actor")
        critic = mlflow.pytorch.load_model(artifacts_path / "critic")
        auxiliaries = mlflow.pytorch.load_state_dict(artifacts_path / "auxiliaries")

        pi_opt = optim.Adam(actor.parameters())
        pi_opt.load_state_dict(auxiliaries["pi_opt"])

        q_opt = optim.Adam(critic.parameters())
        q_opt.load_state_dict(auxiliaries["q_opt"])

        start_epoch = auxiliaries["epoch"]

    else:
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        act_limit = env.action_space.high[0]

        hidden_sizes = [256, 256]

        actor = Actor(obs_dim, act_dim, hidden_sizes, act_limit=act_limit)
        critic = DoubleCritic(obs_dim, act_dim, hidden_sizes)

        learning_rate = 3e-4
        pi_opt = optim.Adam(actor.parameters(), lr=learning_rate)
        q_opt = optim.Adam(critic.parameters(), lr=learning_rate)

        start_epoch = 0

    buffer_size = int(1e6)
    buffer = ReplayBuffer(
        buffer_size,
        env.observation_space.shape[0],
        env.action_space.shape[0],
    )

    sac = SAC(
        alpha=0.2,
        gamma=0.99,
        polyak=0.995,
        reward_scale=10.,
    )

    # Start run only if process id 0
    if proc_id() == 0 and args.logging:
        context = mlflow.start_run(run_id)
        mlflow.log_param("environment", environment)
    else:
        context = nullcontext()

    with context:
        sac.train(
            start_epoch=start_epoch,
            epochs=1000,
            batch_size=64,
            steps_per_epoch=5000,
            start_steps=10000,
            update_after=10000,
            update_every=50,
            max_ep_len=5000,
            save_every=10,
            buffer=buffer,
            env=env,
            actor=actor,
            critic=critic,
            pi_opt=pi_opt,
            q_opt=q_opt,
            render=args.render,
            logging=args.logging,
        )


if __name__ == "__main__":
    main()
