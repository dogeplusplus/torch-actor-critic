import gym
import torch
import mlflow
import logging
import torch.optim as optim

from pathlib import Path
from contextlib import nullcontext
from mlflow.tracking import MlflowClient
from argparse import ArgumentParser, Namespace

from sac.algorithm import SAC
from buffer.replay_buffer import ReplayBuffer
from buffer.visual_replay_buffer import VisualReplayBuffer
from networks.linear import Actor, DoubleCritic
from networks.convolutional import VisualActor, VisualDoubleCritic
from sac.mpi import (
    mpi_fork,
    proc_id,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_session(run_id):
    client = MlflowClient()
    run = client.get_run(run_id)
    sac_params = run.data.params

    artifacts_path = Path("mlruns", "0", run_id, "artifacts")
    actor = mlflow.pytorch.load_model(artifacts_path / "actor")
    critic = mlflow.pytorch.load_model(artifacts_path / "critic")
    auxiliaries = mlflow.pytorch.load_state_dict(artifacts_path / "auxiliaries")

    pi_opt = optim.Adam(actor.parameters())
    pi_opt.load_state_dict(auxiliaries["pi_opt"])

    q_opt = optim.Adam(critic.parameters())
    q_opt.load_state_dict(auxiliaries["q_opt"])
    start_epoch = auxiliaries["epoch"]

    # Remove environment as not needed in the constructor
    sac_params.pop("environment", None)
    sac_params = {
        k: int(float(v)) if float(v).is_integer() else float(v) for k,
        v in sac_params.items()
    }
    return actor, critic, pi_opt, q_opt, start_epoch, sac_params


def init_session(environment: str):
    env = gym.make(environment)

    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    act_limit = env.action_space.high[0]

    hidden_sizes = [256, 256, 256, 256]

    if environment == "DeepMindWallRunner-v0":
        # Extra parameters needed for convolutional policy/critic
        filters = [32, 64, 64]
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        vis_dim = (3, 64, 64)
        actor = VisualActor(
            obs_dim,
            act_dim,
            vis_dim,
            hidden_sizes,
            act_limit,
            filters,
            kernel_sizes,
            strides,
        )
        critic = VisualDoubleCritic(
            obs_dim,
            act_dim,
            vis_dim,
            hidden_sizes,
            filters,
            kernel_sizes,
            strides,
        )
    else:
        actor = Actor(obs_dim, act_dim, hidden_sizes, act_limit=act_limit)
        critic = DoubleCritic(obs_dim, act_dim, hidden_sizes)

    start_epoch = 0
    learning_rate = 3e-4
    pi_opt = optim.Adam(actor.parameters(), lr=learning_rate)
    q_opt = optim.Adam(critic.parameters(), lr=learning_rate)

    return actor, critic, pi_opt, q_opt, start_epoch


def init_buffer(environment: str, size: int) -> ReplayBuffer:
    env = gym.make(environment)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if environment == "DeepMindWallRunner-v0":
        buffer = VisualReplayBuffer(size, act_dim)
    else:
        buffer = ReplayBuffer(size, obs_dim, act_dim)

    return buffer


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
    torch.set_num_threads(2)
    run_id = args.run
    mlflow.set_experiment(args.experiment)

    # Start run only if process id 0
    if proc_id() == 0 and args.logging:
        context = mlflow.start_run(run_id)
    else:
        context = nullcontext()

    buffer_size = int(1e6)
    buffer = init_buffer(args.environment, buffer_size)

    if run_id is not None:
        actor, critic, pi_opt, q_opt, start_epoch, params = load_session(run_id)
    else:
        actor, critic, pi_opt, q_opt, start_epoch = init_session(args.environment)
        params = dict(
            alpha=0.2,
            gamma=0.99,
            polyak=0.995,
            reward_scale=1.,
            epochs=1000,
            batch_size=64,
            steps_per_epoch=5000,
            start_steps=1000,
            update_after=1000,
            update_every=50,
            max_ep_len=5000,
            save_every=10,
        )
        if proc_id() == 0 and args.logging:
            mlflow.log_params(params)
            mlflow.log_param("environment", args.environment)
            mlflow.log_param("buffer_size", buffer_size)

    sac = SAC(**params)
    env = gym.make(args.environment)
    mpi_fork(args.cpus)

    with context:
        sac.train(
            start_epoch=start_epoch,
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
