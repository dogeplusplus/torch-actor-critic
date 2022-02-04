import gym
import tqdm
import torch
import mlflow
import numpy as np
import torch.nn as nn
import torch.optim as optim

from mpi4py import MPI
from copy import deepcopy
from itertools import count
from torch import FloatTensor
from argparse import ArgumentParser
from pathlib import Path

from sac.buffer import ReplayBuffer, Batch
from sac.networks import Actor, DoubleCritic
from sac.mpi import (
    mpi_avg_grads,
    mpi_fork,
    setup_pytorch_for_mpi,
    sync_params,
    proc_id,
    num_procs,
)


def eval_pi_loss(
    actor: Actor,
    critic: DoubleCritic,
    state: FloatTensor,
    next_state: FloatTensor,
    alpha: float
) -> torch.FloatTensor:
    pi, logp_pi = actor(next_state)
    sa2 = torch.cat([state, pi], dim=1)
    q1, q2 = critic(sa2)

    q_pi = torch.min(q1, q2)
    loss_pi = (alpha * logp_pi - q_pi).mean()

    return loss_pi


def eval_q_loss(
    actor: Actor,
    critic: DoubleCritic,
    target_critic: DoubleCritic,
    states: FloatTensor,
    actions: FloatTensor,
    rewards: FloatTensor,
    next_states: FloatTensor,
    done: FloatTensor,
    alpha: float,
    gamma: float,
) -> torch.FloatTensor:

    a2, logp_ac = actor(next_states)
    sa2 = torch.cat([next_states, a2], dim=-1)
    q1_target, q2_target = target_critic(sa2)
    q_target = torch.min(q1_target, q2_target)

    backup = rewards + gamma * (1 - done) * (
        q_target - alpha * logp_ac
    )

    sa = torch.cat([states, actions], dim=-1)
    q1, q2 = critic(sa)

    loss_q1 = ((q1 - backup) ** 2).mean()
    loss_q2 = ((q2 - backup) ** 2).mean()
    loss_q = loss_q1 + loss_q2

    return loss_q


def update_targets(source: nn.Module, target: nn.Module, polyak: float):
    with torch.no_grad():
        for src, targ in zip(source.parameters(), target.parameters()):
            targ.data.mul_(polyak)
            targ.data.add_((1 - polyak) * src.data)


class SAC(object):
    def __init__(self, alpha, gamma, polyak):
        self.alpha = alpha
        self.gamma = gamma
        self.polyak = polyak

        setup_pytorch_for_mpi()

    def update_critic(
        self,
        q_opt: optim.Optimizer,
        actor: Actor,
        critic: DoubleCritic,
        target_critic: DoubleCritic,
        samples: Batch
    ) -> FloatTensor:
        q_opt.zero_grad()
        loss_q = eval_q_loss(
            actor,
            critic,
            target_critic,
            samples.states,
            samples.actions,
            samples.rewards,
            samples.next_states,
            samples.done,
            self.alpha,
            self.gamma,
        )
        loss_q.backward()
        mpi_avg_grads(critic)
        q_opt.step()

        return loss_q

    def update_policy(self, pi_opt: optim.Optimizer, actor: Actor, critic: DoubleCritic, samples: Batch) -> FloatTensor:

        for p in critic.parameters():
            p.requires_grad = False

        pi_opt.zero_grad()
        loss_pi = eval_pi_loss(
            actor,
            critic,
            samples.states,
            samples.next_states,
            self.alpha
        )
        mpi_avg_grads(actor)
        loss_pi.backward()
        pi_opt.step()

        for p in critic.parameters():
            p.requires_grad = True

        return loss_pi

    def save_model(self, actor, critic, pi_opt, q_opt, epoch):
        mlflow.pytorch.log_model(actor, "actor")
        mlflow.pytorch.log_model(critic, "critic")

        mlflow.pytorch.log_state_dict({
            "pi_opt": pi_opt.state_dict(),
            "q_opt": q_opt.state_dict(),
            "epoch": epoch,
        }, "auxiliaries.pth")

    def train(
        self,
        start_epoch: int,
        epochs: int,
        batch_size: int,
        start_steps: int,
        steps_per_epoch: int,
        max_ep_len: int,
        env: gym.Env,
        actor: Actor,
        critic: DoubleCritic,
        buffer: ReplayBuffer,
        pi_opt: optim.Optimizer,
        q_opt: optim.Optimizer,
        update_after: int,
        update_every: int,
        save_every: int,
        render: bool = True
    ):
        target_critic = deepcopy(critic)
        for p in target_critic.parameters():
            p.requires_grad = False

        sync_params(actor)
        sync_params(critic)
        sync_params(target_critic)

        # Log parameters in mlflow run
        if proc_id() == 0:
            parameters = dict(
                alpha=self.alpha,
                gamma=self.gamma,
                polyak=self.polyak,
                start_steps=start_steps,
                steps_per_epoch=steps_per_epoch,
                max_ep_len=max_ep_len,
                update_after=update_after,
                update_every=update_every,
                save_every=save_every,
                batch_size=batch_size,
            )

            for name, value in parameters.items():
                mlflow.log_param(name, value)

        comm = MPI.COMM_WORLD
        seed = 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        state = env.reset()

        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        step = 0
        ep_ret = 0
        ep_len = 0

        pbar = tqdm.trange(start_epoch, start_epoch + epochs, ncols=0, initial=start_epoch)
        metrics = {
            "episode_length": 0,
            "reward": 0,
            "loss_q": 0,
            "loss_pi": 0,
        }
        for e in pbar:
            episode_rewards = []
            episode_lengths = []
            losses_pi = []
            losses_q = []

            for t in range(local_steps_per_epoch):

                if step < start_steps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action, _ = actor(FloatTensor(state))
                        action = action.cpu().detach().numpy()

                next_state, reward, done, _ = env.step(action)

                # Bypass max episode length limitation of gym
                done = False if ep_len == max_ep_len else done

                if render and proc_id() == 0:
                    env.render()

                ep_len += 1
                ep_ret += reward

                buffer.store(state, action, reward, next_state, done)
                state = next_state

                epoch_ended = t % local_steps_per_epoch == local_steps_per_epoch - 1

                if done or epoch_ended:
                    episode_rewards.append(ep_ret)
                    episode_lengths.append(ep_len)

                    state = env.reset()
                    ep_ret = 0
                    ep_len = 0

                if proc_id() == 0:
                    for id in range(1, num_procs()):
                        extra_rewards = comm.recv(source=id, tag=420)
                        extra_lengths = comm.recv(source=id, tag=1337)

                        episode_rewards.extend(extra_rewards)
                        episode_lengths.extend(extra_lengths)
                else:
                    comm.send(episode_rewards, dest=0, tag=420)
                    comm.send(episode_lengths, dest=0, tag=1337)

                if step > update_after and step % update_every == 0:
                    for _ in range(update_every):
                        samples = buffer.sample(batch_size)
                        loss_q = self.update_critic(q_opt, actor, critic, target_critic, samples)
                        loss_pi = self.update_policy(pi_opt, actor, critic, samples)
                        update_targets(critic, target_critic, self.polyak)

                        losses_pi.append(loss_pi.cpu().detach().numpy())
                        losses_q.append(loss_q.cpu().detach().numpy())

                step += 1

            metrics = {
                "episode_length": np.mean(episode_lengths),
                "reward": np.mean(episode_rewards),
                "loss_q": np.mean(losses_q),
                "loss_pi": np.mean(losses_pi),
            }
            pbar.set_postfix(metrics)
            if proc_id() == 0:
                if e % save_every == 0:
                    self.save_model(actor, critic, pi_opt, q_opt, e)

                # End of epoch
                mlflow.log_metrics(metrics, e)

            episode_lengths = []
            episode_rewards = []
            losses_pi = []
            losses_q = []

            state = env.reset()
            ep_ret = 0
            ep_len = 0


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


def parse_arguments():
    parser = ArgumentParser("Soft Actor-Critic trainer for MuJoCo.")
    parser.add_argument("--run", type=str, default=None, help="Path to pre-existing mlflow run")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    env = gym.make("Humanoid-v3")
    env._max_episode_steps = 10000

    cpus = 1
    mpi_fork(cpus)

    run_id = args.run

    if run_id is not None:
        artifacts_path = Path("mlruns", "0", run_id, "artifacts")

        actor = mlflow.pytorch.load_model(artifacts_path.joinpath("actor"))
        critic = mlflow.pytorch.load_model(artifacts_path.joinpath("critic"))

        auxiliaries = mlflow.pytorch.load_state_dict(artifacts_path.joinpath("auxiliaries.pth"))
        pi_opt = optim.Adam(actor.parameters())
        pi_opt.load_state_dict(auxiliaries["pi_opt"])

        q_opt = optim.Adam(critic.parameters())
        q_opt.load_state_dict(auxiliaries["q_opt"])

        start_epoch = auxiliaries["epoch"]
    else:
        start_epoch = 0
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        act_limit = env.action_space.high[0]
        hidden_sizes = [256, 256]

        actor = Actor(obs_dim, act_dim, hidden_sizes, act_limit=act_limit)
        critic = DoubleCritic(obs_dim, act_dim, hidden_sizes)

        learning_rate = 3e-4

        pi_opt = optim.Adam(actor.parameters(), lr=learning_rate)
        q_opt = optim.Adam(critic.parameters(), lr=learning_rate)

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
    )

    with mlflow.start_run(run_id):
        sac.train(
            start_epoch=start_epoch,
            epochs=1000,
            batch_size=500,
            steps_per_epoch=50,
            start_steps=10000,
            update_after=1000,
            update_every=50,
            max_ep_len=5000,
            save_every=10,
            render=False,
            buffer=buffer,
            env=env,
            actor=actor,
            critic=critic,
            pi_opt=pi_opt,
            q_opt=q_opt,
        )


if __name__ == "__main__":
    main()
