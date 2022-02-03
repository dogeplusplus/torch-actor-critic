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
    def __init__(self, env):
        self.env = env
        setup_pytorch_for_mpi()

        act_dim = self.env.action_space.shape[0]
        obs_dim = self.env.observation_space.shape[0]
        act_limit = self.env.action_space.high[0]
        hidden_sizes = [256, 256]

        self.actor = Actor(obs_dim, act_dim, hidden_sizes, act_limit=act_limit)
        self.critic = DoubleCritic(obs_dim, act_dim, hidden_sizes)
        self.target_critic = deepcopy(self.critic)

        for p in self.target_critic.parameters():
            p.requires_grad = False

        sync_params(self.actor)
        sync_params(self.critic)
        sync_params(self.target_critic)

    def update_critic(self, samples: Batch, alpha: float, gamma: float) -> FloatTensor:
        self.q_opt.zero_grad()
        loss_q = eval_q_loss(
            self.actor,
            self.critic,
            self.target_critic,
            samples.states,
            samples.actions,
            samples.rewards,
            samples.next_states,
            samples.done,
            alpha,
            gamma,
        )
        loss_q.backward()
        mpi_avg_grads(self.critic)
        self.q_opt.step()

        return loss_q

    def update_policy(self, samples: Batch, alpha: float) -> FloatTensor:

        for p in self.critic.parameters():
            p.requires_grad = False

        self.pi_opt.zero_grad()
        loss_pi = eval_pi_loss(
            self.actor,
            self.critic,
            samples.states,
            samples.next_states,
            alpha
        )
        mpi_avg_grads(self.actor)
        loss_pi.backward()
        self.pi_opt.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        return loss_pi

    def save_model(self):
        mlflow.pytorch.log_model(self.actor, "actor")
        mlflow.pytorch.log_model(self.critic, "critic")
        mlflow.pytorch.log_model(self.target_critic, "target_critic")

    def train(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        alpha: float,
        gamma: float,
        polyak: float,
        start_steps: int,
        steps_per_epoch: int,
        buffer_size: int,
        max_ep_len: int,
        update_after: int,
        update_every: int,
        save_every: int,
        render: bool = True
    ):
        # Log parameters in mlflow run
        if proc_id() == 0:
            for name, param_value in vars().items():
                if name != "self":
                    mlflow.log_param(name, param_value)

        comm = MPI.COMM_WORLD
        seed = 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        state = self.env.reset()

        self.pi_opt = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_opt = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
        )

        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        step = 0
        ep_ret = 0
        ep_len = 0

        pbar = tqdm.tqdm(range(epochs), ncols=0)
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
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action, _ = self.actor(FloatTensor(state))
                        action = action.cpu().detach().numpy()

                next_state, reward, done, _ = self.env.step(action)

                # Bypass max episode length limitation of gym
                done = False if ep_len == max_ep_len else done

                if render and proc_id() == 0:
                    self.env.render()

                ep_len += 1
                ep_ret += reward

                self.buffer.store(state, action, reward, next_state, done)
                state = next_state

                epoch_ended = t % local_steps_per_epoch == local_steps_per_epoch - 1

                if done or epoch_ended:
                    episode_rewards.append(ep_ret)
                    episode_lengths.append(ep_len)

                    state = self.env.reset()
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
                        samples = self.buffer.sample(batch_size)
                        loss_q = self.update_critic(samples, alpha, gamma)
                        loss_pi = self.update_policy(samples, alpha)
                        update_targets(self.critic, self.target_critic, polyak)

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
                    self.save_model()

                # End of epoch
                mlflow.log_metrics(metrics, e)

            episode_lengths = []
            episode_rewards = []
            losses_pi = []
            losses_q = []

            state = self.env.reset()
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


def main():
    env = gym.make("Humanoid-v3")
    env._max_episode_steps = 10000

    cpus = 1
    mpi_fork(cpus)

    sac = SAC(env)
    sac.train(
        epochs=1000,
        learning_rate=3e-4,
        batch_size=500,
        alpha=0.2,
        gamma=0.99,
        polyak=0.995,
        steps_per_epoch=5000,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        buffer_size=int(1e6),
        max_ep_len=5000,
        save_every=10,
        render=False,
    )


if __name__ == "__main__":
    main()
