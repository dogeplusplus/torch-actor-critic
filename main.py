import gym
import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from mpi4py import MPI
from copy import deepcopy
from torch import FloatTensor
from dataclasses import dataclass

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


@dataclass(frozen=True)
class TrainingParameters:
    epochs: int
    steps_per_epoch: int
    batch_size: int
    start_steps: int
    update_after: int
    update_every: int
    learning_rate: float
    alpha: float
    gamma: float
    polyak: float
    buffer_size: int = int(1e6)
    max_ep_len: int = 4000


def eval_pi_loss(
    actor: Actor,
    critic: DoubleCritic,
    state: FloatTensor,
    next_state: FloatTensor,
    alpha: float
) -> torch.FloatTensor:
    pi, logp_pi = actor(next_state)
    sa2 = torch.cat([state, pi], dim=-1)
    q1, q2 = critic(sa2)

    q_pi = torch.min(torch.cat([q1, q2], dim=-1))
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
    q_target = torch.where(q1_target < q2_target, q1_target, q2_target)

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
    def __init__(self, env, params: TrainingParameters):
        self.env = env
        self.buffer = ReplayBuffer(
            params.buffer_size,
            env.observation_space.shape[0],
            env.action_space.shape[0],
        )
        self.params = params
        setup_pytorch_for_mpi()


        act_dim = self.env.action_space.shape[0]
        obs_dim = self.env.observation_space.shape[0]
        hidden_sizes = [256, 256]

        self.actor = Actor(obs_dim, act_dim, hidden_sizes)
        self.critic = DoubleCritic(obs_dim, act_dim, hidden_sizes)
        self.target_critic = deepcopy(self.critic)

        for p in self.target_critic.parameters():
            p.requires_grad = False

        sync_params(self.actor)
        sync_params(self.critic)
        sync_params(self.target_critic)

        self.actor
        self.critic
        self.target_critic

        self.pi_opt = optim.Adam(self.actor.parameters(), lr=self.params.learning_rate)
        self.q_opt = optim.Adam(self.critic.parameters(), lr=self.params.learning_rate)


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


    def train(self, render: bool = True):
        comm = MPI.COMM_WORLD
        seed = 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        state = self.env.reset()
        params = self.params

        local_steps_per_epoch = int(params.steps_per_epoch / num_procs())
        step = 0
        ep_ret = 0
        ep_len = 0

        pbar = tqdm.tqdm(range(params.epochs), ncols=0)
        for _ in pbar:
            episode_rewards = []
            episode_lengths = []
            losses_pi = []
            losses_q = []

            for t in range(local_steps_per_epoch):

                if step < params.start_steps:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action, _ = self.actor(
                            FloatTensor(state)
                        )
                        action = action.cpu().detach().numpy()

                next_state, reward, done, _ = self.env.step(action)

                if render and proc_id() == 0:
                    self.env.render()

                ep_len += 1
                ep_ret += reward

                self.buffer.store(state, action, reward, next_state, done)
                state = next_state

                timeout = ep_len == params.max_ep_len
                terminal = done or timeout
                epoch_ended = t % local_steps_per_epoch == local_steps_per_epoch - 1

                if terminal or epoch_ended:
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


                if step > params.update_after and step % params.update_every == 0:
                    for _ in range(params.update_every):
                        samples = self.buffer.sample(params.batch_size)

                        loss_q = self.update_critic(samples, params.alpha, params.gamma)
                        loss_pi = self.update_policy(samples, params.alpha)
                        update_targets(self.critic, self.target_critic, params.polyak)

                        losses_pi.append(loss_pi.cpu().detach().numpy())
                        losses_q.append(loss_q.cpu().detach().numpy())

                        metrics = {
                            "Episode Length": np.mean(episode_lengths),
                            "Cumulative Reward": np.mean(episode_rewards),
                            "Q Loss": np.mean(losses_q),
                            "PI Loss": np.mean(losses_pi),
                        }
                        pbar.set_postfix(metrics)


                step += 1

            # End of epoch
            episode_lengths = []
            episode_rewards = []

            state = self.env.reset()
            ep_ret = 0
            ep_len = 0

def main():
    env = gym.make("Humanoid-v2")

    training_params = TrainingParameters(
        epochs=500,
        steps_per_epoch=10000,
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        learning_rate=3e-4,
        alpha=0.2,
        gamma=0.99,
        polyak=0.995
    )

    cpus = 8 
    mpi_fork(cpus)

    sac = SAC(env, training_params)
    sac.train()


if __name__ == "__main__":
    main()
