import gym
import tqdm
import torch
import mlflow
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim


from mpi4py import MPI
from copy import deepcopy
from torch import FloatTensor

from buffer.replay_buffer import ReplayBuffer, Batch
from sac.mpi import (
    mpi_avg_grads,
    setup_pytorch_for_mpi,
    sync_params,
    proc_id,
    num_procs,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def eval_pi_loss(
    actor: nn.Module,
    critic: nn.Module,
    state: FloatTensor,
    next_state: FloatTensor,
    alpha: float
) -> torch.FloatTensor:
    pi, logp_pi = actor(next_state)
    q1, q2 = critic(state, pi)

    q_pi = torch.min(q1, q2)
    loss_pi = (alpha * logp_pi - q_pi).mean()

    return loss_pi


def eval_q_loss(
    actor: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    states: FloatTensor,
    actions: FloatTensor,
    rewards: FloatTensor,
    next_states: FloatTensor,
    done: FloatTensor,
    alpha: float,
    gamma: float,
    reward_scale: float,
) -> torch.FloatTensor:

    with torch.no_grad():
        a2, logp_ac = actor(next_states)
        q1_target, q2_target = target_critic(next_states, a2)
        q_target = torch.min(q1_target, q2_target)

        backup = reward_scale * rewards + gamma * (1 - done) * (
            q_target - alpha * logp_ac
        )

    q1, q2 = critic(states, actions)
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
    def __init__(
        self,
        alpha: float,
        gamma: float,
        polyak: float,
        reward_scale: float,
        epochs: int,
        batch_size: int,
        start_steps: int,
        steps_per_epoch: int,
        max_ep_len: int,
        update_after: int,
        update_every: int,
        save_every: int,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.polyak = polyak
        self.reward_scale = reward_scale
        self.epochs = epochs
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_ep_len = max_ep_len
        self.update_after = update_after
        self.update_every = update_every
        self.save_every = save_every

        setup_pytorch_for_mpi()

    def update_critic(
        self,
        q_opt: optim.Optimizer,
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
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
            self.reward_scale,
        )
        loss_q.backward()
        mpi_avg_grads(critic)
        q_opt.step()

        return loss_q

    def update_policy(self, pi_opt: optim.Optimizer, actor: nn.Module, critic: nn.Module, samples: Batch) -> FloatTensor:
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

    def save_model(
        self,
        actor: nn.Module,
        critic: nn.Module,
        pi_opt: optim.Optimizer,
        q_opt: optim.Optimizer,
        epoch: int,
    ):
        mlflow.pytorch.log_model(actor, "actor")
        mlflow.pytorch.log_model(critic, "critic")

        auxiliaries_path = "auxiliaries"
        mlflow.pytorch.log_state_dict({
            "pi_opt": pi_opt.state_dict(),
            "q_opt": q_opt.state_dict(),
            "epoch": epoch,
        }, artifact_path=auxiliaries_path)

    def train(
        self,
        start_epoch: int,
        env: gym.Env,
        actor: nn.Module,
        critic: nn.Module,
        buffer: ReplayBuffer,
        pi_opt: optim.Optimizer,
        q_opt: optim.Optimizer,
        render: bool = True,
        logging: bool = True,
    ):
        target_critic = deepcopy(critic)
        for p in target_critic.parameters():
            p.requires_grad = False

        sync_params(actor)
        sync_params(critic)
        sync_params(target_critic)

        comm = MPI.COMM_WORLD
        seed = 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        state = env.reset()

        step = 0
        ep_ret = 0
        ep_len = 0

        pbar = tqdm.trange(start_epoch, start_epoch + self.epochs, ncols=0, initial=start_epoch)
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

            for t in range(self.steps_per_epoch):
                if step < self.start_steps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # Handle case with dm_control vs generic gym environment
                        try:
                            action, _ = actor(state)
                        except TypeError:
                            action, _ = actor(FloatTensor(state))
                        action = action.cpu().detach().numpy()

                next_state, reward, done, _ = env.step(action)

                # Bypass max episode length limitation of gym
                done = False if ep_len == self.max_ep_len else done

                if render and proc_id() == 0:
                    env.render()

                ep_len += 1
                ep_ret += reward

                buffer.store(state, action, reward, next_state, done)
                state = next_state

                epoch_ended = t % self.steps_per_epoch == self.steps_per_epoch - 1

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

                if step > self.update_after and step % self.update_every == 0:
                    for _ in range(self.update_every):
                        samples = buffer.sample(self.batch_size)
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
            if proc_id() == 0 and logging:
                if e % self.save_every == 0:
                    self.save_model(actor, critic, pi_opt, q_opt, e)

                # End of epoch
                mlflow.log_metrics(metrics, e)

            metrics["step"] = step
            pbar.set_postfix(metrics)
            episode_lengths = []
            episode_rewards = []
            losses_pi = []
            losses_q = []

            state = env.reset()
            ep_ret = 0
            ep_len = 0
