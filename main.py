import gym
import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from copy import deepcopy
from torch import FloatTensor
from dataclasses import dataclass

from sac.buffer import ReplayBuffer
from sac.networks import Actor, DoubleCritic


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
    state: np.ndarray,
    next_state: np.ndarray,
    alpha: float
) -> torch.FloatTensor:
    pi, logp_pi = actor(next_state)
    sa2 = torch.cat([state, pi], axis=-1)
    with torch.no_grad():
        q1, q2 = critic(sa2)

    q_pi = torch.min(torch.cat([q1, q2], axis=-1))
    loss_pi = torch.mean(alpha * logp_pi - q_pi)

    return loss_pi


def eval_q_loss(
    actor: Actor,
    critic: DoubleCritic,
    target_critic: DoubleCritic,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    done: np.ndarray,
    alpha: float,
    gamma: float,
) -> torch.FloatTensor:

    with torch.no_grad():
        a2, logp_ac = actor(next_states)
        sa2 = torch.cat([next_states, a2], axis=-1)
        q1_target, q2_target = target_critic(sa2)
        q_target = torch.where(q1_target < q2_target, q1_target, q2_target)

        backup = rewards + gamma * (1 - done) * (
            q_target - alpha * logp_ac
    )

    sa = torch.cat([states, actions], axis=-1)
    q1, q2 = critic(sa)

    loss_q1 = torch.mean((q1 - backup) ** 2)
    loss_q2 = torch.mean((q2 - backup) ** 2)
    loss_q = loss_q1 + loss_q2

    return loss_q


def update_targets(source: nn.Module, target: nn.Module, polyak: float):
    import pdb; pdb.set_trace()
    pass


class SAC(object):
    def __init__(self, env, params: TrainingParameters):
        self.env = env
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.buffer = ReplayBuffer(
            params.buffer_size,
            env.observation_space.shape[0],
            env.action_space.shape[0],
            self.device,
        )
        self.params = params


    def train(self):
        state = self.env.reset()
        params = self.params

        step = 0

        act_dim = self.env.action_space.shape[0]
        obs_dim = self.env.observation_space.shape[0]
        hidden_sizes = [256, 256]

        actor = Actor(obs_dim, act_dim, hidden_sizes)
        double_critic = DoubleCritic(obs_dim, act_dim, hidden_sizes)
        target_critic = deepcopy(double_critic)

        actor.to(self.device)
        double_critic.to(self.device)
        target_critic.to(self.device)

        pi_opt = optim.Adam(actor.parameters())
        q_opt = optim.Adam(double_critic.parameters())

        for e in range(params.epochs):
            ep_ret = 0
            ep_len = 0

            pbar = tqdm.tqdm(
                range(params.steps_per_epoch),
                desc=f"Epoch {e+1:>4}",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            )

            cumulative_metrics = {
                "q_loss": 0,
                "pi_loss": 0,
            }

            for i in pbar:
                if step < params.start_steps:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action, _ = actor(
                            FloatTensor(state).to(self.device)
                        )
                        action = action.cpu().detach().numpy()

                next_state, reward, done, _ = self.env.step(action)
                self.env.render()
                ep_len += 1
                ep_ret += reward

                self.buffer.store(state, action, reward, next_state, done)
                state = next_state

                timeout = ep_len == params.max_ep_len
                terminal = done or timeout
                epoch_ended = i % params.steps_per_epoch == params.steps_per_epoch - 1

                if terminal or epoch_ended:
                    state = self.env.reset()
                    ep_ret = 0
                    ep_len = 0

                if step > params.update_after and step % params.update_every == 0:
                    for _ in range(params.update_every):
                        samples = self.buffer.sample(params.batch_size)

                        q_opt.zero_grad()
                        loss_q = eval_q_loss(
                                actor,
                                double_critic,
                                target_critic,
                                samples.states,
                                samples.actions,
                                samples.rewards,
                                samples.next_states,
                                samples.done,
                                params.alpha,
                                params.gamma,
                        )
                        loss_q.backward()
                        q_opt.step()


                        pi_opt.zero_grad()
                        loss_pi = eval_pi_loss(
                            actor,
                            double_critic,
                            samples.states,
                            samples.next_states,
                            params.alpha
                        )

                        loss_pi.backward()
                        pi_opt.step()


                        # q_state_targ = update_targets(double_critic, target_critic, params.polyak)

                        cumulative_metrics["pi_loss"] += loss_pi.cpu().detach().numpy()
                        cumulative_metrics["q_loss"] += loss_q.cpu().detach().numpy()

                        metrics = {
                            "Episode Length": ep_len,
                            "Cumulative Reward": ep_ret,
                            "Q Loss": f"{cumulative_metrics['q_loss'] / (i + 1):.4g}",
                            "PI Loss": f"{cumulative_metrics['pi_loss'] / (i + 1):.4g}",
                        }
                        pbar.set_postfix(metrics)

                step += 1

            state = self.env.reset()
            ep_ret = 0
            ep_len = 0

def main():
    env = gym.make("Humanoid-v2")

    training_params = TrainingParameters(
        epochs=10,
        steps_per_epoch=4000,
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        learning_rate=1e-3,
        alpha=0.2,
        gamma=0.99,
        polyak=0.99
    )

    sac = SAC(env, training_params)
    sac.train()


if __name__ == "__main__":
    main()
