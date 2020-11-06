import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argh
import os
import wandb
from dataclasses import dataclass
from typing import Any
from random import sample, random
from tqdm import tqdm
from utils import FrameStackingAndResizingEnv
from models import ConvModel
from hyperparams import do_boltzman_exploration, memory_size, min_rb_size, sample_size, lr, eps_decay, discount_factor, env_steps_before_train, tgt_model_update, epochs_before_test, eps_max, eps_min


@dataclass
class Sarsd:  # State, action, reward, next_state, done
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    def __init__(self, buffer_size=1_000_000):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.i = 0

    def insert(self, sars):
        self.buffer[self.i % self.buffer_size] = sars
        self.i += 1

    def sample(self, num_samples):
        assert num_samples < min(self.i, self.buffer_size)
        if self.i < self.buffer_size:
            return sample(self.buffer[:self.i], num_samples)
        return sample(self.buffer, num_samples)


def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())


def train_step(model, state_transitions, tgt, num_actions, gamma=discount_factor):
    cur_states = torch.stack(([torch.Tensor(s.state)for s in state_transitions]))
    rewards = torch.stack(([torch.Tensor([s.reward])for s in state_transitions]))
    # The action mask indicates whether an action is valid or invalid for each state
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]))
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions]))
    actions = [s.action for s in state_transitions]

    # Need to compute the qvals of the next state
    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]  # (N, num_actions)

    model.opt.zero_grad()
    qvals = model(cur_states)  # (N, num_actions)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)  # Multiplies qvals with one_hot_actions for efficiency, gived me the actions for the state

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals * one_hot_actions, -1), rewards.squeeze() + mask[:, 0] * qvals_next * gamma)
    # loss = (rewards + qvals_next - torch.sum(qvals*one_hot_actions, -1).mean()) # -1 for direction
    loss.backward()
    model.opt.step()

    return loss


def run_test_episode(model, env, max_steps=1000):  # -> reward, movie?
    frames = []
    obs = env.reset()
    frames.append(env.frame)

    i = 0
    done = False
    reward = 0
    while not done and i < max_steps:
        action = model(torch.Tensor(obs).unsqueeze(0)).max(-1)[-1].item()
        obs, r, done, _ = env.step(action)
        reward += r
        frames.append(env.frame)
        i += 1

    return reward, np.stack(frames, 0)


def main(name=input("Name the run: "), test=False, chkpt=None):
    if not test:
        wandb.init(project="atari-breakout", name=name)

    env = gym.make("BreakoutDeterministic-v4")
    env = FrameStackingAndResizingEnv(env, 84, 84, 4)

    test_env = gym.make("BreakoutDeterministic-v4")
    test_env = FrameStackingAndResizingEnv(test_env, 84, 84, 4)

    last_observation = env.reset()

    "Set the model and targetmodel"
    m = ConvModel(env.observation_space.shape, env.action_space.n, lr=lr)
    if chkpt is not None:
        m.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f"Models/{chkpt}")))
    tgt = ConvModel(env.observation_space.shape, env.action_space.n)
    update_tgt_model(m, tgt)

    "Create replaybuffer and other variables"
    rb = ReplayBuffer(memory_size)
    steps_since_train = 0
    epochs_since_tgt = 0
    epochs_since_test = 0
    step_num = -1 * min_rb_size  # Want to run the iteration for min_rb_size before starting to actually learn
    episode_rewards = []
    total_reward = 0

    tq = tqdm()
    try:
        while True:
            if test:
                env.render("rgb_array")
                time.sleep(0.05)
            tq.update(1)

            eps = eps_decay ** (step_num)
            if test:
                eps = 0
            elif eps <= eps_min:
                eps = 0.1
                

            "Exploration vs exploitation"
            if do_boltzman_exploration:
                logits = m(torch.Tensor(last_observation).unsqueeze(0))[0]
                action = torch.distributions.Categorical(
                    logits=logits).sample().item()
            else:
                if random() < eps:
                    action = (
                        env.action_space.sample()
                    )
                else:
                    action = m(torch.Tensor(last_observation).unsqueeze(
                        0)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            total_reward += reward

            "Insert to replaybuffer the new observation"
            rb.insert(Sarsd(last_observation, action,
                            reward, observation, done))

            last_observation = observation

            if done:
                episode_rewards.append(total_reward) 
                if test:
                    print(total_reward)
                total_reward = 0
                observation = env.reset()

            steps_since_train += 1
            step_num += 1

            if ((not test) and rb.i > min_rb_size and steps_since_train > env_steps_before_train):
                loss = train_step(m, rb.sample(sample_size), tgt, env.action_space.n)
                wandb.log(
                    {
                        "loss": loss.detach().item(),
                        "eps": eps,
                        "avg_reward": np.mean(episode_rewards),
                    },
                    step=step_num,
                )
                episode_rewards = []
                epochs_since_tgt += 1
                epochs_since_test += 1

                "Run test episode"
                if epochs_since_test > epochs_before_test:
                    rew, frames = run_test_episode(m, test_env)
                    # T, H, W, C
                    wandb.log({'test_reward': rew, 'test_video': wandb.Video(
                        frames.transpose(0, 3, 1, 2), str(rew), fps=25, format='mp4')})
                    epochs_since_test = 0

                "Update target model"
                if epochs_since_tgt > tgt_model_update:
                    print("updating target model")
                    update_tgt_model(m, tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), os.path.join(os.path.dirname(__file__), f"Models/{step_num}.pth"))

                steps_since_train = 0

    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    argh.dispatch_command(main)
