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
from random import sample
from tqdm import tqdm
from models import ConvModel
from constants import memory_size, min_rb_size, sample_size, lr, eps_decay, discount_factor, env_steps_before_train, epochs_before_tgt_model_update, epochs_before_test, episode_max_steps, eps_min, exploration_method, env_type, optimizer_function


@dataclass
class GameInformation:
    "Dataclass including the elements state, action, reward, next_state and done"
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    """
    Buffer for achieving experience replay, storing earlier experiences.

    Input:\n
    - Buffer size, max number of elements in buffer
    """

    def __init__(self, buffer_size=memory_size):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.i = 0

    def insert(self, gameInfo):
        self.buffer[self.i % self.buffer_size] = gameInfo
        self.i += 1

    def sample(self, num_samples):
        assert num_samples < min(self.i, self.buffer_size)
        if self.i < self.buffer_size:
            return sample(self.buffer[:self.i], num_samples)
        return sample(self.buffer, num_samples)


def update_target_model(m, target):
    target.load_state_dict(m.state_dict())


def train_step(model, state_transitions, target, num_actions, gamma=discount_factor):
    """
    Function for running a training step

    Input:\n
    - model, the model used for calculating the Q-table
    - state_transitions, the states in a batch from the replay buffer
    - target, the target model
    - num_actions, the number of actions in the actions_space from enviroment
    - gamma, the discount factor (default discount_factor from hyperparams)

    Output:\n
    - loss, number representing the Huber loss of the training step
    """
    # Have to stack the torches because of sample size in state_transitions
    current_states = torch.stack(([torch.Tensor(s.state)for s in state_transitions]))
    rewards = torch.stack(([torch.Tensor([s.reward])for s in state_transitions]))
    # The action mask indicates whether an action is valid or invalid for each state
    action_mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]))
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions]))
    actions = [s.action for s in state_transitions]

    # Need to compute the qvals of the next state
    with torch.no_grad():
        qvals_next = target(next_states).max(-1)[0]  # (N, num_actions)

    model.opt.zero_grad()
    q_vals = model(current_states)  # (N, num_actions)
    # Multiplies q_vals with one_hot_actions for efficiency, gived me the actions for the state
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)

    loss_fn = nn.SmoothL1Loss()  # Huber loss
    loss = loss_fn(torch.sum(q_vals * one_hot_actions, -1), rewards.squeeze() + action_mask[:, 0] * qvals_next * gamma)
    # loss = (rewards + qvals_next - torch.sum(q_vals*one_hot_actions, -1).mean())  # -1 for direction # First implementation of MSE lossfunction
    loss.backward()
    model.opt.step()

    return loss


def run_test_episode(model, env, max_steps=episode_max_steps):
    """
    Run one episode of the game 

    Input:\n
    - model, the model network
    - env, the enviroment
    - max_steps, the maximum steps possible to do in the enviroment

    Output:\n
    - reward, the score of the test_run epsiode
    - movie, stack of frames making a movie
    """
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

    movie = np.stack(frames, 0)

    return reward, movie


def main(name=None, chkpt=None, test_run=False, local_run=False):
    "Sync to wandb cloud as standard, but sync locally if local_run and not at all if test_run"
    if not test_run:
        if local_run:
            os.environ["WANDB_MODE"] = "dryrun"
        if name == None:
            name = input("Name the run: ")
        wandb.init(project="atari-breakout", name=name, config={
            'memory_size': memory_size,
            'min_rb_size': min_rb_size,
            'sample_size': sample_size,
            'lr': lr,
            'eps_min': eps_min,
            'eps_decay': eps_decay,
            'discount_factor': discount_factor,
            'env_steps_before_train': env_steps_before_train,
            'epochs_before_tgt_model_update': epochs_before_tgt_model_update,
            'epochs_before_test': epochs_before_test,
            'episode_max_steps': episode_max_steps,
            'optimizer_function': optimizer_function.__name__,
            'exploration_method': exploration_method.__name__,
            'env_type': env_type.__name__
        })

    "Create enviroments and reset"
    env = env_type(gym.make("BreakoutDeterministic-v4"), 84, 84, 4)
    test_env = env_type(gym.make("BreakoutDeterministic-v4"), 84, 84, 4)
    last_observation = env.reset()

    "Set the model and targetmodel"
    m = ConvModel(env.observation_space.shape, env.action_space.n, lr=lr)
    if chkpt is not None:
        m.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f"Models/{chkpt}")))
    target = ConvModel(env.observation_space.shape, env.action_space.n)
    update_target_model(m, target)

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
            if test_run:
                env.render()
                time.sleep(0.05)
            tq.update(1)

            "Updating epsilon"
            eps = eps_decay ** (step_num)
            if test_run:
                eps = 0
            elif eps < eps_min:
                eps = eps_min

            "Exploration vs exploitation, Boltzmann with eps_decay vs. Epsilon Greedy (defined in constants.py)"
            action = exploration_method(model=m, env=env, last_observation=last_observation, eps=eps)

            "Perform step and insert observation to replaybuffer"
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            rb.insert(GameInformation(last_observation, action, reward, observation, done))
            last_observation = observation

            "Reset and append total_reward to episode_rewards if done"
            if done:
                episode_rewards.append(total_reward)
                if test_run:
                    print(total_reward)
                total_reward = 0
                observation = env.reset()

            "Train if ran enough steps since last training"
            steps_since_train += 1
            step_num += 1
            if ((not test_run) and rb.i > min_rb_size and steps_since_train > env_steps_before_train):
                loss = train_step(m, rb.sample(sample_size), target, env.action_space.n)
                if not local_run:
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

                "Run test_run episode"
                if epochs_since_test > epochs_before_test:
                    rew, frames = run_test_episode(m, test_env)
                    if not local_run:
                        wandb.log({'test_reward': rew, 'test_video': wandb.Video(frames.transpose(0, 3, 1, 2), str(rew), fps=25, format='mp4')})
                    epochs_since_test = 0

                "Update target model"
                if epochs_since_tgt > epochs_before_tgt_model_update:
                    print("updating target model")
                    update_target_model(m, target)
                    epochs_since_tgt = 0
                    torch.save(target.state_dict(), os.path.join(os.path.dirname(__file__), f"Models/{step_num}.pth"))

                steps_since_train = 0

    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    argh.dispatch_command(main)
