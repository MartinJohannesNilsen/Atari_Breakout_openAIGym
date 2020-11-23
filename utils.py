"""
This file includes the following utilities:
- New enviroment with framestacking and rezising, transitioned to grey-scale
- New enviroment which inherit the framestack and resize, but also removes the fire action from action_space
- New action_space class without the fire action
- Test framestack and print out frames as enviroment sees them
- Exploration vs exploitation methods, both epsilon greedy and boltzmann with eps_decay
"""
import cv2
import numpy as np
import gym
from random import choice
import os
from collections import deque
from random import random
import torch


class FrameStackingAndResizingEnv:
    """
    My own version of the atari enviroment, with framestacking, resizing and gray-scaling

    Parameters:\n
    - env, enviroment made with gym.make("env_name")
    - w, width in pixels
    - h, height in pixels
    - num_stack, number of stacked frames (default 4)
    """

    def __init__(self, env, w, h, num_stack=4):
        self.env = env
        self.n = num_stack
        self.w = w
        self.h = h
        self.buffer = np.zeros((num_stack, h, w), 'uint8')
        self.frame = None

    def _preprocess_frame(self, frame):
        """
        Preprocess each frame, using cv2. This includes rezising and change the color space to grey-scale.

        Input:\n
        - frame, a single frame of the game

        Output:\n
        - image, new preprocessed frame
        """
        image = cv2.resize(frame, (self.w, self.h))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def step(self, action):
        """
        Perform step, with new image as state

        Input:\n
        """
        image, reward, done, info = self.env.step(action)
        self.frame = image.copy()
        image = self._preprocess_frame(image)
        self.buffer[1:self.n, :, :] = self.buffer[0:self.n-1, :, :]
        self.buffer[0, :, :] = image
        return self.buffer.copy(), reward, done, info

    def render(self, mode="human"):
        """
        Render the game

        Input:\n
        - mode, the mode to render in (default "human" from openAI's own enviroments for Atari)

        Output:\n
        - self.frame or a rendered view from gyms own enviroment, depending on "rgb_array" as mode or not
        """
        if mode == 'rgb_array':
            return self.frame
        return self.env.render(mode)

    def reset(self):
        """
        Resets the enviroment

        Output:\n
        - last observation, a copy of the buffer of frames
        """
        image = self.env.reset()
        self.frame = image.copy()
        image = self._preprocess_frame(image)
        self.buffer = np.stack([image]*self.n, 0)
        return self.buffer.copy()

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        # Burde kanskje tatt i bruk Gyms egne: gym.spaces.Box()?
        return np.zeros((self.n, self.h, self.w))

    @property
    def action_space(self):
        return self.env.action_space


class no_fire_action_space:
    """
    Action space for the enviroment without the action "fire". 
    New mapping of numbers to steps is ['noop', 'right', 'left']
    """

    def __init__(self):
        self.n = 3

    def sample(self):
        # return choice([0, 2, 3])  # Removes the fire option (1)
        return choice([0, 1, 2])  # Removes the fire option, now 0=Noop, 1=right and 2=left


class NoFireInActionSpaceEnv(FrameStackingAndResizingEnv):
    """
    Enviroment in which removes the responsibility of starting the game from the agent, and starts the game at reset()
    """

    def __init__(self, env, w, h, num_stack=4):
        super(NoFireInActionSpaceEnv, self).__init__(env, w, h, num_stack)
        self.lives = 5

    def reset(self):
        """
        Resets the enviroment

        Output:\n
        - last observation, a copy of the buffer of frames
        """
        image = self.env.reset()
        image, _, done, _ = self.env.step(1)
        self.lives = 5
        self.frame = image.copy()
        image = self._preprocess_frame(image)
        self.buffer = np.stack([image]*self.n, 0)
        return self.buffer.copy()

    @property
    def action_space(self):
        return no_fire_action_space()

    def step(self, action):
        """
        Run the steps as given by OpenAI Gym, but map the actions to the new action_space ['noop', 'right', 'left'], instead of the original ['noop', 'fire', 'right', 'left'] 
        """
        assert action < 3, "Action should be in the interval [0,2] with reduced action_space"
        if action == 0:
            observation, reward, done, info = super(NoFireInActionSpaceEnv, self).step(0)  # Noop
        elif action == 1:
            observation, reward, done, info = super(NoFireInActionSpaceEnv, self).step(2)  # Right
        elif action == 2:
            observation, reward, done, info = super(NoFireInActionSpaceEnv, self).step(3)  # Left
        else:
            observation, reward, done, info = None

        if info['ale.lives'] < self.lives:
            self.env.step(1)
            self.lives -= 1

        return observation, reward, done, info


def test_FrameStackingAndresizingEnv(number_of_frames=20):
    env = gym.make("BreakoutDeterministic-v4")
    env = FrameStackingAndResizingEnv(env, 480, 640)
    print_path = os.path.join(os.path.dirname(__file__), f"FramestackingAndPreprocessing/")

    image = env.reset()
    i = 0
    ims = deque()
    for i in range(image.shape[0]):
        # ims.appendleft(image[i, :, :])
        ims.append(image[i, :, :])
    if not cv2.imwrite(print_path+f"/{i}.png", np.hstack(ims)):
        raise Exception("Could not write image")

    env.step(1)

    for _ in range(number_of_frames):
        i += 1
        # image, _, _, _ = env.step(choice([0, 2, 3]))
        image, _, _, _ = env.step(3)  # Move to the left

        ims = deque()
        for i in range(image.shape[0]):
            # ims.appendleft(image[i, :, :])
            ims.append(image[i, :, :])
        if not cv2.imwrite(print_path+f"/{i}.png", np.hstack(ims)):
            raise Exception("Could not write image")


def Boltzmann(model, env, last_observation, eps=None):
    """
    Boltzmann takes the possibility of selecting one of the possible actions, and samples with the possibilites in mind. Also known as softmax exploration. 
    If using the epsilon value, we take the eps_decay concept from epsilon greedy, but samples with the possibilites from Boltzmann. 

    Input:\n
    - model, the machine learning model
    - env, the enviroment
    - last_observation, the last observation recieved from env.step()
    - eps, the epsilon value (Default None for allways using Boltzmann distrbution, else a value for own version with epsilon decay)

    Output:\n
    - action, the action in which to take next
    """
    if eps != None:
        # Boltzmann exploration with epsilon_decay for exploration vs exploitation. Need to exploit for utilizing the experience replay.
        if random() < eps:
            # Explore with Boltzmann
            logits = model(torch.Tensor(last_observation).unsqueeze(0))[0]  # One tensor, in which we sample from using Categorical
            action = torch.distributions.Categorical(logits=logits).sample().item()
        else:
            # Exploit
            action = model(torch.Tensor(last_observation).unsqueeze(0)).max(-1)[-1].item()
    else:
        # Use regular sampling from Boltzmann
        logits = model(torch.Tensor(last_observation).unsqueeze(0))[0]
        action = torch.distributions.Categorical(logits=logits).sample().item()
    return action


def EpsilonGreedy(model, env, last_observation, eps):
    """
    Epsilon Greedy with decreasing epsilon value. Implemented to decrease exponentially to the eps_min (eps_decay^step = eps). 
    With eps_decay = 0.999999, it takes around 2_300_000 steps to get to eps_min = 0.1

    Input:\n
    - model, the machine learning model
    - env, the enviroment
    - last_observation, the last observation recieved from env.step()
    - eps, the epsilon value (Default None for allways using Boltzmann distrbution, else a value for own version with epsilon decay)

    Output:\n
    - action, the action in which to take next
    """
    if random() < eps:
        # Explore randomly
        action = env.action_space.sample()
    else:
        # Exploit
        action = model(torch.Tensor(last_observation).unsqueeze(0)).max(-1)[-1].item()
    return action


if __name__ == "__main__":
    test_FrameStackingAndresizingEnv(number_of_frames=20)
