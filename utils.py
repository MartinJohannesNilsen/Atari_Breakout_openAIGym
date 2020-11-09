"""This class includes utils, as of now only the new enviroment with FrameStacking and rezising, with transition to grey-scale"""
import cv2
import numpy as np
import gym
from random import randint, choice
import os


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
    def __init__(self):
        self.n = 3

    def sample(self):
        return choice([0, 2, 3])  # Removes the fire option (1)


class NoFireInActionSpaceEnv(FrameStackingAndResizingEnv):
    def __init__(self, env, w, h, num_stack=4):
        super(NoFireInActionSpaceEnv, self).__init__(env, w, h, num_stack)

    def reset(self):
        """
        Resets the enviroment

        Output:\n
        - last observation, a copy of the buffer of frames
        """
        image = self.env.reset()
        image, _, done, _ = self.env.step(1)
        if done:
            self.reset()
        self.frame = image.copy()
        image = self._preprocess_frame(image)
        self.buffer = np.stack([image]*self.n, 0)
        return self.buffer.copy()

    @property
    def action_space(self):
        return no_fire_action_space()


if __name__ == "__main__":
    env = gym.make("BreakoutDeterministic-v4")
    env = FrameStackingAndResizingEnv(env, 480, 640)
    print_path = os.path.join(os.path.dirname(__file__), f"Frame/")

    image = env.reset()
    idx = 0
    ims = []
    for i in range(image.shape[-1]):
        ims.append(image[:, :, i])
    if not cv2.imwrite(print_path+f"/{idx}.png", np.hstack(ims)):
        raise Exception("Could not write image")

    env.step(1)

    for _ in range(10):
        idx += 1
        image, _, _, _ = env.step(randint(0, 3))

        ims = []
        for i in range(image.shape[-1]):
            ims.append(image[:, :, i])

        if not cv2.imwrite(print_path+f"/{idx}.png", np.hstack(ims)):
            raise Exception("Could not write image")
