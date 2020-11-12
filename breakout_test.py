"Play the game Atari Brakeout using the keyboard"
import gym
from gym.utils.play import play

play(gym.make('BreakoutDeterministic-v4'), zoom=3)
