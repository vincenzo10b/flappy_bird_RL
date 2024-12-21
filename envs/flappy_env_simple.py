from typing import Dict, Tuple, Optional, Union

import gym
import numpy as np
import pygame

from flappy_bird_gym.envs.game.flappy_ai import Flappy


class FlappyBirdEnv(gym.Env):


    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 255, [*screen_size, 3])
        self._game = None

    def _get_observation(self):
        return pygame.surfarray.array3d(self._game.get_screen())

    def reset(self):
        """ Resets the environment (starts a new game).
        """
        if self._game is None:
            self._game = Flappy()
        self._game.start()
        return self._get_observation(), {"score": 0}

    def step(self, action):
        alive = self._game.update_state(action)
        obs = self._get_observation()

        reward = 1

        done = not alive
        info = {"score": self._game.score}

        return obs, reward, done, info
        
    def get_rgb_array(self):
        return pygame.surfarray.array3d(self._game.get_screen())
            

    def close(self):
        """ Closes the environment. """
        if self._game is not None:
            pygame.display.quit()
            self._game = None

        super().close()
