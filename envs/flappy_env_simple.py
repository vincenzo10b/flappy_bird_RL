from typing import Dict, Tuple, Optional, Union

import gymnasium as gym
import numpy as np
import pygame

from envs.game.flappy_ai import Flappy
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24
PIPE_WIDTH = 52
PIPE_HEIGHT = 320


class FlappyBirdEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -np.inf, 0, 0], dtype=np.float32),  # Minimum values for each observation
            high=np.array([SCREEN_HEIGHT, np.inf, SCREEN_WIDTH, SCREEN_HEIGHT], dtype=np.float32),  # Maximum values
            dtype=np.float32
        )
        self._game = None

    def _get_observation(self):
        up_pipe = low_pipe = None
        h_dist = 0
        for up_pipe, low_pipe in zip(self._game.pipes.upper,
                                     self._game.pipes.lower):
            h_dist = (low_pipe.x + PIPE_WIDTH / 2
                      - (self._game.player.x - PLAYER_WIDTH / 2))
            h_dist += 3  # extra distance to compensate for the buggy hit-box
            if h_dist >= 0:
                break

        upper_pipe_y = up_pipe.y + PIPE_HEIGHT
        lower_pipe_y = low_pipe.y
        player_y = self._game.player.y
        vel_y = self._game.player.vel_y

        v_dist = (upper_pipe_y + lower_pipe_y) / 2 - (player_y
                                                      + PLAYER_HEIGHT/2)

        h_dist /= SCREEN_WIDTH
        v_dist /= SCREEN_HEIGHT
        vel_y /= SCREEN_HEIGHT

        return np.array([
            h_dist,
            v_dist,
        ])


    def reset(self):
        """ Resets the environment (starts a new game).
        """
        if self._game is None:
            self._game = Flappy(SCREEN_WIDTH, SCREEN_HEIGHT)
        self._game.start()
        return self._get_observation(), {"score": 0}

    def step(self, action):
        alive = self._game.update_state(action)
        obs = self._get_observation()
        done = not alive
        h_dist, v_dist = obs
        reward = 0
        if done:
            reward = -100
        else:
            reward = 0
        # if v_dist < 0.03 and v_dist > -0.03 and h_dist < 0.3 and h_dist >= 0:
            # reward = 
        # elif v_dist > 0.4 or v_dist < -0.4:
            # reward = 0

        
        info = {"score": self._game.score.score}

        return obs, reward, done, info
    
    #TODO
    def render():
        return None
        
        
    def get_rgb_array(self):
        return pygame.surfarray.array3d(self._game.get_screen())
            

    def close(self):
        """ Closes the environment. """
        if self._game is not None:
            pygame.display.quit()
            self._game = None

        super().close()
