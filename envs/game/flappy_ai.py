import asyncio
import sys

import pygame
from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    PlayerActions,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

class Flappy:
    def __init__(self, screen_width, screen_height):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(screen_width, screen_height)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )

    def start(self):
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.welcome_message = WelcomeMessage(self.config)
        self.game_over_message = GameOver(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)
        
    
    def update_state(self, action):
        if action == PlayerActions.FLAP:
            self.player.flap()
        
        if self.player.collided(self.pipes, self.floor):
            return False

        for i, pipe in enumerate(self.pipes.upper):
            if self.player.crossed(pipe):
                self.score.add()

        for event in pygame.event.get():
            self.check_quit_event(event)

        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.score.tick()
        self.player.tick()

        pygame.display.update()
        #await asyncio.sleep(0)
        self.config.tick()
        return True

    
        
    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()
    
    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)
    
    def get_screen(self):
        return self.config.screen
