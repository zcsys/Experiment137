import pygame
import time
import json
import torch
from base_vars import *

class Button:
    def __init__(self, screen, x, y, width, height, label, font,
                 default_color = (200, 200, 200), hover_color = (170, 170, 170),
                 clicked_color = (150, 150, 150), text_color = (0, 0, 0)):
        self.screen = screen
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.font = font
        self.default_color = default_color
        self.hover_color = hover_color
        self.clicked_color = clicked_color
        self.text_color = text_color
        self.current_color = default_color
        self.clicked = False

    def draw(self):
        pygame.draw.rect(self.screen, self.current_color, self.rect)
        text_surf = self.font.render(self.label, True, self.text_color)
        text_rect = text_surf.get_rect(center = self.rect.center)
        self.screen.blit(text_surf, text_rect)

    def is_hovered(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

    def handle_event(self, event):
        mouse_pos = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered(mouse_pos):
            self.clicked = True
            self.current_color = self.clicked_color
        elif event.type == pygame.MOUSEBUTTONUP and self.clicked:
            self.clicked = False
            self.current_color = (self.hover_color if self.is_hovered(mouse_pos)
                                  else self.default_color)
            return True
        elif event.type == pygame.MOUSEMOTION:
            if self.is_hovered(mouse_pos):
                self.current_color = self.hover_color
            else:
                self.current_color = self.default_color
        return False

class UIManager:
    def __init__(self, screen, menu_width):
        self.screen = screen
        self.menu_width = menu_width
        self.font = pygame.font.Font(None, 24)

        self.save_button = Button(self.screen,
                                  screen.get_width() - menu_width + 10, 10, 160,
                                  40, "Save", self.font)
        self.play_pause_button = Button(self.screen,
                                        screen.get_width() - menu_width + 10,
                                        60, 160, 40, "Pause", self.font)

    def handle_event(self, event, simulation):
        if self.save_button.handle_event(event):
            simulation.save_simulation()
        if self.play_pause_button.handle_event(event):
            simulation.toggle_pause()
            self.play_pause_button.label = ("Play" if simulation.paused
                                            else "Pause")

    def draw(self, state, num_things):
        # Draw the right menu section (white background)
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.screen.get_width() - self.menu_width, 0,
                         self.menu_width, self.screen.get_height()))

        # Draw buttons
        self.save_button.draw()
        self.play_pause_button.draw()

        # Display simulation state (Epochs, Periods, Steps)
        start_y = self.screen.get_height() // 2

        epoch_text = self.font.render(f"Epoch: {state.get('epochs', 0)}", True,
                                      (0, 0, 0))
        period_text = self.font.render(f"Period: {state.get('periods', 0)} " +
                                       f"({state.get('crr_period_dur', 0)}\")",
                                       True, (0, 0, 0))
        steps_text = self.font.render(f"Steps: {state.get('steps', 0)}", True,
                                      (0, 0, 0))
        population_text = self.font.render(f"Pop.: {num_things}", True,
                                           (0, 0, 0))

        self.screen.blit(epoch_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y))
        self.screen.blit(period_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 30))
        self.screen.blit(steps_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 60))
        self.screen.blit(population_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 90))

class Simulation:
    def __init__(self, things_object, load_file = None):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Experiment 137.03: LUCA")
        self.ui_manager = UIManager(self.screen, MENU_WIDTH)
        self.paused = False
        self.things = things_object
        self.steps, self.periods, self.epochs = 0, 0, 0
        self.period_start_time = time.time()
        self.crr_period_dur = 0
        if load_file:
            self.load_simulation(load_file)

    def update_state(self):
        self.steps += 1
        if self.steps == 2400:
            self.periods += 1
            self.steps = 0
            current_time = time.time()
            self.crr_period_dur = int(current_time - self.period_start_time)
            self.period_start_time = current_time
        if self.periods == 80:
            self.epochs += 1
            self.periods = 0
            self.save_simulation()

    def get_state(self):
        return {
            'steps': self.steps,
            'periods': self.periods,
            'epochs': self.epochs,
            'period_start_time': self.period_start_time,
            'crr_period_dur': self.crr_period_dur
        }

    def load_state(self, state):
        self.steps = state.get('steps', 0)
        self.periods = state.get('periods', 0)
        self.epochs = state.get('epochs', 0)
        self.period_start_time = time.time()
        self.crr_period_dur = state.get('crr_period_dur', 0)

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.ui_manager.handle_event(event, self)

            if not self.paused:
                self.things.update_positions()
                self.update_state()

            self.screen.fill(BLACK)
            self.things.draw(self.screen)
            self.ui_manager.draw(self.get_state(), self.things.num_things)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def save_simulation(self):
        filename = f"simulation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        combined_state = {
            "simulation_state": self.get_state(),
            "things_state": self.things.get_state()
        }
        with open(filename, 'w') as f:
            json.dump(combined_state, f)
        print(f"Simulation saved to {filename}")

    def load_simulation(self, filename):
        with open(filename, 'r') as f:
            saved_data = json.load(f)
            self.load_state(saved_data["simulation_state"])
            self.things.load_state(saved_data["things_state"])

    def toggle_pause(self):
        self.paused = not self.paused
