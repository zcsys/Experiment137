import pygame
import time
import json
import torch
import math
from base_vars import *
from rules import Rules

def draw_dashed_circle(surface, color, center, radius, dash_length = 5,
                       gap_length = 5):
    angle = 0
    total_circumference = 2 * math.pi * radius
    dash_angle = (dash_length / total_circumference) * 360
    gap_angle = (gap_length / total_circumference) * 360
    while angle < 360:
        start_x = center[0] + radius * math.cos(math.radians(angle))
        start_y = center[1] + radius * math.sin(math.radians(angle))
        end_angle = angle + dash_angle
        end_x = center[0] + radius * math.cos(math.radians(end_angle))
        end_y = center[1] + radius * math.sin(math.radians(end_angle))
        pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 1)
        angle += dash_angle + gap_angle

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
    def __init__(self, screen, menu_width, paused):
        self.screen = screen
        self.menu_width = menu_width
        self.font = pygame.font.Font(None, 24)

        self.save_button = Button(self.screen,
                                  screen.get_width() - menu_width + 10, 10, 160,
                                  40, "Save", self.font)
        self.play_pause_button = Button(self.screen,
                                        screen.get_width() - menu_width + 10,
                                        60, 160, 40, "Play" if paused else
                                        "Pause", self.font)
        self.sight_toggle_button = Button(self.screen, screen.get_width() -
                                          menu_width + 10, 110, 160, 40,
                                          "Toggle Sight", self.font)
        self.force_toggle_button = Button(self.screen, screen.get_width() -
                                          menu_width + 10, 160, 160, 40,
                                          "Toggle Forces", self.font)
        self.info_toggle_button = Button(self.screen, screen.get_width() -
                                          menu_width + 10, 210, 160, 40,
                                          "Toggle Info", self.font)
        self.network_toggle_button = Button(self.screen, screen.get_width() -
                                          menu_width + 10, 260, 160, 40,
                                          "Toggle Network", self.font)

        self.show_info = False
        self.show_sight = False
        self.show_forces = False
        self.show_network = False

    def handle_event(self, event, simulation):
        if self.save_button.handle_event(event):
            simulation.save_simulation()
        if self.play_pause_button.handle_event(event):
            simulation.toggle_pause()
            self.play_pause_button.label = ("Play" if simulation.paused
                                            else "Pause")
        if self.sight_toggle_button.handle_event(event):
            self.show_sight = not self.show_sight
        if self.force_toggle_button.handle_event(event):
            self.show_forces = not self.show_forces
        if self.info_toggle_button.handle_event(event):
            self.show_info = not self.show_info
        if self.network_toggle_button.handle_event(event):
            self.show_network = not self.show_network

    def draw(self, state, N, Pop, E):
        # Draw the right menu section (white background)
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.screen.get_width() - self.menu_width, 0,
                         self.menu_width, self.screen.get_height()))

        # Draw buttons
        self.save_button.draw()
        self.play_pause_button.draw()
        self.sight_toggle_button.draw()
        self.force_toggle_button.draw()
        self.info_toggle_button.draw()
        self.network_toggle_button.draw()

        # Display simulation state (Epochs, Periods, Steps)
        start_y = self.screen.get_height() // 2

        epoch_text = self.font.render(f"Epoch: {state.get('epochs', 0)}", True,
                                      (0, 0, 0))
        period_text = self.font.render(f"Period: {state.get('periods', 0)} " +
                                       f"(\'{state.get('crr_period_dur', 0)})",
                                       True, (0, 0, 0))
        steps_text = self.font.render(f"Steps: {state.get('steps', 0)}", True,
                                      (0, 0, 0))
        N_text = self.font.render(f"N: {N}", True, (0, 0, 0))
        Pop_text = self.font.render(f"Pop.: {Pop}", True, (0, 0, 0))
        E_text = self.font.render(f"E: {int(E)}k", True, (0, 0, 0))

        self.screen.blit(epoch_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y))
        self.screen.blit(period_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 30))
        self.screen.blit(steps_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 60))
        self.screen.blit(N_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 90))
        self.screen.blit(Pop_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 120))
        self.screen.blit(E_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 150))

class Simulation:
    def __init__(self, things_object, load_file = None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Experiment 137.03: FUCA")
        self.things = things_object
        self.period_start_time = time.time()
        self.transparent_surface = pygame.Surface(
            (SIMUL_WIDTH, SIMUL_HEIGHT), pygame.SRCALPHA
        ).convert_alpha()

        if load_file:
            with open(load_file, 'r') as f:
                saved_data = json.load(f)
                self.load_state(saved_data["simulation_state"])
            self.paused = True
            self.ui_manager = UIManager(self.screen, MENU_WIDTH, self.paused)
            print(f"Simulation restored from {load_file}")
            return

        self.paused = False
        self.ui_manager = UIManager(self.screen, MENU_WIDTH, self.paused)
        self.steps, self.periods, self.epochs = 0, 0, 0
        self.crr_period_dur = 0

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
        self.crr_period_dur = state.get('crr_period_dur', 0)

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            # print("\n\n==== BEGIN STEP ====\n")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.ui_manager.handle_event(event, self)

            if not self.paused:
                self.things.final_action()
                self.update_state()

            self.screen.fill(colors["0"])
            self.things.draw(self.screen, self.ui_manager.show_info,
                             self.ui_manager.show_sight,
                             self.ui_manager.show_forces,
                             self.ui_manager.show_network)

            if not self.paused:
                Rules(self, [0, 1])

            # The Arbeitor of Truth
            """
            self.transparent_surface.fill((0, 0, 0, 0))
            pygame.draw.circle(
                self.transparent_surface,
                (254, 254, 0, 50),
                (SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2),
                610
            )
            pygame.draw.circle(
                self.transparent_surface,
                (254, 254, 0, 75),
                (SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2),
                377
            )
            pygame.draw.circle(
                self.transparent_surface,
                (254, 254, 0, 100),
                (SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2),
                233
            )
            pygame.draw.circle(
                self.transparent_surface,
                (254, 254, 0, 125),
                (SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2),
                144
            )
            pygame.draw.circle(
                self.transparent_surface,
                (254, 254, 0, 150),
                (SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2),
                89
            )
            pygame.draw.circle(
                self.transparent_surface,
                (254, 254, 0, 175),
                (SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2),
                55
            )
            self.screen.blit(
                self.transparent_surface,
                (1570 - self.steps - (self.periods % 2) * 2400, 0)
            )
            """

            # Draw the right pane
            self.ui_manager.draw(
                self.get_state(),
                self.things.N,
                self.things.Pop,
                self.things.E
            )

            # Put it all on display and limit FPS
            pygame.display.flip()
            # clock.tick(24)

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

    def toggle_pause(self):
        self.paused = not self.paused
