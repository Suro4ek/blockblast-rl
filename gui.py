"""GUI visualization for Block Blast using pygame."""

import argparse
import sys

import pygame
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from blockblast.env import BlockBlastEnv
from blockblast.game import GRID_SIZE


# Colors
BACKGROUND = (30, 30, 40)
GRID_BG = (50, 50, 65)
GRID_LINES = (70, 70, 85)
EMPTY_CELL = (60, 60, 75)
FILLED_CELL = (100, 200, 255)
BLOCK_COLORS = [
    (255, 100, 100),  # Red
    (100, 255, 100),  # Green
    (255, 255, 100),  # Yellow
]
TEXT_COLOR = (220, 220, 230)
HIGHLIGHT = (255, 255, 255, 100)

# Layout
CELL_SIZE = 50
GRID_PADDING = 20
BLOCK_PREVIEW_SIZE = 18
BLOCK_PANEL_WIDTH = 180
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + GRID_PADDING * 2 + BLOCK_PANEL_WIDTH + 40
WINDOW_HEIGHT = max(GRID_SIZE * CELL_SIZE + GRID_PADDING * 2 + 100, 580)


class BlockBlastGUI:
    def __init__(self, model_path: str | None = None):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Block Blast RL Agent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Load model
        self.model = None
        if model_path:
            print(f"Loading model from {model_path}...")
            self.model = MaskablePPO.load(model_path)

        # Environment
        self.env = BlockBlastEnv()
        self.wrapped_env = ActionMasker(self.env, lambda e: e.action_masks())

        # Game state
        self.obs = None
        self.running = True
        self.paused = False
        self.game_over = False
        self.step_delay = 500  # ms between steps
        self.last_step_time = 0
        self.last_action = None
        self.total_reward = 0
        self.steps = 0

    def reset(self):
        """Reset the game."""
        self.obs, _ = self.wrapped_env.reset()
        self.game_over = False
        self.last_action = None
        self.total_reward = 0
        self.steps = 0

    def step(self):
        """Execute one step."""
        if self.game_over or self.paused:
            return

        # Get action
        if self.model is not None:
            action, _ = self.model.predict(
                self.obs, deterministic=True, action_masks=self.env.action_masks()
            )
        else:
            # Random valid action
            valid_actions = self.env.game.get_valid_actions()
            if not valid_actions:
                self.game_over = True
                return
            block_idx, row, col = valid_actions[np.random.randint(len(valid_actions))]
            action = block_idx * 64 + row * 8 + col

        self.last_action = action
        self.obs, reward, terminated, truncated, _ = self.wrapped_env.step(action)
        self.total_reward += reward
        self.steps += 1

        if terminated or truncated:
            self.game_over = True

    def draw_grid(self):
        """Draw the main game grid."""
        grid_x = GRID_PADDING
        grid_y = GRID_PADDING + 60

        # Grid background
        pygame.draw.rect(
            self.screen, GRID_BG,
            (grid_x - 5, grid_y - 5,
             GRID_SIZE * CELL_SIZE + 10, GRID_SIZE * CELL_SIZE + 10),
            border_radius=10
        )

        # Cells
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = grid_x + col * CELL_SIZE
                y = grid_y + row * CELL_SIZE

                if self.env.game.grid[row, col] == 1:
                    color = FILLED_CELL
                else:
                    color = EMPTY_CELL

                pygame.draw.rect(
                    self.screen, color,
                    (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4),
                    border_radius=5
                )

        # Highlight last placed block
        if self.last_action is not None and not self.game_over:
            block_idx = self.last_action // 64
            pos = self.last_action % 64
            row, col = pos // 8, pos % 8

            highlight_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(
                highlight_surface, (*BLOCK_COLORS[block_idx], 100),
                (0, 0, CELL_SIZE, CELL_SIZE),
                border_radius=5
            )

    def draw_blocks(self):
        """Draw available blocks panel."""
        panel_x = GRID_SIZE * CELL_SIZE + GRID_PADDING * 2 + 20
        panel_y = GRID_PADDING + 60

        # Panel title
        title = self.small_font.render("Available Blocks:", True, TEXT_COLOR)
        self.screen.blit(title, (panel_x, panel_y - 30))

        # Block preview area size (5x5 max block size)
        preview_area = 5 * BLOCK_PREVIEW_SIZE
        block_height = preview_area + 30  # Extra space for label

        for i, block in enumerate(self.env.game.current_blocks):
            block_y = panel_y + i * (block_height + 10)

            # Block container background
            pygame.draw.rect(
                self.screen, (45, 45, 55),
                (panel_x - 5, block_y - 5, BLOCK_PANEL_WIDTH - 20, block_height),
                border_radius=8
            )

            # Block index label
            label_color = BLOCK_COLORS[i] if block else (80, 80, 90)
            label = self.small_font.render(f"[{i + 1}]", True, label_color)
            self.screen.blit(label, (panel_x + 5, block_y + 2))

            if block is None:
                # Used indicator
                used_text = self.font.render("Used", True, (70, 70, 80))
                self.screen.blit(used_text, (panel_x + 50, block_y + preview_area // 2 - 5))
            else:
                # Block name
                name_text = self.small_font.render(block.name, True, (150, 150, 160))
                self.screen.blit(name_text, (panel_x + 40, block_y + 2))

                # Draw block shape centered in preview area
                shape_width = block.width * BLOCK_PREVIEW_SIZE
                shape_height = block.height * BLOCK_PREVIEW_SIZE
                offset_x = (preview_area - shape_width) // 2 + 10
                offset_y = 25  # Below label

                for dr in range(block.height):
                    for dc in range(block.width):
                        if block.shape[dr, dc] == 1:
                            x = panel_x + offset_x + dc * BLOCK_PREVIEW_SIZE
                            y = block_y + offset_y + dr * BLOCK_PREVIEW_SIZE
                            pygame.draw.rect(
                                self.screen, BLOCK_COLORS[i],
                                (x + 1, y + 1,
                                 BLOCK_PREVIEW_SIZE - 2, BLOCK_PREVIEW_SIZE - 2),
                                border_radius=3
                            )

    def draw_info(self):
        """Draw game info."""
        # Title
        title = self.font.render("Block Blast RL", True, TEXT_COLOR)
        self.screen.blit(title, (GRID_PADDING, 15))

        # Score
        score_text = self.font.render(f"Score: {self.env.game.score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (GRID_PADDING + 200, 15))

        # Steps
        steps_text = self.small_font.render(f"Steps: {self.steps}", True, TEXT_COLOR)
        self.screen.blit(steps_text, (GRID_PADDING + 350, 20))

        # Status panel at bottom
        status_y = GRID_SIZE * CELL_SIZE + GRID_PADDING + 80

        if self.game_over:
            status = self.font.render("GAME OVER - Press R to restart", True, (255, 100, 100))
        elif self.paused:
            status = self.font.render("PAUSED - Press SPACE to continue", True, (255, 255, 100))
        else:
            mode = "AI Agent" if self.model else "Random"
            status = self.small_font.render(
                f"Mode: {mode} | SPACE: pause | R: restart | +/-: speed | Q: quit",
                True, (150, 150, 160)
            )

        self.screen.blit(status, (GRID_PADDING, status_y))

        # Speed indicator
        speed_text = self.small_font.render(f"Delay: {self.step_delay}ms", True, TEXT_COLOR)
        self.screen.blit(speed_text, (WINDOW_WIDTH - 120, status_y))

    def draw(self):
        """Draw everything."""
        self.screen.fill(BACKGROUND)
        self.draw_grid()
        self.draw_blocks()
        self.draw_info()
        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.step_delay = max(50, self.step_delay - 50)
                elif event.key == pygame.K_MINUS:
                    self.step_delay = min(2000, self.step_delay + 50)

    def run(self):
        """Main game loop."""
        self.reset()

        while self.running:
            self.handle_events()

            # Auto-step
            current_time = pygame.time.get_ticks()
            if current_time - self.last_step_time >= self.step_delay:
                self.step()
                self.last_step_time = current_time

            self.draw()
            self.clock.tick(60)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Block Blast GUI Visualization")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--delay", type=int, default=500, help="Initial delay between steps (ms)")
    args = parser.parse_args()

    gui = BlockBlastGUI(model_path=args.model)
    gui.step_delay = args.delay
    gui.run()


if __name__ == "__main__":
    main()
