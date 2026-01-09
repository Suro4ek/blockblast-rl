"""Gymnasium environment for Block Blast."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from blockblast.game import BlockBlastGame, GRID_SIZE
from blockblast.blocks import BLOCKS


# Maximum block size for observation padding
MAX_BLOCK_SIZE = 5
NUM_BLOCKS = 3


class BlockBlastEnv(gym.Env):
    """
    Gymnasium environment for Block Blast game.

    Observation:
        Dict with:
        - "grid": 8x8 binary array of current board state
        - "blocks": 3 x 5 x 5 array of available blocks (padded)
        - "block_mask": 3-element binary array (1 if block available)

    Action:
        Discrete(3 * 64) = 192 actions
        action = block_idx * 64 + row * 8 + col

    Reward:
        - Points from placing blocks (cells + line bonuses)
        - Small negative reward for invalid moves (masked, shouldn't happen)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.game = BlockBlastGame()

        # Action space: block_idx * 64 + row * 8 + col
        self.action_space = spaces.Discrete(NUM_BLOCKS * GRID_SIZE * GRID_SIZE)

        # Observation space
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0, high=1,
                shape=(GRID_SIZE, GRID_SIZE),
                dtype=np.int8
            ),
            "blocks": spaces.Box(
                low=0, high=1,
                shape=(NUM_BLOCKS, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE),
                dtype=np.int8
            ),
            "block_mask": spaces.Box(
                low=0, high=1,
                shape=(NUM_BLOCKS,),
                dtype=np.int8
            ),
        })

    def _get_obs(self) -> dict:
        """Get current observation."""
        # Pad blocks to MAX_BLOCK_SIZE x MAX_BLOCK_SIZE
        blocks = np.zeros((NUM_BLOCKS, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE), dtype=np.int8)
        block_mask = np.zeros(NUM_BLOCKS, dtype=np.int8)

        for i, block in enumerate(self.game.current_blocks):
            if block is not None:
                blocks[i, :block.height, :block.width] = block.shape
                block_mask[i] = 1

        return {
            "grid": self.game.grid.copy(),
            "blocks": blocks,
            "block_mask": block_mask,
        }

    def _get_info(self) -> dict:
        """Get additional info."""
        return {
            "score": self.game.score,
            "valid_actions": len(self.game.get_valid_actions()),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """Execute one step in the environment."""
        # Decode action
        block_idx = action // (GRID_SIZE * GRID_SIZE)
        position = action % (GRID_SIZE * GRID_SIZE)
        row = position // GRID_SIZE
        col = position % GRID_SIZE

        # Check if action is valid
        block = self.game.current_blocks[block_idx]
        if block is None or not self.game.can_place(block, row, col):
            # Invalid action - this shouldn't happen with proper masking
            return self._get_obs(), -10.0, False, False, self._get_info()

        # Count valid actions before move
        valid_before = len(self.game.get_valid_actions())

        # Execute action
        base_reward = float(self.game.place_block(block_idx, row, col))

        # Check for game over
        terminated = self.game.is_game_over()
        truncated = False

        # Reward shaping for better learning
        reward = base_reward

        if terminated:
            # Penalty for game over (but not too harsh)
            reward -= 20.0
        else:
            # Bonus for keeping options open (more valid moves = better position)
            valid_after = len(self.game.get_valid_actions())
            # Small bonus for maintaining flexibility
            reward += valid_after * 0.05

            # Bonus for keeping grid relatively empty (encourages line clearing)
            fill_ratio = np.sum(self.game.grid) / (GRID_SIZE * GRID_SIZE)
            if fill_ratio < 0.3:
                reward += 2.0  # Bonus for clean board
            elif fill_ratio > 0.7:
                reward -= 1.0  # Slight penalty for crowded board

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def action_masks(self) -> np.ndarray:
        """
        Get action mask for MaskablePPO.

        Returns:
            Boolean array of shape (action_space.n,) where True means action is valid.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)

        for block_idx, row, col in self.game.get_valid_actions():
            action = block_idx * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col
            mask[action] = True

        return mask

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(self.game.render())
            print()
        elif self.render_mode == "ansi":
            return self.game.render()

    def close(self):
        """Clean up resources."""
        pass


# Register the environment
gym.register(
    id="BlockBlast-v0",
    entry_point="blockblast.env:BlockBlastEnv",
)
