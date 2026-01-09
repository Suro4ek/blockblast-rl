"""Block Blast game logic."""

import numpy as np
from dataclasses import dataclass, field

from blockblast.blocks import Block, get_random_block


GRID_SIZE = 8


@dataclass
class BlockBlastGame:
    """Block Blast game state and logic."""

    grid: np.ndarray = field(default_factory=lambda: np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8))
    current_blocks: list[Block | None] = field(default_factory=lambda: [None, None, None])
    score: int = 0
    _rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def reset(self, seed: int | None = None) -> None:
        """Reset game to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.grid.fill(0)
        self.score = 0
        self._deal_new_blocks()

    def _deal_new_blocks(self) -> None:
        """Deal 3 new blocks when all current blocks are used."""
        self.current_blocks = [
            get_random_block(self._rng),
            get_random_block(self._rng),
            get_random_block(self._rng),
        ]

    def can_place(self, block: Block, row: int, col: int) -> bool:
        """Check if a block can be placed at the given position."""
        if row < 0 or col < 0:
            return False
        if row + block.height > GRID_SIZE or col + block.width > GRID_SIZE:
            return False

        # Check for overlap with existing blocks
        for dr in range(block.height):
            for dc in range(block.width):
                if block.shape[dr, dc] == 1 and self.grid[row + dr, col + dc] == 1:
                    return False
        return True

    def place_block(self, block_idx: int, row: int, col: int) -> int:
        """
        Place a block on the grid.

        Returns:
            Points earned from this placement.
        """
        block = self.current_blocks[block_idx]
        if block is None:
            raise ValueError(f"Block {block_idx} already used")
        if not self.can_place(block, row, col):
            raise ValueError(f"Cannot place block at ({row}, {col})")

        # Place the block
        for dr in range(block.height):
            for dc in range(block.width):
                if block.shape[dr, dc] == 1:
                    self.grid[row + dr, col + dc] = 1

        # Mark block as used
        self.current_blocks[block_idx] = None

        # Clear lines and calculate score
        cells_placed = block.size
        lines_cleared = self._clear_lines()

        # Scoring: cells + bonus for lines
        points = cells_placed + lines_cleared * 10
        if lines_cleared > 1:
            points += (lines_cleared - 1) * 20  # Combo bonus

        self.score += points

        # Deal new blocks if all are used
        if all(b is None for b in self.current_blocks):
            self._deal_new_blocks()

        return points

    def _clear_lines(self) -> int:
        """Clear completed rows and columns. Returns number of lines cleared."""
        lines_cleared = 0

        # Find complete rows
        rows_to_clear = []
        for r in range(GRID_SIZE):
            if np.all(self.grid[r, :] == 1):
                rows_to_clear.append(r)

        # Find complete columns
        cols_to_clear = []
        for c in range(GRID_SIZE):
            if np.all(self.grid[:, c] == 1):
                cols_to_clear.append(c)

        # Clear rows
        for r in rows_to_clear:
            self.grid[r, :] = 0
            lines_cleared += 1

        # Clear columns
        for c in cols_to_clear:
            self.grid[:, c] = 0
            lines_cleared += 1

        return lines_cleared

    def get_valid_actions(self) -> list[tuple[int, int, int]]:
        """
        Get all valid (block_idx, row, col) actions.

        Returns:
            List of (block_idx, row, col) tuples.
        """
        actions = []
        for block_idx, block in enumerate(self.current_blocks):
            if block is None:
                continue
            for row in range(GRID_SIZE - block.height + 1):
                for col in range(GRID_SIZE - block.width + 1):
                    if self.can_place(block, row, col):
                        actions.append((block_idx, row, col))
        return actions

    def is_game_over(self) -> bool:
        """Check if no valid moves remain."""
        return len(self.get_valid_actions()) == 0

    def render(self) -> str:
        """Render the game state as a string."""
        lines = []
        lines.append(f"Score: {self.score}")
        lines.append("+" + "-" * GRID_SIZE + "+")

        for row in range(GRID_SIZE):
            line = "|"
            for col in range(GRID_SIZE):
                line += "#" if self.grid[row, col] else "."
            line += "|"
            lines.append(line)

        lines.append("+" + "-" * GRID_SIZE + "+")
        lines.append("")
        lines.append("Available blocks:")

        for i, block in enumerate(self.current_blocks):
            if block is None:
                lines.append(f"  [{i}] (used)")
            else:
                lines.append(f"  [{i}] {block.name}:")
                for row in block.shape:
                    lines.append("      " + "".join("#" if c else "." for c in row))

        return "\n".join(lines)
