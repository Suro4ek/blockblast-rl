"""Block definitions for Block Blast game."""

import numpy as np
from dataclasses import dataclass


@dataclass
class Block:
    """A block shape that can be placed on the grid."""
    name: str
    shape: np.ndarray

    @property
    def height(self) -> int:
        return self.shape.shape[0]

    @property
    def width(self) -> int:
        return self.shape.shape[1]

    @property
    def size(self) -> int:
        """Number of filled cells."""
        return int(np.sum(self.shape))


# Define all block shapes
BLOCKS = [
    # Single dot
    Block("dot", np.array([[1]])),

    # Horizontal lines
    Block("h2", np.array([[1, 1]])),
    Block("h3", np.array([[1, 1, 1]])),
    Block("h4", np.array([[1, 1, 1, 1]])),
    Block("h5", np.array([[1, 1, 1, 1, 1]])),

    # Vertical lines
    Block("v2", np.array([[1], [1]])),
    Block("v3", np.array([[1], [1], [1]])),
    Block("v4", np.array([[1], [1], [1], [1]])),
    Block("v5", np.array([[1], [1], [1], [1], [1]])),

    # Squares
    Block("sq2", np.array([[1, 1], [1, 1]])),
    Block("sq3", np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),

    # L-shapes
    Block("L1", np.array([[1, 0], [1, 0], [1, 1]])),
    Block("L2", np.array([[0, 1], [0, 1], [1, 1]])),
    Block("L3", np.array([[1, 1], [1, 0], [1, 0]])),
    Block("L4", np.array([[1, 1], [0, 1], [0, 1]])),

    # Rotated L-shapes (horizontal)
    Block("L5", np.array([[1, 0, 0], [1, 1, 1]])),
    Block("L6", np.array([[0, 0, 1], [1, 1, 1]])),
    Block("L7", np.array([[1, 1, 1], [1, 0, 0]])),
    Block("L8", np.array([[1, 1, 1], [0, 0, 1]])),

    # T-shapes
    Block("T1", np.array([[1, 1, 1], [0, 1, 0]])),
    Block("T2", np.array([[0, 1, 0], [1, 1, 1]])),
    Block("T3", np.array([[1, 0], [1, 1], [1, 0]])),
    Block("T4", np.array([[0, 1], [1, 1], [0, 1]])),

    # Corners (2x2 with one missing)
    Block("corner1", np.array([[1, 1], [1, 0]])),
    Block("corner2", np.array([[1, 1], [0, 1]])),
    Block("corner3", np.array([[1, 0], [1, 1]])),
    Block("corner4", np.array([[0, 1], [1, 1]])),

    # Big L-shapes (3x3)
    Block("bigL1", np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])),
    Block("bigL2", np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])),
    Block("bigL3", np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]])),
    Block("bigL4", np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])),
]


def get_random_block(rng: np.random.Generator | None = None) -> Block:
    """Get a random block."""
    if rng is None:
        rng = np.random.default_rng()
    return BLOCKS[rng.integers(0, len(BLOCKS))]
