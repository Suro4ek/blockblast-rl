"""DQN training script for Block Blast with action masking."""

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from blockblast.env import BlockBlastEnv


class DQNetwork(nn.Module):
    """Deep Q-Network for Block Blast."""

    def __init__(self, n_actions: int):
        super().__init__()

        # Grid encoder (8x8 -> features)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )  # Output: 64 * 8 * 8 = 4096

        # Block encoder (3 x 5x5 -> features)
        self.block_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )  # Output: 32 * 5 * 5 = 800

        # Combined network
        self.fc = nn.Sequential(
            nn.Linear(4096 + 800 + 3, 512),  # +3 for block_mask
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, grid, blocks, block_mask):
        # Grid: (batch, 8, 8) -> (batch, 1, 8, 8)
        grid = grid.unsqueeze(1).float()
        grid_features = self.grid_encoder(grid)

        # Blocks: (batch, 3, 5, 5) -> already correct shape
        blocks = blocks.float()
        block_features = self.block_encoder(blocks)

        # Block mask: (batch, 3)
        block_mask = block_mask.float()

        # Combine all features
        combined = torch.cat([grid_features, block_features, block_mask], dim=1)
        q_values = self.fc(combined)

        return q_values


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)

        return (
            {k: np.array([s[k] for s in states]) for k in states[0].keys()},
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            {k: np.array([s[k] for s in next_states]) for k in next_states[0].keys()},
            np.array(dones, dtype=np.float32),
            np.array(masks),
            np.array(next_masks),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with action masking."""

    def __init__(
        self,
        n_actions: int,
        device: str = "cuda",
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        target_update: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 100_000,
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # Networks
        self.policy_net = DQNetwork(n_actions).to(device)
        self.target_net = DQNetwork(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Step counter
        self.steps = 0

    def get_epsilon(self):
        """Get current epsilon for exploration."""
        return self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-self.steps / self.eps_decay)

    def select_action(self, state: dict, action_mask: np.ndarray) -> int:
        """Select action using epsilon-greedy with masking."""
        epsilon = self.get_epsilon()

        if random.random() < epsilon:
            # Random valid action
            valid_actions = np.where(action_mask)[0]
            return np.random.choice(valid_actions)
        else:
            # Greedy action
            with torch.no_grad():
                grid = torch.tensor(state["grid"], device=self.device).unsqueeze(0)
                blocks = torch.tensor(state["blocks"], device=self.device).unsqueeze(0)
                block_mask = torch.tensor(state["block_mask"], device=self.device).unsqueeze(0)

                q_values = self.policy_net(grid, blocks, block_mask)

                # Mask invalid actions with large negative value
                mask_tensor = torch.tensor(action_mask, device=self.device)
                q_values[~mask_tensor.unsqueeze(0)] = -1e9

                return q_values.argmax(dim=1).item()

    def update(self):
        """Update the network."""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones, masks, next_masks = \
            self.buffer.sample(self.batch_size)

        # Convert to tensors
        grid = torch.tensor(states["grid"], device=self.device)
        blocks = torch.tensor(states["blocks"], device=self.device)
        block_mask = torch.tensor(states["block_mask"], device=self.device)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        next_grid = torch.tensor(next_states["grid"], device=self.device)
        next_blocks = torch.tensor(next_states["blocks"], device=self.device)
        next_block_mask = torch.tensor(next_states["block_mask"], device=self.device)
        next_masks = torch.tensor(next_masks, device=self.device)

        # Current Q values
        current_q = self.policy_net(grid, blocks, block_mask)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values (Double DQN)
        with torch.no_grad():
            # Use policy net to select actions
            next_q_policy = self.policy_net(next_grid, next_blocks, next_block_mask)
            next_q_policy[~next_masks] = -1e9
            next_actions = next_q_policy.argmax(dim=1)

            # Use target net to evaluate
            next_q_target = self.target_net(next_grid, next_blocks, next_block_mask)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = nn.SmoothL1Loss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        """Save model."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]


def train(args):
    """Train DQN agent."""
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Environment
    env = BlockBlastEnv()
    n_actions = env.action_space.n

    # Agent
    agent = DQNAgent(
        n_actions=n_actions,
        device=device,
        lr=args.lr,
        gamma=0.99,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        eps_decay=args.eps_decay,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        agent.load(args.resume)

    # Training loop
    episode_rewards = []
    episode_lengths = []
    best_reward = 0
    losses = []

    pbar = tqdm(total=args.steps, desc="Training")

    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    while agent.steps < args.steps:
        # Get action mask
        action_mask = env.action_masks()

        # Select action
        action = agent.select_action(state, action_mask)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = env.action_masks() if not done else np.zeros(n_actions, dtype=bool)

        # Store transition
        agent.buffer.push(state, action, reward, next_state, done, action_mask, next_mask)

        # Update
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        # Update target network
        if agent.steps % agent.target_update == 0:
            agent.update_target()

        episode_reward += reward
        episode_length += 1
        agent.steps += 1
        pbar.update(1)

        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Logging
            if len(episode_rewards) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                mean_length = np.mean(episode_lengths[-100:])
                mean_loss = np.mean(losses[-1000:]) if losses else 0
                epsilon = agent.get_epsilon()

                pbar.set_postfix({
                    "reward": f"{mean_reward:.1f}",
                    "length": f"{mean_length:.1f}",
                    "eps": f"{epsilon:.3f}",
                    "loss": f"{mean_loss:.4f}",
                })

                # Save best model
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    agent.save(str(save_dir / "best_model.pt"))

            # Save checkpoint
            if len(episode_rewards) % 1000 == 0:
                agent.save(str(save_dir / f"checkpoint_{agent.steps}.pt"))

            # Reset
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            state = next_state

    pbar.close()

    # Save final model
    agent.save(str(save_dir / "final_model.pt"))
    print(f"\nTraining complete!")
    print(f"Best mean reward: {best_reward:.1f}")
    print(f"Final model saved to {save_dir / 'final_model.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train Block Blast with DQN")
    parser.add_argument("--steps", type=int, default=5_000_000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer size")
    parser.add_argument("--target-update", type=int, default=2000, help="Target network update frequency")
    parser.add_argument("--eps-decay", type=int, default=200_000, help="Epsilon decay steps")
    parser.add_argument("--save-dir", type=str, default="models_dqn", help="Save directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
