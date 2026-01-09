"""Training script for Block Blast RL agent."""

import argparse
from pathlib import Path

import torch
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from blockblast.env import BlockBlastEnv


class LoggingCallback(BaseCallback):
    """Custom callback for logging training progress."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log every log_freq steps
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                print(f"Step {self.num_timesteps}: "
                      f"Mean reward (last 100): {mean_reward:.1f}, "
                      f"Mean length: {mean_length:.1f}")
        return True

    def _on_rollout_end(self) -> None:
        # Collect episode stats from infos
        pass


def make_env(seed: int = 0):
    """Create a wrapped BlockBlast environment."""
    def _init():
        env = BlockBlastEnv()
        env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main():
    parser = argparse.ArgumentParser(description="Train Block Blast RL agent")
    parser.add_argument("--steps", type=int, default=5_000_000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for tensorboard logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume training")
    parser.add_argument("--eval-freq", type=int, default=25_000, help="Evaluation frequency")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Create directories
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Create vectorized environment
    print(f"Creating {args.n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(42)])

    # Network architecture - larger for better learning
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Policy network
            vf=[256, 256, 128],  # Value network
        ),
    )

    # Create or load model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env, device=device)
    else:
        print("Creating new model...")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=linear_schedule(args.lr),  # LR decay
            n_steps=2048,  # Steps per env before update
            batch_size=args.batch_size,
            n_epochs=10,  # PPO epochs per update
            gamma=0.995,  # Discount factor (higher = more long-term thinking)
            gae_lambda=0.95,  # GAE lambda
            clip_range=0.2,  # PPO clip range
            clip_range_vf=0.2,  # Value function clip
            ent_coef=0.01,  # Entropy coefficient (exploration)
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log=str(log_dir),
        )

    print(f"\nModel architecture:")
    print(f"  Policy network: {policy_kwargs['net_arch']['pi']}")
    print(f"  Value network: {policy_kwargs['net_arch']['vf']}")
    print(f"  Learning rate: {args.lr} (with linear decay)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {args.steps:,}")
    print()

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.eval_freq // args.n_envs, 1),
        save_path=str(save_dir),
        name_prefix="blockblast",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(log_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=20,  # More episodes for stable evaluation
        deterministic=True,
    )

    logging_callback = LoggingCallback(log_freq=10000)

    # Train
    print(f"Training for {args.steps:,} steps...")
    print("=" * 50)
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_callback, eval_callback, logging_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = save_dir / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
