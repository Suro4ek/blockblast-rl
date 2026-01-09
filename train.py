"""Training script for Block Blast RL agent."""

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from blockblast.env import BlockBlastEnv


def make_env(seed: int = 0):
    """Create a wrapped BlockBlast environment."""
    def _init():
        env = BlockBlastEnv()
        env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Block Blast RL agent")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for tensorboard logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume training")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency")
    args = parser.parse_args()

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

    # Create or load model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)
    else:
        print("Creating new model...")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(log_dir),
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=str(save_dir),
        name_prefix="blockblast",
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(log_dir),
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    print(f"Training for {args.steps} steps...")
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = save_dir / "final_model"
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
