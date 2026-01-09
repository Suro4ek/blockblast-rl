"""Play Block Blast with a trained agent or randomly."""

import argparse
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from blockblast.env import BlockBlastEnv


def play_game(model=None, delay: float = 0.5, seed: int | None = None):
    """Play a single game and display it."""
    env = BlockBlastEnv()
    wrapped_env = ActionMasker(env, lambda e: e.action_masks())

    obs, info = wrapped_env.reset(seed=seed)
    total_reward = 0
    steps = 0

    print("\n" + "=" * 40)
    print("BLOCK BLAST - New Game")
    print("=" * 40)
    print(env.game.render())

    while True:
        # Get action
        if model is not None:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        else:
            # Random valid action
            valid_actions = env.game.get_valid_actions()
            if not valid_actions:
                break
            block_idx, row, col = valid_actions[0]
            action = block_idx * 64 + row * 8 + col

        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        total_reward += reward
        steps += 1

        # Decode action for display
        block_idx = action // 64
        position = action % 64
        row, col = position // 8, position % 8

        print(f"\nStep {steps}: Block {block_idx} at ({row}, {col}) | Reward: {reward:.0f}")
        print(env.game.render())

        if delay > 0:
            time.sleep(delay)

        if terminated or truncated:
            break

    print("\n" + "=" * 40)
    print(f"GAME OVER! Final Score: {env.game.score}")
    print(f"Total Steps: {steps} | Total Reward: {total_reward:.0f}")
    print("=" * 40)

    return env.game.score, steps


def main():
    parser = argparse.ArgumentParser(description="Play Block Blast with trained agent")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between moves (seconds)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Load model if provided
    model = None
    if args.model:
        print(f"Loading model from {args.model}...")
        model = MaskablePPO.load(args.model)
    else:
        print("No model provided, playing with first valid action (greedy)...")

    scores = []
    for i in range(args.games):
        seed = args.seed + i if args.seed is not None else None
        score, steps = play_game(model, delay=args.delay, seed=seed)
        scores.append(score)

    if args.games > 1:
        print(f"\nSummary over {args.games} games:")
        print(f"  Average Score: {sum(scores) / len(scores):.1f}")
        print(f"  Max Score: {max(scores)}")
        print(f"  Min Score: {min(scores)}")


if __name__ == "__main__":
    main()
