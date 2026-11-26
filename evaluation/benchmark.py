"""Benchmark harness for local agent performance and latency profiling.

Evaluates DoomAgent on specified WAD maps, collecting step latencies,
actions, rewards, and survival outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doom_env import make_env
from submission.agent import DoomAgent


def run_benchmark(
    wad_path: str = "maps/two_rooms.wad",
    map_name: str = "MAP01",
    num_episodes: int = 3,
    max_ticks: int = 2100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute evaluation episodes and compute summary metrics."""
    env = make_env(wad_path, map_name=map_name, episode_ticks=max_ticks)
    agent = DoomAgent(env.observation_space, env.action_space, device=device)

    all_latencies: list[float] = []
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    outcomes: list[str] = []

    for ep in range(num_episodes):
        agent.reset()
        obs, info = env.reset(seed=ep * 42)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            t0 = time.perf_counter()
            action = agent.act(obs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            all_latencies.append(latency_ms)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        outcomes.append("exit" if terminated and total_reward > 0 else "death" if terminated else "timeout")

    env.close()

    lat_arr = np.array(all_latencies)
    results = {
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_steps": float(np.mean(episode_steps)),
        "outcomes": outcomes,
        "latency_p50_ms": float(np.percentile(lat_arr, 50)),
        "latency_p90_ms": float(np.percentile(lat_arr, 90)),
        "latency_p99_ms": float(np.percentile(lat_arr, 99)),
        "latency_max_ms": float(np.max(lat_arr)),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local Doom agent benchmark.")
    parser.add_argument("--wad", type=str, default="maps/two_rooms.wad", help="Path to .wad file")
    parser.add_argument("--map", type=str, default="MAP01", help="Map name (e.g. MAP01)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of benchmark episodes")
    args = parser.parse_args()

    print(f"--- Running Benchmark on {args.wad}:{args.map} ({args.episodes} episodes) ---")
    res = run_benchmark(wad_path=args.wad, map_name=args.map, num_episodes=args.episodes)
    for k, v in res.items():
        print(f"  {k}: {v}")
