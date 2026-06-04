import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for p in (PROJECT_ROOT, SCRIPT_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from train_ppo import Args
from utils import get_env


def _dataset_files(mimic_task: str) -> list[str]:
    return [
        f"dataset/high_level_trajectories/{mimic_task}_left_hand_action_list.npy",
        f"dataset/high_level_trajectories/{mimic_task}_right_hand_action_list.npy",
        f"dataset/notes/{mimic_task}.pkl",
    ]


def _skip_if_missing_dataset(mimic_task: str) -> bool:
    missing = [path for path in _dataset_files(mimic_task) if not os.path.exists(path)]
    if missing:
        print(f"SKIP: Dataset not available, skipping smoothness reward test: {missing}")
        return True
    return False


def _unwrap_task_and_physics(env):
    candidates = [env, getattr(env, "unwrapped", None), getattr(getattr(env, "unwrapped", None), "env", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "task") and hasattr(candidate, "physics"):
            return candidate.task, candidate.physics

    cur = env
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if hasattr(cur, "task") and hasattr(cur, "physics"):
            return cur.task, cur.physics
        cur = getattr(cur, "env", None)
    raise RuntimeError("Could not locate task/physics in wrapper chain")


def _step(env, action):
    out = env.step(action)
    if len(out) == 5:
        return out
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, False, info
    raise RuntimeError(f"Unexpected env.step() return length: {len(out)}")


def test_smoothness_reward() -> int:
    mimic_task = "TwinkleTwinkleRousseau"
    if _skip_if_missing_dataset(mimic_task):
        return 0

    args = Args(
        mimic_task=mimic_task,
        use_note_trajectory=True,
        deepmimic=False,
        residual_action=False,
        enable_smoothness_reward=True,
        smoothness_weight=1.0,
        beta_action=1.0,
        beta_accel=1.0,
    )

    env = get_env(args)
    obs, _ = env.reset()
    del obs

    task, physics = _unwrap_task_and_physics(env)

    assert task._current_action is None, "current_action should be None after reset"
    assert task._prev_action is None, "prev_action should be None after reset"
    assert task._current_qpos is None, "current_qpos should be None after reset"
    assert task._prev_qpos is None, "prev_qpos should be None after reset"
    assert task._prev_prev_qpos is None, "prev_prev_qpos should be None after reset"
    print("Test 1 PASSED: tracking state is None after reset")

    action_space = env.action_space
    zero_action = np.zeros(action_space.shape, dtype=action_space.dtype)

    obs, reward, done, trunc, info = _step(env, zero_action)
    del obs, reward, done, trunc, info
    task, physics = _unwrap_task_and_physics(env)
    assert task._current_action is not None, "current_action should be set after first step"
    assert task._prev_action is None, "prev_action should still be None after first step"
    smooth_first = task._compute_smoothness_reward(physics)
    assert smooth_first == 0.0, f"first-step smoothness should be 0.0, got {smooth_first}"
    task.get_reward(physics)
    assert "smoothness_reward" in task._reward_fn.reward_terms, "smoothness_reward missing from reward terms"
    assert task._reward_fn.reward_terms["smoothness_reward"] == 0.0, "reward term should be 0.0 on first step"
    print("Test 2 PASSED: first-step smoothness is 0.0 and tracked state is partial")

    for _ in range(5):
        obs, reward, done, trunc, info = _step(env, zero_action)
        del obs, reward, done, trunc, info
        if done or trunc:
            break
    task, physics = _unwrap_task_and_physics(env)
    action_rate = float(np.sum((task._current_action[:-1] - task._prev_action[:-1]) ** 2))
    smooth_constant = float(task._compute_smoothness_reward(physics))
    task.get_reward(physics)
    recorded_constant = float(task._reward_fn.reward_terms["smoothness_reward"])
    assert np.isclose(action_rate, 0.0), f"constant-action rate penalty should be ~0, got {action_rate}"
    assert np.isclose(smooth_constant, recorded_constant), "direct and recorded smoothness should match"
    assert smooth_constant > 0.0, f"constant-action smoothness should be positive, got {smooth_constant}"
    print(f"Test 3 PASSED: constant actions -> action_rate={action_rate:.6f}, smoothness={smooth_constant:.6f}")

    high = np.full(action_space.shape, 0.5, dtype=action_space.dtype)
    low = np.full(action_space.shape, -0.5, dtype=action_space.dtype)
    for i in range(6):
        action = high if i % 2 == 0 else low
        action = np.clip(action, action_space.low, action_space.high).astype(action_space.dtype)
        obs, reward, done, trunc, info = _step(env, action)
        del obs, reward, done, trunc, info
        if done or trunc:
            break
    task, physics = _unwrap_task_and_physics(env)
    smooth_alternating = float(task._compute_smoothness_reward(physics))
    task.get_reward(physics)
    recorded_alternating = float(task._reward_fn.reward_terms["smoothness_reward"])
    assert np.isclose(smooth_alternating, recorded_alternating), "direct and recorded alternating smoothness should match"
    assert smooth_alternating < smooth_constant, (
        f"alternating smoothness should decrease vs constant: {smooth_alternating} !< {smooth_constant}"
    )
    print(f"Test 4 PASSED: alternating smoothness decreased to {smooth_alternating:.6f}")

    args_disabled = Args(
        mimic_task=mimic_task,
        use_note_trajectory=True,
        enable_smoothness_reward=False,
    )
    env2 = get_env(args_disabled)
    task2, _ = _unwrap_task_and_physics(env2)
    assert "smoothness_reward" not in task2._reward_fn.reward_fns, (
        "smoothness_reward should not be registered when disabled"
    )
    print("Test 5 PASSED: enable_smoothness_reward=False prevents registration")

    print("All smoothness reward tests PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(test_smoothness_reward())
