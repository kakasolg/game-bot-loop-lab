"""Phase 1: Independent CPU-only NumPy BC baseline on BabyAI-GoToLocal-v0."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.utils.baby_ai_bot import BabyAIBot

MISSION_RE = re.compile(r"^go to (?:a|the) ([a-z]+) ([a-z]+)$")
ACTION_NAMES = ("left", "right", "forward")
ALLOWED_ACTIONS = {0, 1, 2}


def parse_mission(mission: str) -> tuple[int, int]:
    match = MISSION_RE.match(mission.strip())
    if not match:
        raise ValueError(f"Unsupported mission format: {mission!r}")
    color_name, type_name = match.groups()
    return COLOR_TO_IDX[color_name], OBJECT_TO_IDX[type_name]


def encode_observation(obs: dict) -> tuple[np.ndarray, dict]:
    """Convert current (obs['image'], obs['mission']) into a 247-d task-relative vector.

    This is an engineered, task-specific representation using ONLY current-step
    egocentric observation and mission strings (no history, no global coordinates,
    no expert map).
    """
    img = obs["image"]
    target_color_idx, target_type_idx = parse_mission(obs["mission"])

    unseen_idx = OBJECT_TO_IDX["unseen"]
    empty_idx = OBJECT_TO_IDX["empty"]
    wall_idx = OBJECT_TO_IDX["wall"]

    obj_grid = img[:, :, 0]
    color_grid = img[:, :, 1]

    cells = np.zeros((7, 7, 5), dtype=np.float32)
    is_unseen = obj_grid == unseen_idx
    is_empty = obj_grid == empty_idx
    is_wall = obj_grid == wall_idx
    is_target = (obj_grid == target_type_idx) & (color_grid == target_color_idx)
    is_non_target = ~(is_unseen | is_empty | is_wall | is_target)

    cells[:, :, 0] = is_unseen
    cells[:, :, 1] = is_empty
    cells[:, :, 2] = is_wall
    cells[:, :, 3] = is_target
    cells[:, :, 4] = is_non_target

    # Front cell is (vx=3, vy=5) in MiniGrid's 7x7 egocentric view.
    front_blocked = float(not (is_empty[3, 5] or is_unseen[3, 5]))
    target_in_view = float(np.any(is_target))

    vec = np.concatenate(
        [cells.reshape(-1), np.array([front_blocked, target_in_view], dtype=np.float32)]
    )
    meta = {
        "front_blocked": bool(front_blocked > 0.5),
        "target_in_view": bool(target_in_view > 0.5),
    }
    return vec, meta


def verify_feature_assertions() -> None:
    """Verify feature extraction on unseen cells, front cell, and duplicate targets."""
    empty_idx = OBJECT_TO_IDX["empty"]
    unseen_idx = OBJECT_TO_IDX["unseen"]
    wall_idx = OBJECT_TO_IDX["wall"]
    key_idx = OBJECT_TO_IDX["key"]
    ball_idx = OBJECT_TO_IDX["ball"]

    def make_blank_img() -> np.ndarray:
        arr = np.zeros((7, 7, 3), dtype=np.uint8)
        arr[:, :, 0] = empty_idx
        return arr

    # 1. Unseen cell test
    img = make_blank_img()
    img[0, 0, 0] = unseen_idx
    vec, meta = encode_observation({"image": img, "mission": "go to the red ball"})
    grid_feats = vec[: 49 * 5].reshape(7, 7, 5)
    assert np.array_equal(grid_feats[0, 0], [1, 0, 0, 0, 0]), "unseen channel mismatch"
    assert not meta["front_blocked"] and not meta["target_in_view"]

    # 2. Front cell (3, 5) tests: empty vs wall vs non-target distractor vs unseen
    img_wall = make_blank_img()
    img_wall[3, 5, 0] = wall_idx
    vec_w, meta_w = encode_observation({"image": img_wall, "mission": "go to the red ball"})
    assert vec_w[49 * 5] == 1.0 and meta_w["front_blocked"]
    assert np.array_equal(vec_w[: 49 * 5].reshape(7, 7, 5)[3, 5], [0, 0, 1, 0, 0])

    img_dist = make_blank_img()
    img_dist[3, 5, 0] = key_idx
    img_dist[3, 5, 1] = COLOR_TO_IDX["blue"]
    vec_d, meta_d = encode_observation({"image": img_dist, "mission": "go to the red ball"})
    assert vec_d[49 * 5] == 1.0 and meta_d["front_blocked"]
    assert np.array_equal(vec_d[: 49 * 5].reshape(7, 7, 5)[3, 5], [0, 0, 0, 0, 1])

    img_unseen_front = make_blank_img()
    img_unseen_front[3, 5, 0] = unseen_idx
    vec_u, meta_u = encode_observation(
        {"image": img_unseen_front, "mission": "go to the red ball"}
    )
    assert vec_u[49 * 5] == 0.0 and not meta_u["front_blocked"]

    # 3. Duplicate matching targets + wrong-color / wrong-type distractors
    img_dup = make_blank_img()
    img_dup[1, 2, :2] = [key_idx, COLOR_TO_IDX["grey"]]  # matching target 1
    img_dup[5, 4, :2] = [key_idx, COLOR_TO_IDX["grey"]]  # matching target 2
    img_dup[2, 2, :2] = [ball_idx, COLOR_TO_IDX["grey"]]  # same color, wrong type
    img_dup[4, 2, :2] = [key_idx, COLOR_TO_IDX["red"]]  # wrong color, same type
    vec_dup, meta_dup = encode_observation(
        {"image": img_dup, "mission": "go to a grey key"}
    )
    gf = vec_dup[: 49 * 5].reshape(7, 7, 5)
    assert np.array_equal(gf[1, 2], [0, 0, 0, 1, 0])
    assert np.array_equal(gf[5, 4], [0, 0, 0, 1, 0])
    assert np.array_equal(gf[2, 2], [0, 0, 0, 0, 1])
    assert np.array_equal(gf[4, 2], [0, 0, 0, 0, 1])
    assert vec_dup[49 * 5 + 1] == 1.0 and meta_dup["target_in_view"]

    # Distractors alone should not set target_in_view
    img_nodup = make_blank_img()
    img_nodup[2, 2, :2] = [ball_idx, COLOR_TO_IDX["grey"]]
    img_nodup[4, 2, :2] = [key_idx, COLOR_TO_IDX["red"]]
    vec_no, meta_no = encode_observation(
        {"image": img_nodup, "mission": "go to a grey key"}
    )
    assert vec_no[49 * 5 + 1] == 0.0 and not meta_no["target_in_view"]


def collect_expert_split(
    split_name: str,
    seeds: list[int],
    deadline: float,
) -> tuple[dict, bool]:
    features: list[np.ndarray] = []
    actions: list[int] = []
    front_blocked_flags: list[bool] = []
    target_visible_flags: list[bool] = []

    anomalies: list[dict] = []
    success_count = 0
    timeout_count = 0
    exception_count = 0
    unexpected_action_count = 0
    episode_lengths: list[int] = []
    action_counts: Counter[str] = Counter()
    timed_out_wall_clock = False

    for seed in seeds:
        if time.monotonic() >= deadline:
            timed_out_wall_clock = True
            break

        env = gym.make("BabyAI-GoToLocal-v0")
        obs, _ = env.reset(seed=seed)
        bot = BabyAIBot(env)
        terminated = False
        truncated = False
        total_reward = 0.0
        prev_action = None
        ep_len = 0
        ep_exc = None

        try:
            while not (terminated or truncated):
                action_obj = bot.replan(prev_action)
                action = int(action_obj)
                if action not in ALLOWED_ACTIONS:
                    unexpected_action_count += 1
                    anomalies.append(
                        {
                            "split": split_name,
                            "seed": seed,
                            "step": ep_len,
                            "type": "unexpected_action",
                            "action": action,
                        }
                    )
                    break

                feat, meta = encode_observation(obs)
                features.append(feat)
                actions.append(action)
                front_blocked_flags.append(meta["front_blocked"])
                target_visible_flags.append(meta["target_in_view"])
                action_counts[ACTION_NAMES[action]] += 1

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                prev_action = action_obj
                ep_len += 1
        except Exception as exc:  # noqa: BLE001
            exception_count += 1
            ep_exc = f"{type(exc).__name__}: {exc}"
            anomalies.append(
                {
                    "split": split_name,
                    "seed": seed,
                    "step": ep_len,
                    "type": "exception",
                    "error": ep_exc,
                }
            )
        finally:
            env.close()

        episode_lengths.append(ep_len)
        if terminated and total_reward > 0.0 and ep_exc is None:
            success_count += 1
        if truncated:
            timeout_count += 1
            anomalies.append(
                {"split": split_name, "seed": seed, "step": ep_len, "type": "timeout"}
            )

    data = {
        "summary": {
            "split": split_name,
            "requested_episodes": len(seeds),
            "collected_episodes": len(episode_lengths),
            "success_count": success_count,
            "timeout_count": timeout_count,
            "exception_count": exception_count,
            "unexpected_action_count": unexpected_action_count,
            "total_steps": len(actions),
            "mean_episode_length": (
                round(sum(episode_lengths) / len(episode_lengths), 4)
                if episode_lengths
                else 0.0
            ),
            "min_episode_length": min(episode_lengths) if episode_lengths else 0,
            "max_episode_length": max(episode_lengths) if episode_lengths else 0,
            "action_counts": dict(action_counts),
            "front_blocked_steps": int(sum(front_blocked_flags)),
            "target_in_view_steps": int(sum(target_visible_flags)),
            "anomalies": anomalies,
            "stopped_by_wall_clock_cap": timed_out_wall_clock,
        },
        "X": np.stack(features) if features else np.zeros((0, 247), dtype=np.float32),
        "y": np.array(actions, dtype=np.int64),
        "front_blocked": np.array(front_blocked_flags, dtype=bool),
        "target_in_view": np.array(target_visible_flags, dtype=bool),
    }
    return data, timed_out_wall_clock


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward_mlp(
    X: np.ndarray, params: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = X @ params["W1"] + params["b1"]
    h1 = np.maximum(0.0, z1)
    logits = h1 @ params["W2"] + params["b2"]
    probs = softmax(logits)
    return z1, h1, probs


def evaluate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    front_blocked: np.ndarray,
    target_in_view: np.ndarray,
    params: dict[str, np.ndarray],
    l2_wd: float,
) -> dict:
    _, _, probs = forward_mlp(X, params)
    n = len(y)
    eps = 1e-12
    ce = float(-np.mean(np.log(np.clip(probs[np.arange(n), y], eps, 1.0))))
    l2_pen = 0.5 * l2_wd * float(np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    preds = np.argmax(probs, axis=1)
    overall_acc = float(np.mean(preds == y))

    per_action = {}
    for cls_idx, cls_name in enumerate(ACTION_NAMES):
        mask = y == cls_idx
        cnt = int(np.sum(mask))
        per_action[cls_name] = {
            "count": cnt,
            "agreement": round(float(np.mean(preds[mask] == y[mask])), 6) if cnt else None,
        }

    vis_mask = target_in_view
    hid_mask = ~target_in_view
    blk_mask = front_blocked
    blk_pred_counts = (
        {ACTION_NAMES[i]: int(np.sum(preds[blk_mask] == i)) for i in range(3)}
        if np.any(blk_mask)
        else {name: 0 for name in ACTION_NAMES}
    )

    return {
        "num_steps": int(n),
        "cross_entropy": round(ce, 6),
        "loss_with_l2": round(ce + l2_pen, 6),
        "overall_agreement": round(overall_acc, 6),
        "per_action": per_action,
        "visible_target": {
            "count": int(np.sum(vis_mask)),
            "agreement": (
                round(float(np.mean(preds[vis_mask] == y[vis_mask])), 6)
                if np.any(vis_mask)
                else None
            ),
        },
        "hidden_target": {
            "count": int(np.sum(hid_mask)),
            "agreement": (
                round(float(np.mean(preds[hid_mask] == y[hid_mask])), 6)
                if np.any(hid_mask)
                else None
            ),
        },
        "blocked_front": {
            "count": int(np.sum(blk_mask)),
            "agreement": (
                round(float(np.mean(preds[blk_mask] == y[blk_mask])), 6)
                if np.any(blk_mask)
                else None
            ),
            "predicted_action_counts": blk_pred_counts,
        },
    }


def train_mlp(
    train_data: dict,
    val_data: dict,
    cfg: dict,
    deadline: float,
) -> tuple[dict[str, np.ndarray], dict, bool]:
    arch = cfg["architecture"]
    tr = cfg["training"]

    rng = np.random.default_rng(arch["init_seed"])
    in_dim = arch["input_dim"]
    hid_dim = arch["hidden_dim"]
    out_dim = arch["output_dim"]

    params = {
        "W1": rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, hid_dim)).astype(
            np.float32
        ),
        "b1": np.zeros(hid_dim, dtype=np.float32),
        "W2": rng.normal(0.0, np.sqrt(1.0 / hid_dim), size=(hid_dim, out_dim)).astype(
            np.float32
        ),
        "b2": np.zeros(out_dim, dtype=np.float32),
    }

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t_adam = 0

    X_tr, y_tr = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    n_tr = len(y_tr)
    bs = tr["batch_size"]
    lr = tr["learning_rate"]
    b1, b2, eps = tr["beta1"], tr["beta2"], tr["eps"]
    wd = tr["l2_weight_decay"]

    shuffle_rng = np.random.default_rng(tr["shuffle_seed"])
    history = []
    best_key = None
    best_epoch = 0
    best_params = {k: val.copy() for k, val in params.items()}
    timed_out_wall_clock = False

    for epoch in range(1, tr["max_epochs"] + 1):
        if time.monotonic() >= deadline:
            timed_out_wall_clock = True
            break

        perm = shuffle_rng.permutation(n_tr)
        for start in range(0, n_tr, bs):
            idx = perm[start : start + bs]
            xb = X_tr[idx]
            yb = y_tr[idx]
            nb = len(yb)

            z1, h1, probs = forward_mlp(xb, params)
            dlogits = probs
            dlogits[np.arange(nb), yb] -= 1.0
            dlogits /= nb

            grads = {
                "W2": h1.T @ dlogits + wd * params["W2"],
                "b2": np.sum(dlogits, axis=0),
            }
            dh1 = dlogits @ params["W2"].T
            dz1 = dh1 * (z1 > 0.0)
            grads["W1"] = xb.T @ dz1 + wd * params["W1"]
            grads["b1"] = np.sum(dz1, axis=0)

            t_adam += 1
            for k in params:
                m[k] = b1 * m[k] + (1.0 - b1) * grads[k]
                v[k] = b2 * v[k] + (1.0 - b2) * (grads[k] ** 2)
                m_hat = m[k] / (1.0 - b1**t_adam)
                v_hat = v[k] / (1.0 - b2**t_adam)
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        _, _, tr_probs = forward_mlp(X_tr, params)
        tr_ce = float(
            -np.mean(np.log(np.clip(tr_probs[np.arange(n_tr), y_tr], 1e-12, 1.0)))
        )
        tr_acc = float(np.mean(np.argmax(tr_probs, axis=1) == y_tr))

        _, _, val_probs = forward_mlp(X_val, params)
        val_ce = float(
            -np.mean(
                np.log(np.clip(val_probs[np.arange(len(y_val)), y_val], 1e-12, 1.0))
            )
        )
        val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))

        history.append(
            {
                "epoch": epoch,
                "train_ce": round(tr_ce, 6),
                "train_agreement": round(tr_acc, 6),
                "val_ce": round(val_ce, 6),
                "val_agreement": round(val_acc, 6),
            }
        )

        # Selection rule: lowest val_ce, tiebreak highest val_acc, earliest epoch
        cand_key = (val_ce, -val_acc, epoch)
        if best_key is None or cand_key < best_key:
            best_key = cand_key
            best_epoch = epoch
            best_params = {k: val.copy() for k, val in params.items()}

    train_metrics = evaluate_dataset(
        train_data["X"],
        train_data["y"],
        train_data["front_blocked"],
        train_data["target_in_view"],
        best_params,
        wd,
    )
    val_metrics = evaluate_dataset(
        val_data["X"],
        val_data["y"],
        val_data["front_blocked"],
        val_data["target_in_view"],
        best_params,
        wd,
    )

    training_report = {
        "selected_epoch": best_epoch,
        "epochs_completed": len(history),
        "stopped_by_wall_clock_cap": timed_out_wall_clock,
        "train_metrics_at_selected": train_metrics,
        "val_metrics_at_selected": val_metrics,
        "epoch_history": history,
    }
    return best_params, training_report, timed_out_wall_clock


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


def evaluate_rollouts(
    seeds: list[int],
    params: dict[str, np.ndarray],
    deadline: float,
) -> tuple[dict, bool]:
    episodes = []
    timed_out_wall_clock = False

    for seed in seeds:
        if time.monotonic() >= deadline:
            timed_out_wall_clock = True
            break

        env = gym.make("BabyAI-GoToLocal-v0")
        obs, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        exc_info = None
        steps = []

        seen_feat_hashes: set[str] = set()
        seen_raw_hashes: set[str] = set()
        seen_poses: dict[tuple[int, int, int], int] = {}

        repeated_feat_count = 0
        repeated_raw_count = 0
        repeated_pose_count = 0
        blocked_fwd_steps = 0
        cur_blocked_fwd_run = 0
        max_blocked_fwd_run = 0
        first_pose_cycle: dict | None = None

        try:
            while not (terminated or truncated):
                feat, meta = encode_observation(obs)
                feat_hash = hash_bytes(feat.tobytes())
                raw_hash = hash_bytes(obs["image"].tobytes() + obs["mission"].encode("utf-8"))

                pos_x, pos_y = int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1])
                heading = int(env.unwrapped.agent_dir)
                pose = (pos_x, pos_y, heading)

                _, _, probs = forward_mlp(feat[None, :], params)
                probs_1d = probs[0]
                action = int(np.argmax(probs_1d))

                is_feat_rep = feat_hash in seen_feat_hashes
                is_raw_rep = raw_hash in seen_raw_hashes
                is_pose_rep = pose in seen_poses

                if is_feat_rep:
                    repeated_feat_count += 1
                if is_raw_rep:
                    repeated_raw_count += 1
                if is_pose_rep:
                    repeated_pose_count += 1
                    if first_pose_cycle is None:
                        prev_step = seen_poses[pose]
                        cycle_len = len(steps) - prev_step
                        first_pose_cycle = {
                            "start_step": prev_step,
                            "repeat_step": len(steps),
                            "cycle_length": cycle_len,
                            "pose": [pos_x, pos_y, heading],
                            "cycle_actions": [
                                s["chosen_action"] for s in steps[prev_step:]
                            ],
                            "cycle_target_in_view_flags": [
                                s["target_in_view"] for s in steps[prev_step:]
                            ],
                            "cycle_front_blocked_flags": [
                                s["front_blocked"] for s in steps[prev_step:]
                            ],
                        }

                seen_feat_hashes.add(feat_hash)
                seen_raw_hashes.add(raw_hash)
                if pose not in seen_poses:
                    seen_poses[pose] = len(steps)

                is_blocked_fwd = bool(meta["front_blocked"] and action == 2)
                if is_blocked_fwd:
                    blocked_fwd_steps += 1
                    cur_blocked_fwd_run += 1
                    if cur_blocked_fwd_run > max_blocked_fwd_run:
                        max_blocked_fwd_run = cur_blocked_fwd_run
                else:
                    cur_blocked_fwd_run = 0

                steps.append(
                    {
                        "step": len(steps),
                        "feat_hash": feat_hash,
                        "raw_obs_hash": raw_hash,
                        "probs": [round(float(p), 6) for p in probs_1d],
                        "chosen_action": action,
                        "chosen_action_name": ACTION_NAMES[action],
                        "agent_pos": [pos_x, pos_y],
                        "agent_dir": heading,
                        "front_blocked": meta["front_blocked"],
                        "target_in_view": meta["target_in_view"],
                        "blocked_forward": is_blocked_fwd,
                    }
                )

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
        except Exception as exc:  # noqa: BLE001
            exc_info = f"{type(exc).__name__}: {exc}"
        finally:
            env.close()

        episodes.append(
            {
                "seed": seed,
                "mission": obs["mission"],
                "success": bool(terminated and total_reward > 0.0 and exc_info is None),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "exception": exc_info,
                "episode_length": len(steps),
                "total_reward": round(total_reward, 6),
                "repeated_feat_obs_steps": repeated_feat_count,
                "repeated_raw_obs_steps": repeated_raw_count,
                "repeated_pose_steps": repeated_pose_count,
                "blocked_forward_steps": blocked_fwd_steps,
                "max_consecutive_blocked_forward": max_blocked_fwd_run,
                "first_pose_cycle": first_pose_cycle,
                "steps": steps,
            }
        )

    n_eps = len(episodes)
    successes = sum(1 for e in episodes if e["success"])
    timeouts = sum(1 for e in episodes if e["truncated"])
    exceptions = sum(1 for e in episodes if e["exception"] is not None)
    ep_with_rep_feat = sum(1 for e in episodes if e["repeated_feat_obs_steps"] > 0)
    ep_with_rep_raw = sum(1 for e in episodes if e["repeated_raw_obs_steps"] > 0)
    ep_with_rep_pose = sum(1 for e in episodes if e["repeated_pose_steps"] > 0)
    ep_with_consec_blk_fwd = sum(
        1 for e in episodes if e["max_consecutive_blocked_forward"] >= 2
    )
    ep_with_any_blk_fwd = sum(1 for e in episodes if e["blocked_forward_steps"] > 0)

    cycle_lengths = Counter(
        e["first_pose_cycle"]["cycle_length"]
        for e in episodes
        if e["first_pose_cycle"] is not None
    )
    cycle_1 = sum(
        1
        for e in episodes
        if e["first_pose_cycle"] and e["first_pose_cycle"]["cycle_length"] == 1
    )
    cycle_2 = sum(
        1
        for e in episodes
        if e["first_pose_cycle"] and e["first_pose_cycle"]["cycle_length"] == 2
    )
    cycle_ge3 = sum(
        1
        for e in episodes
        if e["first_pose_cycle"] and e["first_pose_cycle"]["cycle_length"] >= 3
    )

    summary = {
        "num_episodes_completed": n_eps,
        "stopped_by_wall_clock_cap": timed_out_wall_clock,
        "success_count": successes,
        "timeout_count": timeouts,
        "exception_count": exceptions,
        "mean_episode_length": (
            round(sum(e["episode_length"] for e in episodes) / n_eps, 4) if n_eps else 0.0
        ),
        "episodes_with_repeated_feature_obs": ep_with_rep_feat,
        "episodes_with_repeated_raw_obs": ep_with_rep_raw,
        "episodes_with_repeated_pose": ep_with_rep_pose,
        "episodes_with_any_blocked_forward": ep_with_any_blk_fwd,
        "episodes_with_consecutive_blocked_forward_ge2": ep_with_consec_blk_fwd,
        "total_blocked_forward_steps": sum(e["blocked_forward_steps"] for e in episodes),
        "first_pose_cycle_length_1_count": cycle_1,
        "first_pose_cycle_length_2_count": cycle_2,
        "first_pose_cycle_length_ge3_count": cycle_ge3,
        "first_pose_cycle_length_histogram": {
            str(k): v for k, v in sorted(cycle_lengths.items())
        },
    }
    return {"summary": summary, "episodes": episodes}, timed_out_wall_clock


def main() -> None:
    t0 = time.monotonic()
    cfg = json.loads(Path("phase1_config.json").read_text(encoding="utf-8"))
    cap = float(cfg["limits"]["combined_wall_clock_cap_seconds"])
    deadline = t0 + cap

    # 1. Verify feature extraction assertions before data collection/training
    verify_feature_assertions()

    # 2. Collect training and validation expert demonstrations
    tr_cfg = cfg["seeds"]["train"]
    val_cfg = cfg["seeds"]["val"]
    train_seeds = list(range(tr_cfg["start"], tr_cfg["end_inclusive"] + 1))
    val_seeds = list(range(val_cfg["start"], val_cfg["end_inclusive"] + 1))

    t_col_start = time.monotonic()
    train_data, cap_tr = collect_expert_split("train", train_seeds, deadline)
    val_data, cap_val = collect_expert_split("val", val_seeds, deadline)
    t_col_sec = time.monotonic() - t_col_start

    collection_meta = {
        "train": train_data["summary"],
        "val": val_data["summary"],
        "collection_wall_clock_seconds": round(t_col_sec, 4),
    }
    Path("phase1_collection.json").write_text(
        json.dumps(collection_meta, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        "phase1_demos.npz",
        X_train=train_data["X"],
        y_train=train_data["y"],
        front_blocked_train=train_data["front_blocked"],
        target_in_view_train=train_data["target_in_view"],
        X_val=val_data["X"],
        y_val=val_data["y"],
        front_blocked_val=val_data["front_blocked"],
        target_in_view_val=val_data["target_in_view"],
    )

    total_anomalies = len(train_data["summary"]["anomalies"]) + len(
        val_data["summary"]["anomalies"]
    )
    if total_anomalies > 0 or cap_tr or cap_val:
        print(
            json.dumps(
                {
                    "status": "halted_after_collection",
                    "reason": (
                        "wall_clock_cap" if (cap_tr or cap_val) else "collection_anomalies"
                    ),
                    "collection": collection_meta,
                },
                indent=2,
            )
        )
        return

    # 3. Train single NumPy MLP baseline
    t_train_start = time.monotonic()
    best_params, train_report, cap_train = train_mlp(train_data, val_data, cfg, deadline)
    t_train_sec = time.monotonic() - t_train_start

    # Save weights and training log BEFORE rollout evaluation
    np.savez("phase1_weights.npz", **best_params)
    train_report["training_wall_clock_seconds"] = round(t_train_sec, 4)
    Path("phase1_training.json").write_text(
        json.dumps(train_report, indent=2), encoding="utf-8"
    )

    if cap_train:
        print(
            json.dumps(
                {
                    "status": "halted_after_training_cap",
                    "collection": collection_meta,
                    "training": {
                        k: v for k, v in train_report.items() if k != "epoch_history"
                    },
                },
                indent=2,
            )
        )
        return

    # 4. Evaluate frozen deterministic argmax policy on dev seeds 0..99
    dev_cfg = cfg["seeds"]["dev_eval"]
    dev_seeds = list(range(dev_cfg["start"], dev_cfg["end_inclusive"] + 1))

    t_eval_start = time.monotonic()
    eval_report, cap_eval = evaluate_rollouts(dev_seeds, best_params, deadline)
    t_eval_sec = time.monotonic() - t_eval_start
    total_sec = time.monotonic() - t0

    eval_report["summary"]["eval_wall_clock_seconds"] = round(t_eval_sec, 4)
    eval_report["summary"]["total_phase1_wall_clock_seconds"] = round(total_sec, 4)
    Path("phase1_eval_results.json").write_text(
        json.dumps(eval_report, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": "completed" if not cap_eval else "halted_during_eval_cap",
                "collection": collection_meta,
                "training_selected": {
                    k: v for k, v in train_report.items() if k != "epoch_history"
                },
                "dev_eval_summary": eval_report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
