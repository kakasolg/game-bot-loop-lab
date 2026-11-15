"""Phase 3 bounded expert-assisted retraining reference experiment (DAgger iter 1) + step-matched control."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions
from minigrid.utils.baby_ai_bot import BabyAIBot

from run_phase1 import (
    ACTION_NAMES,
    ALLOWED_ACTIONS,
    encode_observation,
    evaluate_dataset,
    forward_mlp,
    hash_bytes,
    verify_feature_assertions,
)
from run_phase1_continuation import evaluate_frozen_argmax

EXPECTED_PHASE1_WEIGHTS_SHA256 = (
    "e31f335039c27dc2b41a937a82d8a52e062828e15c3e049f8f470c614a6a19fd"
)
WALL_CLOCK_CAP_SEC = 300.0
FAILED_DEV_SEEDS = [3, 15, 29, 43, 50, 80, 94]
EXCLUDED_TRAIN_SEEDS = {10462}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_onpolicy_advisory(
    seeds: list[int],
    student_params: dict[str, np.ndarray],
    deadline: float,
) -> tuple[dict, bool, str | None]:
    """Roll out frozen student deterministically on training seeds with BabyAIBot.replan advising."""
    all_action_names = {int(a): a.name for a in Actions}

    features: list[np.ndarray] = []
    actions: list[int] = []
    front_blocked_flags: list[bool] = []
    target_visible_flags: list[bool] = []
    sample_seeds: list[int] = []
    sample_steps: list[int] = []
    sample_feat_hashes: list[str] = []
    sample_feat_bytes: list[bytes] = []

    excluded_outside_actions: list[dict] = []
    episode_records: list[dict] = []

    total_advisory_calls = 0
    eligible_before_dedup = 0
    student_successes = 0
    student_timeouts = 0
    error_msg: str | None = None

    for seed in seeds:
        if time.monotonic() >= deadline:
            return (
                {
                    "partial": True,
                    "episodes_completed": len(episode_records),
                    "total_advisory_expert_calls": total_advisory_calls,
                    "excluded_outside_actions": excluded_outside_actions,
                    "eligible_label_occurrences_before_dedup": eligible_before_dedup,
                    "samples_retained_after_dedup": len(actions),
                },
                True,
                None,
            )

        env = gym.make("BabyAI-GoToLocal-v0")
        obs, _ = env.reset(seed=seed)
        bot = BabyAIBot(env)
        terminated = False
        truncated = False
        total_reward = 0.0
        prev_student_action = None
        ep_len = 0
        ep_retained = 0
        ep_outside = 0
        ep_seen_pairs: set[tuple[bytes, int]] = set()

        try:
            while not (terminated or truncated):
                feat, meta = encode_observation(obs)
                feat_b = feat.tobytes()
                feat_h = hash_bytes(feat_b)

                _, _, probs = forward_mlp(feat[None, :], student_params)
                probs_1d = probs[0]
                if not np.all(np.isfinite(probs_1d)):
                    raise FloatingPointError(
                        f"Non-finite student probabilities at train seed={seed} step={ep_len}"
                    )
                student_action = int(np.argmax(probs_1d))

                expert_action_obj = bot.replan(prev_student_action)
                expert_action = int(expert_action_obj)
                total_advisory_calls += 1

                if expert_action not in ALLOWED_ACTIONS:
                    ep_outside += 1
                    excluded_outside_actions.append(
                        {
                            "seed": seed,
                            "step": ep_len,
                            "action_id": expert_action,
                            "action_name": all_action_names.get(
                                expert_action, f"unknown_{expert_action}"
                            ),
                        }
                    )
                else:
                    eligible_before_dedup += 1
                    pair_key = (feat_b, expert_action)
                    if pair_key not in ep_seen_pairs:
                        ep_seen_pairs.add(pair_key)
                        features.append(feat)
                        actions.append(expert_action)
                        front_blocked_flags.append(meta["front_blocked"])
                        target_visible_flags.append(meta["target_in_view"])
                        sample_seeds.append(seed)
                        sample_steps.append(ep_len)
                        sample_feat_hashes.append(feat_h)
                        sample_feat_bytes.append(feat_b)
                        ep_retained += 1

                obs, reward, terminated, truncated, _ = env.step(student_action)
                total_reward += float(reward)
                prev_student_action = student_action
                ep_len += 1
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Collection error at seed={seed} step={ep_len}: {type(exc).__name__}: {exc}"
        finally:
            env.close()

        if error_msg is not None:
            return (
                {
                    "partial": True,
                    "episodes_completed": len(episode_records),
                    "total_advisory_expert_calls": total_advisory_calls,
                    "excluded_outside_actions": excluded_outside_actions,
                    "eligible_label_occurrences_before_dedup": eligible_before_dedup,
                    "samples_retained_after_dedup": len(actions),
                },
                False,
                error_msg,
            )

        succ = bool(terminated and total_reward > 0.0)
        if succ:
            student_successes += 1
        elif truncated:
            student_timeouts += 1

        episode_records.append(
            {
                "seed": seed,
                "steps": ep_len,
                "success": succ,
                "truncated": bool(truncated),
                "retained_samples": ep_retained,
                "outside_action_steps": ep_outside,
            }
        )

    return (
        {
            "partial": False,
            "X": np.stack(features).astype(np.float32),
            "y": np.array(actions, dtype=np.int64),
            "front_blocked": np.array(front_blocked_flags, dtype=bool),
            "target_in_view": np.array(target_visible_flags, dtype=bool),
            "sample_seeds": np.array(sample_seeds, dtype=np.int64),
            "sample_steps": np.array(sample_steps, dtype=np.int64),
            "sample_feat_hashes": sample_feat_hashes,
            "sample_feat_bytes": sample_feat_bytes,
            "episode_records": episode_records,
            "student_rollout_summary": {
                "episodes": len(episode_records),
                "successes": student_successes,
                "timeouts": student_timeouts,
            },
            "total_advisory_expert_calls": total_advisory_calls,
            "excluded_outside_actions": excluded_outside_actions,
            "eligible_label_occurrences_before_dedup": eligible_before_dedup,
            "samples_retained_after_dedup": len(actions),
        },
        False,
        None,
    )


def compute_supervision_cost_report(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    onpolicy: dict,
    phase2_cycle_hashes: set[str],
) -> dict:
    """Compute supervision-cost and label-overlap diagnostics without altering deduplication."""
    orig_bytes = [X_orig[i].tobytes() for i in range(len(y_orig))]
    orig_hashes = [hash_bytes(b) for b in orig_bytes]
    orig_pair_set = {(orig_bytes[i], int(y_orig[i])) for i in range(len(y_orig))}

    orig_labels_by_feat: dict[bytes, set[int]] = defaultdict(set)
    for b, lbl in zip(orig_bytes, y_orig):
        orig_labels_by_feat[b].add(int(lbl))

    onp_bytes = onpolicy["sample_feat_bytes"]
    onp_y = onpolicy["y"]
    onp_hashes = onpolicy["sample_feat_hashes"]

    onp_labels_by_feat: dict[bytes, set[int]] = defaultdict(set)
    onp_counts_by_feat: dict[bytes, Counter] = defaultdict(Counter)
    already_in_orig_count = 0
    distinct_pairs_in_onpolicy: set[tuple[bytes, int]] = set()
    distinct_new_pairs: set[tuple[bytes, int]] = set()

    for b, lbl in zip(onp_bytes, onp_y):
        lbl_i = int(lbl)
        onp_labels_by_feat[b].add(lbl_i)
        onp_counts_by_feat[b][lbl_i] += 1
        pair = (b, lbl_i)
        distinct_pairs_in_onpolicy.add(pair)
        if pair in orig_pair_set:
            already_in_orig_count += 1
        else:
            distinct_new_pairs.add(pair)

    comb_labels_by_feat: dict[bytes, set[int]] = defaultdict(set)
    for b, lbls in orig_labels_by_feat.items():
        comb_labels_by_feat[b].update(lbls)
    for b, lbls in onp_labels_by_feat.items():
        comb_labels_by_feat[b].update(lbls)

    orig_conflicts = [b for b, s in orig_labels_by_feat.items() if len(s) > 1]
    onp_conflicts = [b for b, s in onp_labels_by_feat.items() if len(s) > 1]
    comb_conflicts = [b for b, s in comb_labels_by_feat.items() if len(s) > 1]

    # How many distinct Phase 2 cycle feature vectors now appear in D_onpolicy / D_combined
    onp_hash_set = set(onp_hashes)
    comb_hash_set = set(orig_hashes) | onp_hash_set
    cycle_hashes_covered_in_onpolicy = sorted(phase2_cycle_hashes & onp_hash_set)
    cycle_hashes_covered_in_combined = sorted(phase2_cycle_hashes & comb_hash_set)

    outside_counts = Counter(
        x["action_name"] for x in onpolicy["excluded_outside_actions"]
    )
    outside_episodes = sorted({x["seed"] for x in onpolicy["excluded_outside_actions"]})

    return {
        "total_advisory_expert_calls": onpolicy["total_advisory_expert_calls"],
        "calls_outside_student_action_set": len(onpolicy["excluded_outside_actions"]),
        "outside_action_breakdown": dict(outside_counts),
        "outside_action_episode_count": len(outside_episodes),
        "outside_action_episode_seeds_sample": outside_episodes[:20],
        "eligible_label_occurrences_before_dedup": onpolicy[
            "eligible_label_occurrences_before_dedup"
        ],
        "samples_retained_after_dedup": onpolicy["samples_retained_after_dedup"],
        "duplicates_removed_by_per_episode_dedup": (
            onpolicy["eligible_label_occurrences_before_dedup"]
            - onpolicy["samples_retained_after_dedup"]
        ),
        "retained_samples_already_in_original_train": already_in_orig_count,
        "retained_samples_new_feature_action_pairs": (
            onpolicy["samples_retained_after_dedup"] - already_in_orig_count
        ),
        "distinct_feature_action_pairs_in_onpolicy": len(distinct_pairs_in_onpolicy),
        "distinct_new_feature_action_pairs_added": len(distinct_new_pairs),
        "distinct_feature_vectors_in_original_train": len(orig_labels_by_feat),
        "distinct_feature_vectors_in_onpolicy": len(onp_labels_by_feat),
        "distinct_feature_vectors_in_combined": len(comb_labels_by_feat),
        "feature_vectors_with_conflicting_expert_labels": {
            "within_original_train": len(orig_conflicts),
            "within_onpolicy": len(onp_conflicts),
            "across_combined_dataset": len(comb_conflicts),
        },
        "phase2_distinct_cycle_feature_vectors_total": len(phase2_cycle_hashes),
        "phase2_cycle_feature_vectors_matched_in_onpolicy": len(
            cycle_hashes_covered_in_onpolicy
        ),
        "phase2_cycle_feature_vectors_matched_in_combined": len(
            cycle_hashes_covered_in_combined
        ),
        "onpolicy_action_distribution": {
            ACTION_NAMES[k]: int(np.sum(onp_y == k)) for k in (0, 1, 2)
        },
        "onpolicy_front_blocked_count": int(np.sum(onpolicy["front_blocked"])),
        "onpolicy_target_in_view_count": int(np.sum(onpolicy["target_in_view"])),
    }


def train_with_update_schedule(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    fb_tr: np.ndarray,
    tv_tr: np.ndarray,
    val_data: dict,
    cfg: dict,
    checkpoint_interval_updates: int,
    num_checkpoint_slots: int,
    deadline: float,
    source_mask_onpolicy: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict, bool]:
    """Train from init_seed=42 for exact U_target updates with fixed checkpoint slots."""
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

    X_val, y_val = val_data["X"], val_data["y"]
    n_tr = len(y_tr)
    bs = tr["batch_size"]
    lr = tr["learning_rate"]
    b1, b2, eps = tr["beta1"], tr["beta2"], tr["eps"]
    wd = tr["l2_weight_decay"]

    total_updates_target = num_checkpoint_slots * checkpoint_interval_updates
    batches_per_pass = math.ceil(n_tr / bs)
    shuffle_rng = np.random.default_rng(tr["shuffle_seed"])

    history = []
    best_key = None
    best_slot = 0
    best_update_step = 0
    best_params = {k: val.copy() for k, val in params.items()}
    timed_out_wall_clock = False

    while t_adam < total_updates_target:
        if time.monotonic() >= deadline:
            timed_out_wall_clock = True
            break

        perm = shuffle_rng.permutation(n_tr)
        for start in range(0, n_tr, bs):
            if t_adam >= total_updates_target:
                break
            if time.monotonic() >= deadline:
                timed_out_wall_clock = True
                break

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

            if t_adam % checkpoint_interval_updates == 0:
                slot = t_adam // checkpoint_interval_updates
                _, _, tr_probs = forward_mlp(X_tr, params)
                tr_ce = float(
                    -np.mean(
                        np.log(np.clip(tr_probs[np.arange(n_tr), y_tr], 1e-12, 1.0))
                    )
                )
                tr_acc = float(np.mean(np.argmax(tr_probs, axis=1) == y_tr))

                _, _, val_probs = forward_mlp(X_val, params)
                val_ce = float(
                    -np.mean(
                        np.log(
                            np.clip(
                                val_probs[np.arange(len(y_val)), y_val], 1e-12, 1.0
                            )
                        )
                    )
                )
                val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))

                history.append(
                    {
                        "checkpoint_slot": slot,
                        "optimizer_step": t_adam,
                        "epoch_equivalent": round(t_adam / batches_per_pass, 4),
                        "train_ce": round(tr_ce, 6),
                        "train_agreement": round(tr_acc, 6),
                        "val_ce": round(val_ce, 6),
                        "val_agreement": round(val_acc, 6),
                    }
                )

                cand_key = (val_ce, -val_acc, slot)
                if best_key is None or cand_key < best_key:
                    best_key = cand_key
                    best_slot = slot
                    best_update_step = t_adam
                    best_params = {k: val.copy() for k, val in params.items()}

        if timed_out_wall_clock:
            break

    train_metrics = evaluate_dataset(X_tr, y_tr, fb_tr, tv_tr, best_params, wd)
    val_metrics = evaluate_dataset(
        val_data["X"],
        val_data["y"],
        val_data["front_blocked"],
        val_data["target_in_view"],
        best_params,
        wd,
    )

    split_metrics = {}
    if source_mask_onpolicy is not None:
        orig_mask = ~source_mask_onpolicy
        split_metrics["original_train_subset"] = evaluate_dataset(
            X_tr[orig_mask],
            y_tr[orig_mask],
            fb_tr[orig_mask],
            tv_tr[orig_mask],
            best_params,
            wd,
        )
        split_metrics["onpolicy_train_subset"] = evaluate_dataset(
            X_tr[source_mask_onpolicy],
            y_tr[source_mask_onpolicy],
            fb_tr[source_mask_onpolicy],
            tv_tr[source_mask_onpolicy],
            best_params,
            wd,
        )

    return (
        best_params,
        {
            "actual_optimizer_updates": t_adam,
            "target_optimizer_updates": total_updates_target,
            "checkpoint_interval_updates": checkpoint_interval_updates,
            "checkpoint_slots_evaluated": len(history),
            "selected_checkpoint_slot": best_slot,
            "selected_optimizer_step": best_update_step,
            "selected_epoch_equivalent": round(
                best_update_step / batches_per_pass, 4
            ),
            "stopped_by_wall_clock_cap": timed_out_wall_clock,
            "train_metrics_at_selected": train_metrics,
            "val_metrics_at_selected": val_metrics,
            "train_subset_metrics_at_selected": split_metrics,
            "checkpoint_history": history,
        },
        timed_out_wall_clock,
    )


def summarize_eval_with_phase1_comparison(
    eval_res: dict, phase1_ep_by_seed: dict[int, dict]
) -> dict:
    """Summarize effectiveness metrics including rescued/harmed episodes and length splits."""
    episodes = eval_res["episodes"]
    succ_lengths = [e["episode_length"] for e in episodes if e["success"]]
    fail_lengths = [e["episode_length"] for e in episodes if not e["success"]]

    rescued = sorted(
        e["seed"]
        for e in episodes
        if e["success"] and not phase1_ep_by_seed[e["seed"]]["success"]
    )
    harmed = sorted(
        e["seed"]
        for e in episodes
        if not e["success"] and phase1_ep_by_seed[e["seed"]]["success"]
    )

    failed_seed_outcomes = {}
    for s in FAILED_DEV_SEEDS:
        ep = next(e for e in episodes if e["seed"] == s)
        failed_seed_outcomes[str(s)] = {
            "seed": s,
            "mission": ep["mission"],
            "success": ep["success"],
            "truncated": ep["truncated"],
            "exception": ep["exception"],
            "episode_length": ep["episode_length"],
            "blocked_forward_steps": ep["blocked_forward_steps"],
            "cycle_length": (
                ep["first_pose_cycle"]["cycle_length"]
                if ep["first_pose_cycle"] is not None
                else None
            ),
            "cycle_entry_step": (
                ep["first_pose_cycle"]["start_step"]
                if ep["first_pose_cycle"] is not None
                else None
            ),
            "cycle_repeat_step": (
                ep["first_pose_cycle"]["repeat_step"]
                if ep["first_pose_cycle"] is not None
                else None
            ),
        }

    remaining_failures = sorted(e["seed"] for e in episodes if not e["success"])

    out = dict(eval_res["summary"])
    out["mean_episode_length_successful_only"] = (
        round(sum(succ_lengths) / len(succ_lengths), 4) if succ_lengths else None
    )
    out["mean_episode_length_failed_only"] = (
        round(sum(fail_lengths) / len(fail_lengths), 4) if fail_lengths else None
    )
    out["rescued_vs_phase1_count"] = len(rescued)
    out["rescued_vs_phase1_seeds"] = rescued
    out["harmed_vs_phase1_count"] = len(harmed)
    out["harmed_vs_phase1_seeds"] = harmed
    out["remaining_failure_seeds"] = remaining_failures
    out["phase1_failed_7_seeds_outcomes"] = failed_seed_outcomes
    return out


def save_partial_and_exit(partial: dict) -> None:
    Path("phase3_results.json").write_text(
        json.dumps(partial, indent=2), encoding="utf-8"
    )
    print(json.dumps(partial, indent=2))


def main() -> None:
    t0 = time.monotonic()
    deadline = t0 + WALL_CLOCK_CAP_SEC

    verify_feature_assertions()

    weights_path = Path("phase1_weights.npz")
    weights_sha = sha256_file(weights_path)
    if weights_sha != EXPECTED_PHASE1_WEIGHTS_SHA256:
        raise RuntimeError(
            f"Unexpected phase1_weights.npz sha256: {weights_sha} != {EXPECTED_PHASE1_WEIGHTS_SHA256}"
        )

    phase1_npz = np.load(weights_path)
    phase1_params = {k: phase1_npz[k] for k in phase1_npz.files}
    cfg = json.loads(Path("phase1_config.json").read_text(encoding="utf-8"))
    saved_phase1_eval = json.loads(
        Path("phase1_eval_results.json").read_text(encoding="utf-8")
    )
    phase1_ep_by_seed = {ep["seed"]: ep for ep in saved_phase1_eval["episodes"]}
    phase1_training = json.loads(
        Path("phase1_training.json").read_text(encoding="utf-8")
    )
    phase2_audit = json.loads(
        Path("phase2_audit_results.json").read_text(encoding="utf-8")
    )
    phase2_cycle_hashes = {
        ds["feat_hash"]
        for ep in phase2_audit["question_a_and_b_cycles"].values()
        for ds in ep["distinct_cycle_states"]
    }

    demos = np.load("phase1_demos_filtered.npz")
    X_orig = demos["X_train"]
    y_orig = demos["y_train"]
    fb_orig = demos["front_blocked_train"]
    tv_orig = demos["target_in_view_train"]

    val_data = {
        "X": demos["X_val"],
        "y": demos["y_val"],
        "front_blocked": demos["front_blocked_val"],
        "target_in_view": demos["target_in_view_val"],
    }

    train_seeds = [s for s in range(10000, 10500) if s not in EXCLUDED_TRAIN_SEEDS]
    dev_seeds = list(range(0, 100))

    # 1. Collect on-policy advisory dataset D_onpolicy
    t_coll_start = time.monotonic()
    onpolicy, timed_out, err = collect_onpolicy_advisory(
        train_seeds, phase1_params, deadline
    )
    t_coll_sec = round(time.monotonic() - t_coll_start, 4)

    if timed_out or err is not None:
        save_partial_and_exit(
            {
                "status": "halted_during_collection",
                "stopped_by_wall_clock_cap": timed_out,
                "error": err,
                "collection_seconds": t_coll_sec,
                "partial_collection": onpolicy,
            }
        )
        raise SystemExit(1)

    supervision_report = compute_supervision_cost_report(
        X_orig, y_orig, onpolicy, phase2_cycle_hashes
    )

    X_comb = np.concatenate([X_orig, onpolicy["X"]], axis=0)
    y_comb = np.concatenate([y_orig, onpolicy["y"]], axis=0)
    fb_comb = np.concatenate([fb_orig, onpolicy["front_blocked"]], axis=0)
    tv_comb = np.concatenate([tv_orig, onpolicy["target_in_view"]], axis=0)
    source_mask_onpolicy = np.concatenate(
        [
            np.zeros(len(y_orig), dtype=bool),
            np.ones(len(onpolicy["y"]), dtype=bool),
        ]
    )

    np.savez(
        "phase3_demos_dagger1.npz",
        X_onpolicy=onpolicy["X"],
        y_onpolicy=onpolicy["y"],
        front_blocked_onpolicy=onpolicy["front_blocked"],
        target_in_view_onpolicy=onpolicy["target_in_view"],
        sample_seeds_onpolicy=onpolicy["sample_seeds"],
        sample_steps_onpolicy=onpolicy["sample_steps"],
        X_combined=X_comb,
        y_combined=y_comb,
        front_blocked_combined=fb_comb,
        target_in_view_combined=tv_comb,
        source_is_onpolicy=source_mask_onpolicy,
    )

    bs = cfg["training"]["batch_size"]
    max_slots = cfg["training"]["max_epochs"]  # 60
    batches_per_comb_epoch = math.ceil(len(y_comb) / bs)
    total_updates_target = max_slots * batches_per_comb_epoch

    # 2. Train step-matched original-data-only control
    t_ctrl_start = time.monotonic()
    ctrl_params, ctrl_train_report, timed_out_ctrl = train_with_update_schedule(
        X_orig,
        y_orig,
        fb_orig,
        tv_orig,
        val_data,
        cfg,
        checkpoint_interval_updates=batches_per_comb_epoch,
        num_checkpoint_slots=max_slots,
        deadline=deadline,
        source_mask_onpolicy=None,
    )
    t_ctrl_sec = round(time.monotonic() - t_ctrl_start, 4)
    np.savez("phase3_weights_control.npz", **ctrl_params)

    if timed_out_ctrl:
        save_partial_and_exit(
            {
                "status": "halted_during_control_training",
                "stopped_by_wall_clock_cap": True,
                "collection_seconds": t_coll_sec,
                "control_training_seconds": t_ctrl_sec,
                "supervision_cost_report": supervision_report,
                "control_training": ctrl_train_report,
            }
        )
        raise SystemExit(1)

    # 3. Train aggregated-data DAgger iteration 1 checkpoint
    t_dag_start = time.monotonic()
    dag_params, dag_train_report, timed_out_dag = train_with_update_schedule(
        X_comb,
        y_comb,
        fb_comb,
        tv_comb,
        val_data,
        cfg,
        checkpoint_interval_updates=batches_per_comb_epoch,
        num_checkpoint_slots=max_slots,
        deadline=deadline,
        source_mask_onpolicy=source_mask_onpolicy,
    )
    t_dag_sec = round(time.monotonic() - t_dag_start, 4)
    np.savez("phase3_weights_dagger.npz", **dag_params)

    if timed_out_dag:
        save_partial_and_exit(
            {
                "status": "halted_during_dagger_training",
                "stopped_by_wall_clock_cap": True,
                "collection_seconds": t_coll_sec,
                "control_training_seconds": t_ctrl_sec,
                "dagger_training_seconds": t_dag_sec,
                "supervision_cost_report": supervision_report,
                "control_training": ctrl_train_report,
                "dagger_training": dag_train_report,
            }
        )
        raise SystemExit(1)

    # 4. Evaluate all three policies on dev 0..99 (and verify Phase 1 replay match)
    t_eval_start = time.monotonic()
    p1_eval, p1_timeout = evaluate_frozen_argmax(dev_seeds, phase1_params, deadline)
    ctrl_eval, ctrl_timeout = evaluate_frozen_argmax(dev_seeds, ctrl_params, deadline)
    dag_eval, dag_timeout = evaluate_frozen_argmax(dev_seeds, dag_params, deadline)
    t_eval_sec = round(time.monotonic() - t_eval_start, 4)

    if p1_timeout or ctrl_timeout or dag_timeout:
        save_partial_and_exit(
            {
                "status": "halted_during_evaluation",
                "stopped_by_wall_clock_cap": True,
                "collection_seconds": t_coll_sec,
                "control_training_seconds": t_ctrl_sec,
                "dagger_training_seconds": t_dag_sec,
                "eval_seconds": t_eval_sec,
            }
        )
        raise SystemExit(1)

    # Verify Phase 1 re-evaluation matches saved Phase 1 results identically
    p1_exact_match = (
        p1_eval["summary"] == saved_phase1_eval["summary"]
        and all(
            e["success"] == phase1_ep_by_seed[e["seed"]]["success"]
            and e["episode_length"] == phase1_ep_by_seed[e["seed"]]["episode_length"]
            for e in p1_eval["episodes"]
        )
    )

    p1_summary = summarize_eval_with_phase1_comparison(p1_eval, phase1_ep_by_seed)
    ctrl_summary = summarize_eval_with_phase1_comparison(ctrl_eval, phase1_ep_by_seed)
    dag_summary = summarize_eval_with_phase1_comparison(dag_eval, phase1_ep_by_seed)

    # Common solved seeds across all 3 policies for fair step-length comparison
    common_success_seeds = sorted(
        e["seed"]
        for e, ec, ed in zip(
            p1_eval["episodes"], ctrl_eval["episodes"], dag_eval["episodes"]
        )
        if e["success"] and ec["success"] and ed["success"]
    )
    common_set = set(common_success_seeds)
    common_lengths = {
        "count": len(common_success_seeds),
        "phase1_mean_length": round(
            sum(e["episode_length"] for e in p1_eval["episodes"] if e["seed"] in common_set)
            / len(common_set),
            4,
        ),
        "control_mean_length": round(
            sum(e["episode_length"] for e in ctrl_eval["episodes"] if e["seed"] in common_set)
            / len(common_set),
            4,
        ),
        "dagger_mean_length": round(
            sum(e["episode_length"] for e in dag_eval["episodes"] if e["seed"] in common_set)
            / len(common_set),
            4,
        ),
    }

    total_sec = round(time.monotonic() - t0, 4)

    results = {
        "status": "completed",
        "baseline_verification": {
            "phase1_weights_sha256": weights_sha,
            "phase1_dev_reeval_matches_saved": p1_exact_match,
            "phase1_selected_epoch": phase1_training["selected_epoch"],
            "phase1_actual_optimizer_updates": phase1_training["selected_epoch"] * 0 + 60 * math.ceil(len(y_orig) / bs),
            "phase1_selected_optimizer_step": phase1_training["selected_epoch"] * math.ceil(len(y_orig) / bs),
        },
        "schedule_parameters": {
            "original_train_samples": int(len(y_orig)),
            "onpolicy_retained_samples": int(len(onpolicy["y"])),
            "combined_train_samples": int(len(y_comb)),
            "batch_size": bs,
            "batches_per_combined_epoch_B_comb": batches_per_comb_epoch,
            "checkpoint_slots": max_slots,
            "total_optimizer_updates_U_target": total_updates_target,
        },
        "onpolicy_student_rollout_summary": onpolicy["student_rollout_summary"],
        "supervision_cost_report": supervision_report,
        "excluded_outside_action_samples": onpolicy["excluded_outside_actions"],
        "training_reports": {
            "step_matched_control": {
                k: v for k, v in ctrl_train_report.items() if k != "checkpoint_history"
            },
            "aggregated_dagger": {
                k: v for k, v in dag_train_report.items() if k != "checkpoint_history"
            },
        },
        "dev_evaluation_summaries": {
            "phase1_checkpoint": p1_summary,
            "step_matched_control": ctrl_summary,
            "aggregated_dagger_checkpoint": dag_summary,
            "common_solved_subset_lengths": common_lengths,
        },
        "runtime": {
            "cpu_only": True,
            "wall_clock_cap_seconds": WALL_CLOCK_CAP_SEC,
            "collection_seconds": t_coll_sec,
            "control_training_seconds": t_ctrl_sec,
            "dagger_training_seconds": t_dag_sec,
            "eval_seconds": t_eval_sec,
            "total_seconds": total_sec,
            "completed_within_cap": total_sec < WALL_CLOCK_CAP_SEC,
        },
    }

    Path("phase3_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
