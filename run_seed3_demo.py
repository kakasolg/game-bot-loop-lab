"""Capture two real-frame illustrative episodes on dev seed 3 (Baseline vs Wrapped).

STRICT SCOPE:
- Executes ONLY dev seed 3 in BabyAI-GoToLocal-v0 (max_steps = 64).
- Runs Episode 1 (Baseline) first, verifies exact agreement with Phase 2/4 records,
  and stops without running Episode 2 if Episode 1 diverges or errors.
- Runs Episode 2 (Wrapped) only if Episode 1 verifies cleanly.
- Saves comparison GIF, keyframe PNG strip, and metadata under demo/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import minigrid
import numpy as np
import pygame

from external_wrapper import ExternalObservationCountWrapper
from run_phase1 import ACTION_NAMES, encode_observation, forward_mlp

REQUIRED_BANNER = "Illustrative development-set example: seed 3. Not a confirmation-set episode."
DEMO_SEED = 3
OUTPUT_DIR = Path("demo")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_dependencies() -> dict:
    """Inspect runtime and rendering libraries before creating any environment."""
    mods = {
        "gymnasium": importlib.util.find_spec("gymnasium") is not None,
        "minigrid": importlib.util.find_spec("minigrid") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "pygame": importlib.util.find_spec("pygame") is not None,
        "PIL": importlib.util.find_spec("PIL") is not None,
        "imageio": importlib.util.find_spec("imageio") is not None,
        "matplotlib": importlib.util.find_spec("matplotlib") is not None,
        "cv2": importlib.util.find_spec("cv2") is not None,
    }
    pygame.init()
    pygame.font.init()
    probe_surface = pygame.Surface((16, 16))
    probe_surface.fill((10, 20, 30))
    buf = io.BytesIO()
    pygame.image.save(probe_surface, buf, "probe.png")
    png_ok = buf.getvalue().startswith(b"\x89PNG\r\n\x1a\n")

    test_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    test_rgb[:, :8] = (220, 40, 40)
    test_rgb[:, 8:] = (40, 180, 90)
    gif_bytes = encode_gif89a([test_rgb, test_rgb[:, ::-1]], [50, 50])
    loaded = pygame.image.load(io.BytesIO(gif_bytes), "probe.gif")
    gif_ok = loaded.get_size() == (16, 16)

    return {
        "modules": mods,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "gymnasium": gym.__version__,
            "minigrid": minigrid.__version__,
            "pygame": pygame.ver,
        },
        "png_offscreen_ok": png_ok,
        "gif89a_pure_python_ok": gif_ok,
        "missing_optional_packages": [k for k in ("PIL", "imageio", "matplotlib", "cv2") if not mods[k]],
        "required_ready": bool(mods["gymnasium"] and mods["minigrid"] and mods["numpy"] and mods["pygame"] and png_ok and gif_ok),
    }


def _lzw_compress_gif(indices: np.ndarray, min_code_size: int) -> list[bytes]:
    """Encode 2D uint8 palette indices using standard GIF variable-width LSB LZW."""
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    next_code = clear_code + 2
    max_table = 4096

    table: dict[tuple[int, int], int] = {}
    bit_buf = 0
    bit_count = 0
    out_bytes = bytearray()

    def emit(code: int) -> None:
        nonlocal bit_buf, bit_count
        bit_buf |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            out_bytes.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8

    emit(clear_code)
    flat = indices.reshape(-1).tolist()
    if not flat:
        emit(end_code)
        if bit_count > 0:
            out_bytes.append(bit_buf & 0xFF)
        return [bytes(out_bytes)]

    prefix = flat[0]
    for k in flat[1:]:
        pair = (prefix, k)
        if pair in table:
            prefix = table[pair]
        else:
            emit(prefix)
            if next_code < max_table:
                table[pair] = next_code
                next_code += 1
                if next_code > (1 << code_size) and code_size < 12:
                    code_size += 1
            else:
                emit(clear_code)
                table.clear()
                code_size = min_code_size + 1
                next_code = clear_code + 2
            prefix = k

    emit(prefix)
    emit(end_code)
    if bit_count > 0:
        out_bytes.append(bit_buf & 0xFF)

    data = bytes(out_bytes)
    return [data[i : i + 255] for i in range(0, len(data), 255)]


def _quantize_for_gif(frames: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build exact global palette if <=256 colors exist, else uniform 6x7x6 palette."""
    h, w, _ = frames[0].shape
    stacked = np.stack(frames, axis=0)
    flat = stacked.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2].astype(np.uint32)
    uniq, inv = np.unique(packed, return_inverse=True)

    if len(uniq) <= 256:
        palette_rgb = np.stack(
            [(uniq >> 16) & 0xFF, (uniq >> 8) & 0xFF, uniq & 0xFF], axis=1
        ).astype(np.uint8)
        idx_frames = inv.reshape(len(frames), h, w).astype(np.uint8)
        return palette_rgb, [idx_frames[i] for i in range(len(frames))]

    # Uniform 6x7x6 (252-color) fallback if anti-aliased shades exceed 256 colors
    r = np.linspace(0, 255, 6, dtype=np.uint8)
    g = np.linspace(0, 255, 7, dtype=np.uint8)
    b = np.linspace(0, 255, 6, dtype=np.uint8)
    rr, gg, bb = np.meshgrid(r, g, b, indexing="ij")
    palette_rgb = np.stack([rr.ravel(), gg.ravel(), bb.ravel()], axis=1)
    idx_frames = []
    for f in frames:
        ri = np.round(f[:, :, 0] * 5.0 / 255.0).astype(np.uint8)
        gi = np.round(f[:, :, 1] * 6.0 / 255.0).astype(np.uint8)
        bi = np.round(f[:, :, 2] * 5.0 / 255.0).astype(np.uint8)
        idx_frames.append((ri * 42 + gi * 6 + bi).astype(np.uint8))
    return palette_rgb, idx_frames


def encode_gif89a(frames: list[np.ndarray], delays_cs: list[int]) -> bytes:
    """Encode RGB uint8 frames (H, W, 3) into a looping animated GIF89a byte string."""
    if len(frames) != len(delays_cs) or not frames:
        raise ValueError("frames and delays_cs must have equal non-zero length.")

    h, w, _ = frames[0].shape
    palette_rgb, idx_frames = _quantize_for_gif(frames)

    n_colors = len(palette_rgb)
    color_bits = 2
    while (1 << color_bits) < n_colors:
        color_bits += 1
    table_size = 1 << color_bits
    padded_palette = np.zeros((table_size, 3), dtype=np.uint8)
    padded_palette[:n_colors] = palette_rgb

    out = bytearray()
    out.extend(b"GIF89a")
    gct_flag = 0x80 | ((color_bits - 1) << 4) | (color_bits - 1)
    out.extend(struct.pack("<HHBBB", w, h, gct_flag, 0, 0))
    out.extend(padded_palette.tobytes())

    # Netscape 2.0 loop extension (infinite loop: 0)
    out.extend(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")

    min_code_size = max(2, color_bits)
    for idx_img, delay_cs in zip(idx_frames, delays_cs):
        out.extend(b"\x21\xF9\x04\x04")
        out.extend(struct.pack("<HB", max(1, int(delay_cs)), 0))
        out.append(0)
        out.extend(b"\x2C")
        out.extend(struct.pack("<HHHHB", 0, 0, w, h, 0))
        out.append(min_code_size)
        for block in _lzw_compress_gif(idx_img, min_code_size):
            out.append(len(block))
            out.extend(block)
        out.append(0)

    out.append(0x3B)
    return bytes(out)


def _lzw_decode_gif(data: bytes, min_code_size: int, expected_pixels: int) -> np.ndarray:
    """Decode standard GIF LZW image data back into uint8 palette indices."""
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    next_code = clear_code + 2
    max_table = 4096

    dictionary: list[list[int]] = [[i] for i in range(clear_code)] + [[], []]
    bit_pos = 0
    total_bits = len(data) * 8

    def next_sym() -> int:
        nonlocal bit_pos
        val = 0
        for b in range(code_size):
            if bit_pos >= total_bits:
                return end_code
            byte_i = bit_pos >> 3
            bit_i = bit_pos & 7
            if (data[byte_i] >> bit_i) & 1:
                val |= 1 << b
            bit_pos += 1
        return val

    out: list[int] = []
    prev_entry: list[int] | None = None
    while len(out) < expected_pixels:
        code = next_sym()
        if code == clear_code:
            code_size = min_code_size + 1
            next_code = clear_code + 2
            dictionary = [[i] for i in range(clear_code)] + [[], []]
            prev_entry = None
            continue
        if code == end_code:
            break
        if code < len(dictionary):
            entry = dictionary[code]
        elif code == len(dictionary) and prev_entry is not None:
            entry = prev_entry + [prev_entry[0]]
        else:
            raise ValueError(f"Invalid GIF LZW code {code}")
        out.extend(entry)
        if prev_entry is not None and next_code < max_table:
            dictionary.append(prev_entry + [entry[0]])
            next_code += 1
            if next_code >= (1 << code_size) and code_size < 12:
                code_size += 1
        prev_entry = entry
    return np.array(out[:expected_pixels], dtype=np.uint8)


def decode_gif89a_frames(gif_bytes: bytes) -> tuple[list[np.ndarray], list[int]]:
    """Decode our generated GIF89a bytes back into lossless RGB uint8 frames and delays."""
    pos = 6
    w, h, flags, _, _ = struct.unpack("<HHBBB", gif_bytes[pos : pos + 7])
    pos += 7
    gct_colors = 1 << ((flags & 7) + 1)
    palette = np.frombuffer(gif_bytes[pos : pos + 3 * gct_colors], dtype=np.uint8).reshape(gct_colors, 3)
    pos += 3 * gct_colors

    frames: list[np.ndarray] = []
    delays: list[int] = []
    cur_delay = 50

    while pos < len(gif_bytes):
        blk = gif_bytes[pos]
        pos += 1
        if blk == 0x3B:
            break
        if blk == 0x21:
            label = gif_bytes[pos]
            pos += 1
            if label == 0xF9:
                sz = gif_bytes[pos]
                cur_delay = struct.unpack("<H", gif_bytes[pos + 2 : pos + 4])[0]
                pos += 1 + sz + 1  # sub-block + terminator
            else:
                while pos < len(gif_bytes):
                    sz = gif_bytes[pos]
                    pos += 1
                    if sz == 0:
                        break
                    pos += sz
        elif blk == 0x2C:
            _, _, fw, fh, _ = struct.unpack("<HHHHB", gif_bytes[pos : pos + 9])
            pos += 9
            min_code = gif_bytes[pos]
            pos += 1
            lzw_chunks = bytearray()
            while True:
                sz = gif_bytes[pos]
                pos += 1
                if sz == 0:
                    break
                lzw_chunks.extend(gif_bytes[pos : pos + sz])
                pos += sz
            idx = _lzw_decode_gif(bytes(lzw_chunks), min_code, fw * fh).reshape(fh, fw)
            frames.append(palette[idx].copy())
            delays.append(cur_delay)
        else:
            raise ValueError(f"Unexpected GIF block {blk:#x}")
    return frames, delays


def _surface_to_rgb_array(surf: pygame.Surface) -> np.ndarray:
    arr = pygame.surfarray.array3d(surf)
    return np.transpose(arr, (1, 0, 2)).copy()


def _rgb_array_to_surface(rgb: np.ndarray) -> pygame.Surface:
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def render_comparison_panel(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    left_title: str,
    right_title: str,
    left_caption: str,
    right_caption: str,
    subtitle: str,
    highlight_right: bool = False,
) -> np.ndarray:
    """Compose a crisp wide side-by-side comparison frame with non-clipping text."""
    fh, fw, _ = left_frame.shape
    pad = 16
    col_gap = 24
    col_w = max(fw, 348)
    top_h = 54
    label_h = 26
    cap_h = 34
    total_w = pad * 2 + col_gap + col_w * 2
    total_h = top_h + label_h + fh + cap_h + pad

    surf = pygame.Surface((total_w, total_h))
    surf.fill((22, 24, 28))

    font_banner = pygame.font.SysFont("dejavusansmono,consolas,monospace", 14, bold=True)
    font_sub = pygame.font.SysFont("dejavusansmono,consolas,monospace", 12)
    font_head = pygame.font.SysFont("dejavusansmono,consolas,monospace", 13, bold=True)
    font_cap = pygame.font.SysFont("dejavusansmono,consolas,monospace", 11)

    # Top required banner
    pygame.draw.rect(surf, (38, 45, 58), (0, 0, total_w, top_h))
    b_txt = font_banner.render(REQUIRED_BANNER, False, (255, 224, 110))
    surf.blit(b_txt, (pad, 7))
    s_txt = font_sub.render(subtitle, False, (215, 222, 232))
    surf.blit(s_txt, (pad, 30))

    col_lx = pad
    col_rx = pad + col_w + col_gap
    lx = col_lx + (col_w - fw) // 2
    rx = col_rx + (col_w - fw) // 2
    img_y = top_h + label_h

    lt_txt = font_head.render(left_title, False, (245, 130, 130))
    rt_txt = font_head.render(right_title, False, (130, 235, 165))
    surf.blit(lt_txt, (col_lx, top_h + 5))
    surf.blit(rt_txt, (col_rx, top_h + 5))

    surf.blit(_rgb_array_to_surface(left_frame), (lx, img_y))
    surf.blit(_rgb_array_to_surface(right_frame), (rx, img_y))

    if highlight_right:
        pygame.draw.rect(surf, (255, 210, 60), (rx - 2, img_y - 2, fw + 4, fh + 4), 2)

    cap_y = img_y + fh + 8
    lc_txt = font_cap.render(left_caption, False, (220, 220, 225))
    rc_color = (255, 220, 90) if highlight_right else (220, 220, 225)
    rc_txt = font_cap.render(right_caption, False, rc_color)
    surf.blit(lc_txt, (col_lx, cap_y))
    surf.blit(rc_txt, (col_rx, cap_y))

    return _surface_to_rgb_array(surf)


def run_seed3_episode(
    params: dict[str, np.ndarray],
    wrapper: ExternalObservationCountWrapper | None,
) -> dict:
    """Execute one BabyAI-GoToLocal-v0 episode on dev seed 3 with rgb_array capture."""
    env = gym.make("BabyAI-GoToLocal-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=DEMO_SEED)
    max_steps = int(env.unwrapped.max_steps)
    if max_steps != 64:
        env.close()
        raise RuntimeError(f"Expected max_steps=64, observed {max_steps}")

    if wrapper is not None:
        wrapper.reset()

    frames: list[np.ndarray] = [env.render().copy()]
    steps: list[dict] = []
    terminated = False
    truncated = False
    total_reward = 0.0
    blocked_forward_steps = 0
    override_steps = 0
    exc_info = None
    mission = str(obs["mission"])

    try:
        step_idx = 0
        while not (terminated or truncated):
            feat, meta = encode_observation(obs)
            pose = [
                int(env.unwrapped.agent_pos[0]),
                int(env.unwrapped.agent_pos[1]),
                int(env.unwrapped.agent_dir),
            ]
            _, _, probs = forward_mlp(feat[None, :], params)
            probs_1d = probs[0]
            base_action = int(np.argmax(probs_1d))

            if wrapper is None:
                chosen_action = base_action
                overridden = False
                prior_counts = [0, 0, 0]
            else:
                chosen_action, w_meta = wrapper.select_action(
                    obs["image"], obs["mission"], probs_1d
                )
                overridden = bool(w_meta["overridden"])
                prior_counts = [int(c) for c in w_meta["prior_counts"]]
                if overridden:
                    override_steps += 1

            if meta["front_blocked"] and chosen_action == 2:
                blocked_forward_steps += 1

            obs, reward, terminated, truncated, _ = env.step(chosen_action)
            total_reward += float(reward)
            frames.append(env.render().copy())

            steps.append(
                {
                    "step_index_0based": step_idx,
                    "step_number_1based": step_idx + 1,
                    "pose": pose,
                    "base_action": ACTION_NAMES[base_action],
                    "chosen_action": ACTION_NAMES[chosen_action],
                    "overridden": overridden,
                    "prior_counts": prior_counts,
                    "front_blocked": bool(meta["front_blocked"]),
                    "target_in_view": bool(meta["target_in_view"]),
                    "probs": [round(float(x), 6) for x in probs_1d],
                }
            )
            step_idx += 1
    except Exception as exc:  # noqa: BLE001
        exc_info = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    success = bool(terminated and total_reward > 0.0 and exc_info is None)
    outcome = "exception" if exc_info else ("success" if success else ("timeout" if truncated else "other"))
    return {
        "seed": DEMO_SEED,
        "mission": mission,
        "max_steps": max_steps,
        "outcome": outcome,
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "exception": exc_info,
        "episode_length": len(steps),
        "total_reward": round(total_reward, 6),
        "blocked_forward_steps": blocked_forward_steps,
        "override_steps": override_steps,
        "steps": steps,
        "frames": frames,
    }


def compose_and_save_media(
    base_frames: list[np.ndarray],
    wrap_frames: list[np.ndarray],
    base_steps: list[dict],
    wrap_steps: list[dict],
    mission: str,
) -> tuple[Path, Path]:
    """Compose side-by-side comparison GIF and 4-keyframe PNG strip."""
    subtitle = f"Mission: \"{mission}\" | BabyAI-GoToLocal-v0 (max_steps=64)"
    gif_frames: list[np.ndarray] = []
    gif_delays_cs: list[int] = []

    gif_frames.append(
        render_comparison_panel(
            base_frames[0],
            wrap_frames[0],
            "Baseline (frozen argmax)",
            "Wrapped (+ExternalObservationCount)",
            "Before step 0 | pose [1,1,0]",
            "Before step 0 | pose [1,1,0]",
            subtitle,
        )
    )
    gif_delays_cs.append(90)

    for k in range(11):
        bs = base_steps[k] if k < len(base_steps) else None
        ws = wrap_steps[k]
        if ws["overridden"]:
            rcap = f"Step {k} OVERRIDE: prior={ws['prior_counts']} {ws['base_action']}->{ws['chosen_action']}"
            delay = 170
        elif k == 10:
            rcap = f"Step 10: {ws['chosen_action']} -> SUCCESS (11 steps)"
            delay = 160
        else:
            rcap = f"Step {k}: {ws['chosen_action']} | pose {ws['pose']}"
            delay = 70

        if k < 2 and bs is not None:
            lcap = f"Step {k}: {bs['chosen_action']} | pose {bs['pose']}"
        else:
            lcap = f"Step {k}: blocked fwd loop ([2,1,1])"

        gif_frames.append(
            render_comparison_panel(
                base_frames[k + 1],
                wrap_frames[k + 1],
                "Baseline (frozen argmax)",
                "Wrapped (+ExternalObservationCount)",
                lcap,
                rcap,
                subtitle,
                highlight_right=ws["overridden"] or (k == 10),
            )
        )
        gif_delays_cs.append(delay)

    gif_frames.append(
        render_comparison_panel(
            base_frames[-1],
            wrap_frames[-1],
            "Baseline (frozen argmax)",
            "Wrapped (+ExternalObservationCount)",
            "Step 63 end (11..63 loop) -> TIMEOUT @ 64",
            "Completed in 11 steps (1 override at step 3)",
            subtitle,
            highlight_right=True,
        )
    )
    gif_delays_cs.append(220)

    gif_path = OUTPUT_DIR / "seed3_baseline_vs_wrapper.gif"
    gif_path.write_bytes(encode_gif89a(gif_frames, gif_delays_cs))

    kf_panels = [gif_frames[0], gif_frames[3], gif_frames[4], gif_frames[-1]]
    ph, pw, _ = kf_panels[0].shape
    grid_h = ph * 4 + 12
    grid_surf = pygame.Surface((pw, grid_h))
    grid_surf.fill((12, 14, 18))
    for i, pan in enumerate(kf_panels):
        grid_surf.blit(_rgb_array_to_surface(pan), (0, i * (ph + 4)))
    kf_path = OUTPUT_DIR / "seed3_keyframes.png"
    pygame.image.save(grid_surf, str(kf_path))
    return gif_path, kf_path


def recomposite_from_existing_gif() -> int:
    """Re-layout banners/captions from the already-captured lossless GIF frames (0 episodes)."""
    gif_path = OUTPUT_DIR / "seed3_baseline_vs_wrapper.gif"
    meta_path = OUTPUT_DIR / "seed3_demo_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    old_frames, _ = decode_gif89a_frames(gif_path.read_bytes())
    if len(old_frames) != 13 or old_frames[0].shape[:2] != (386, 560):
        raise RuntimeError(f"Unexpected captured GIF frame count/shape: {len(old_frames)}, {old_frames[0].shape}")

    # Exact 256x256 MiniGrid pixel regions from initial composite layout (pad=16, top=54, label=24)
    base_frames = [f[78:334, 16:272].copy() for f in old_frames]
    wrap_frames = [f[78:334, 288:544].copy() for f in old_frames]
    new_gif, new_kf = compose_and_save_media(
        base_frames,
        wrap_frames,
        meta["baseline"]["first_5_steps"],
        meta["wrapped"]["steps"],
        meta["wrapped"].get("mission", "go to the red key"),
    )
    meta["artifacts"] = {
        "gif": str(new_gif).replace("\\", "/"),
        "gif_bytes": new_gif.stat().st_size,
        "gif_sha256": sha256_file(new_gif),
        "keyframes_png": str(new_kf).replace("\\", "/"),
        "keyframes_png_bytes": new_kf.stat().st_size,
        "keyframes_png_sha256": sha256_file(new_kf),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"episodes_executed_for_recomposite": 0, "total_demo_episodes_recorded": meta["episodes_executed"], **meta["artifacts"]}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate illustrative dev-seed-3 comparison demo.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify dependencies, hashes, and encoders without running any episode.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the 2 authorized dev-seed-3 episodes and write demo artifacts.",
    )
    parser.add_argument(
        "--recomposite-from-gif",
        action="store_true",
        help="Re-compose visual layout from already captured lossless GIF frames (0 episodes).",
    )
    args = parser.parse_args(argv)
    if sum([args.check_only, args.run, args.recomposite_from_gif]) != 1:
        parser.error("Specify exactly one of --check-only, --run, or --recomposite-from-gif.")

    dep_report = check_dependencies()
    if args.recomposite_from_gif:
        if not dep_report["required_ready"]:
            return 1
        return recomposite_from_existing_gif()
    plan = json.loads(Path("phase5_confirmation_plan.json").read_text(encoding="utf-8"))
    weights_path = Path("phase1_weights.npz")
    wrapper_path = Path("external_wrapper.py")
    hash_ok = (
        sha256_file(weights_path) == plan["locked_artifacts_sha256"]["phase1_weights.npz"]
        and sha256_file(wrapper_path) == plan["locked_artifacts_sha256"]["external_wrapper.py"]
    )
    if not dep_report["required_ready"] or not hash_ok:
        print(json.dumps({"dependencies": dep_report, "locked_hashes_ok": hash_ok}, indent=2))
        return 1

    if args.check_only:
        print(json.dumps({"dependencies": dep_report, "locked_hashes_ok": hash_ok, "episodes_executed": 0}, indent=2))
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p1_npz = np.load(weights_path)
    p1_params = {k: p1_npz[k] for k in p1_npz.files}

    # Episode 1: Frozen Baseline on dev seed 3
    base_ep = run_seed3_episode(p1_params, None)
    episodes_executed = 1

    phase2_saved = json.loads(Path("phase2_audit_results.json").read_text(encoding="utf-8"))["question_a_and_b_cycles"]["3"]
    base_expected_ok = (
        base_ep["exception"] is None
        and base_ep["outcome"] == "timeout"
        and base_ep["episode_length"] == 64
        and base_ep["blocked_forward_steps"] == 62
        and base_ep["steps"][0]["pose"] == [1, 1, 0]
        and base_ep["steps"][0]["chosen_action"] == "forward"
        and base_ep["steps"][1]["pose"] == [2, 1, 0]
        and base_ep["steps"][1]["chosen_action"] == "right"
        and all(
            s["pose"] == [2, 1, 1] and s["chosen_action"] == "forward"
            for s in base_ep["steps"][2:]
        )
    )

    if not base_expected_ok:
        partial_meta = {
            "status": "halted_baseline_divergence",
            "episodes_executed": episodes_executed,
            "baseline_summary": {k: v for k, v in base_ep.items() if k != "frames"},
            "expected_phase2_cycle_length": phase2_saved["cycle_length"],
        }
        (OUTPUT_DIR / "seed3_baseline_halted_metadata.json").write_text(
            json.dumps(partial_meta, indent=2), encoding="utf-8"
        )
        print("Baseline episode 1 diverged from saved Phase 2 records; halting without running episode 2.")
        return 1

    # Episode 2: Frozen Policy + ExternalObservationCountWrapper on dev seed 3
    wrap_ep = run_seed3_episode(p1_params, ExternalObservationCountWrapper())
    episodes_executed = 2

    phase4_saved = json.loads(Path("phase4_results.json").read_text(encoding="utf-8"))["focus_seeds_outcomes_and_traces"]["3"]
    saved_wrap_steps = phase4_saved["wrapped_first_16_steps"]
    wrap_steps_match = (
        len(wrap_ep["steps"]) == len(saved_wrap_steps)
        and all(
            cur["step_index_0based"] == ref["step"]
            and cur["pose"] == ref["pose"]
            and cur["base_action"] == ref["base"]
            and cur["chosen_action"] == ref["chosen"]
            and cur["overridden"] == ref["overridden"]
            and cur["prior_counts"] == ref["prior_counts"]
            for cur, ref in zip(wrap_ep["steps"], saved_wrap_steps)
        )
    )
    override_indices_0based = [s["step_index_0based"] for s in wrap_ep["steps"] if s["overridden"]]
    wrap_expected_ok = (
        wrap_ep["exception"] is None
        and wrap_ep["outcome"] == "success"
        and wrap_ep["episode_length"] == 11
        and wrap_ep["override_steps"] == 1
        and override_indices_0based == [3]
        and wrap_steps_match
    )

    # Check pixel stability of Baseline loop across steps 2..64
    base_loop_frames_identical = all(
        np.array_equal(base_ep["frames"][2], f) for f in base_ep["frames"][3:]
    )

    # Compose comparison media artifacts via shared helper
    gif_path, kf_path = compose_and_save_media(
        [*base_ep["frames"][:12], base_ep["frames"][-1]],
        [*wrap_ep["frames"][:12], wrap_ep["frames"][-1]],
        base_ep["steps"],
        wrap_ep["steps"],
        base_ep["mission"],
    )

    meta = {
        "label": REQUIRED_BANNER,
        "seed": DEMO_SEED,
        "env_id": "BabyAI-GoToLocal-v0",
        "max_steps": 64,
        "episodes_executed": episodes_executed,
        "indexing_convention": "0-indexed decision steps (0..episode_length-1); frame 0 is pre-step-0 reset render and frame t+1 is post-step-t render.",
        "dependencies": dep_report,
        "baseline": {
            "outcome": base_ep["outcome"],
            "success": base_ep["success"],
            "episode_length": base_ep["episode_length"],
            "blocked_forward_steps": base_ep["blocked_forward_steps"],
            "matches_phase2_record": base_expected_ok,
            "loop_frames_2_to_64_pixel_identical": base_loop_frames_identical,
            "first_5_steps": base_ep["steps"][:5],
        },
        "wrapped": {
            "outcome": wrap_ep["outcome"],
            "success": wrap_ep["success"],
            "episode_length": wrap_ep["episode_length"],
            "override_steps": wrap_ep["override_steps"],
            "override_step_indices_0based": override_indices_0based,
            "override_step_indices_1based": [i + 1 for i in override_indices_0based],
            "matches_phase4_record": wrap_expected_ok,
            "steps": wrap_ep["steps"],
        },
        "artifacts": {
            "gif": str(gif_path).replace("\\", "/"),
            "gif_bytes": gif_path.stat().st_size,
            "gif_sha256": sha256_file(gif_path),
            "keyframes_png": str(kf_path).replace("\\", "/"),
            "keyframes_png_bytes": kf_path.stat().st_size,
            "keyframes_png_sha256": sha256_file(kf_path),
        },
    }
    meta_path = OUTPUT_DIR / "seed3_demo_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "episodes_executed": episodes_executed,
                "baseline_outcome": base_ep["outcome"],
                "baseline_length": base_ep["episode_length"],
                "baseline_matches_saved": base_expected_ok,
                "wrapped_outcome": wrap_ep["outcome"],
                "wrapped_length": wrap_ep["episode_length"],
                "wrapped_overrides": wrap_ep["override_steps"],
                "override_step_0based": override_indices_0based,
                "wrapped_matches_saved": wrap_expected_ok,
                "gif_bytes": gif_path.stat().st_size,
                "keyframes_png_bytes": kf_path.stat().st_size,
            },
            indent=2,
        )
    )
    return 0 if (base_expected_ok and wrap_expected_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
