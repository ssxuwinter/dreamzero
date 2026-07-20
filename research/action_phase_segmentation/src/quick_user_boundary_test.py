#!/usr/bin/env python3
"""Quick action-only signal check using the user-labelled episode 1925 boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_pilot_sequences import load_aligned_sequences, load_joint_range


FEATURES = (
    "joint_step",
    "joint_step_change",
    "joint_jerk",
    "direction_change",
    "gripper_value",
    "gripper_change",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=(
            "research/action_chunk_compute/experiments/h2_wam_static_compute/"
            "results/diagnostic_contiguous_12/raw_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--dataset", default="/home/admin/.cache/DreamZero-DROID-Data"
    )
    parser.add_argument(
        "--output-dir",
        default="research/action_phase_segmentation/pilot/user_boundary_1925",
    )
    parser.add_argument("--episode", type=int, default=1925)
    parser.add_argument("--schedule", type=int, default=16)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--free-end-seconds", type=float, default=3.0)
    parser.add_argument("--fine-start-seconds", type=float, default=4.0)
    return parser.parse_args()


def simple_features(
    action: np.ndarray, joint_range: np.ndarray, chunk_starts: np.ndarray
) -> dict[str, np.ndarray]:
    count = len(action)
    values = {name: np.full(count, np.nan, dtype=np.float64) for name in FEATURES}
    values["gripper_value"] = action[:, 7].astype(np.float64)

    for chunk_index, start_value in enumerate(chunk_starts):
        start = int(start_value)
        end = (
            int(chunk_starts[chunk_index + 1])
            if chunk_index + 1 < len(chunk_starts)
            else count
        )
        joint = action[start:end, :7] / joint_range[None, :]
        gripper = action[start:end, 7]
        d1 = np.diff(joint, axis=0)
        d2 = np.diff(d1, axis=0)
        d3 = np.diff(d2, axis=0)

        values["joint_step"][start + 1 : end] = np.linalg.norm(d1, axis=1)
        values["joint_step_change"][start + 2 : end] = np.linalg.norm(d2, axis=1)
        values["joint_jerk"][start + 3 : end] = np.linalg.norm(d3, axis=1)
        values["gripper_change"][start + 1 : end] = np.abs(np.diff(gripper))

        if len(d1) >= 2:
            previous = d1[:-1]
            current = d1[1:]
            denominator = np.linalg.norm(previous, axis=1) * np.linalg.norm(
                current, axis=1
            )
            direction = np.zeros(len(current), dtype=np.float64)
            valid = denominator > 1e-12
            cosine = np.ones(len(current), dtype=np.float64)
            cosine[valid] = np.sum(previous[valid] * current[valid], axis=1) / denominator[
                valid
            ]
            direction[valid] = 1.0 - np.clip(cosine[valid], -1.0, 1.0)
            values["direction_change"][start + 2 : end] = direction

    return values


def summarize(
    source: str,
    frames: np.ndarray,
    features: dict[str, np.ndarray],
    free_end: int,
    fine_start: int,
    fps: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    labels = np.full(len(frames), "transition", dtype=object)
    labels[frames < free_end] = "free"
    labels[frames >= fine_start] = "fine"

    chunk_offset = np.arange(len(frames)) % 24
    stable = chunk_offset >= 3
    rows: list[dict[str, object]] = []
    step_rows = {
        "source": np.repeat(source, len(frames)),
        "frame": frames,
        "seconds": frames / fps,
        "label": labels,
        "chunk_offset": chunk_offset,
        "used": stable & (labels != "transition"),
    }
    step_rows.update(features)

    for name, values in features.items():
        free_mask = stable & (labels == "free") & np.isfinite(values)
        fine_mask = stable & (labels == "fine") & np.isfinite(values)
        free_values = values[free_mask]
        fine_values = values[fine_mask]
        y_true = np.concatenate(
            [np.zeros(len(free_values), dtype=int), np.ones(len(fine_values), dtype=int)]
        )
        scores = np.concatenate([free_values, fine_values])
        raw_auc = float(roc_auc_score(y_true, scores))
        rows.append(
            {
                "source": source,
                "feature": name,
                "free_n": len(free_values),
                "fine_n": len(fine_values),
                "free_mean": float(np.mean(free_values)),
                "fine_mean": float(np.mean(fine_values)),
                "free_median": float(np.median(free_values)),
                "fine_median": float(np.median(fine_values)),
                "raw_auc_fine_high": raw_auc,
                "separation_auc": max(raw_auc, 1.0 - raw_auc),
                "fine_direction": "higher" if raw_auc >= 0.5 else "lower",
            }
        )
    return rows, pd.DataFrame(step_rows)


def plot_timeseries(
    episode: int,
    frame: np.ndarray,
    wam: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    free_end: int,
    fine_start: int,
    fps: float,
    output: Path,
) -> None:
    shown = (
        "joint_step",
        "joint_step_change",
        "direction_change",
        "gripper_value",
        "gripper_change",
    )
    labels = {
        "joint_step": "joint step",
        "joint_step_change": "step change",
        "direction_change": "direction change",
        "gripper_value": "gripper",
        "gripper_change": "gripper change",
    }
    seconds = frame / fps
    fig, axes = plt.subplots(len(shown), 1, figsize=(12, 10), sharex=True)
    for axis, name in zip(axes, shown):
        axis.axvspan(seconds[0], free_end / fps, color="#dbeafe", alpha=0.8)
        axis.axvspan(free_end / fps, fine_start / fps, color="#e5e7eb", alpha=0.8)
        axis.axvspan(fine_start / fps, seconds[-1], color="#fee2e2", alpha=0.5)
        axis.plot(seconds, gt[name], color="#94a3b8", linewidth=1.0, label="GT")
        axis.plot(seconds, wam[name], color="#111827", linewidth=1.3, label="WAM")
        axis.set_ylabel(labels[name])
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("video time (seconds)")
    fig.suptitle(
        f"Episode {episode}: user label (free <3s, excluded 3-4s, fine >=4s)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    free_end = int(round(args.free_end_seconds * args.fps))
    fine_start = int(round(args.fine_start_seconds * args.fps))

    aligned = load_aligned_sequences(
        Path(args.predictions), {args.episode}, args.schedule
    )[args.episode]
    joint_range = load_joint_range(Path(args.dataset))
    frame = aligned["frame"]
    all_rows: list[dict[str, object]] = []
    all_steps: list[pd.DataFrame] = []
    computed: dict[str, dict[str, np.ndarray]] = {}
    for source in ("gt", "wam"):
        computed[source] = simple_features(
            aligned[source], joint_range, aligned["chunk_starts"]
        )
        rows, steps = summarize(
            source,
            frame,
            computed[source],
            free_end,
            fine_start,
            args.fps,
        )
        all_rows.extend(rows)
        all_steps.append(steps)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(all_steps, ignore_index=True).to_csv(
        output_dir / "per_step_features.csv", index=False
    )
    plot_timeseries(
        args.episode,
        frame,
        computed["wam"],
        computed["gt"],
        free_end,
        fine_start,
        args.fps,
        output_dir / "timeseries.png",
    )
    manifest = {
        "status": "exploratory_quick_check",
        "episode": args.episode,
        "fps": args.fps,
        "wam_coverage_frames": [int(frame[0]), int(frame[-1])],
        "user_label": {
            "free": [0, free_end - 1],
            "excluded_transition": [free_end, fine_start - 1],
            "fine_within_wam_coverage": [fine_start, int(frame[-1])],
        },
        "derivatives": "Within each independent 24-step chunk only.",
        "excluded_chunk_offsets": [0, 1, 2],
        "warning": "One trajectory only; descriptive signal check, not generalization evidence.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
