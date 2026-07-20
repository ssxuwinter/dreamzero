#!/usr/bin/env python3
"""Validate repeated simple/fine transitions in DROID episode 50458."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_pilot_sequences import load_aligned_sequences, load_joint_range
from quick_user_boundary_test import FEATURES, simple_features


EPISODE = 50458
FPS = 15.0
SEGMENTS = (
    ("simple_1", "simple", 1, 0, 38),
    ("fine_1", "fine", 1, 38, 125),
    ("simple_2", "simple", 2, 125, 165),
    ("fine_2", "fine", 2, 165, 188),
)
GROUPS = {
    "joint_step_only": ("joint_step",),
    "joint_only": (
        "joint_step",
        "joint_step_change",
        "joint_jerk",
        "direction_change",
    ),
    "gripper_only": ("gripper_value", "gripper_change"),
    "joint_gripper": FEATURES,
    "time_only": ("seconds",),
}
COLORS = {
    "simple_1": "#dbeafe",
    "fine_1": "#fee2e2",
    "simple_2": "#bfdbfe",
    "fine_2": "#fecaca",
}


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
    parser.add_argument("--schedule", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        default="research/action_phase_segmentation/pilot/episode_50458_multiphase",
    )
    return parser.parse_args()


def assign_segments(frames: np.ndarray) -> pd.DataFrame:
    segment = np.full(len(frames), "excluded", dtype=object)
    label = np.full(len(frames), "excluded", dtype=object)
    cycle = np.zeros(len(frames), dtype=int)
    for name, kind, cycle_index, start, end in SEGMENTS:
        mask = (frames >= start) & (frames < end)
        segment[mask] = name
        label[mask] = kind
        cycle[mask] = cycle_index
    return pd.DataFrame({"segment": segment, "label": label, "cycle": cycle})


def build_steps(
    source: str,
    frames: np.ndarray,
    features: dict[str, np.ndarray],
) -> pd.DataFrame:
    labels = assign_segments(frames)
    result = pd.DataFrame(
        {
            "source": source,
            "frame": frames,
            "seconds": frames / FPS,
            "chunk_offset": np.arange(len(frames)) % 24,
        }
    )
    result = pd.concat([result, labels], axis=1)
    for name, values in features.items():
        result[name] = values
    result["used"] = (result["chunk_offset"] >= 3) & (result["cycle"] > 0)
    return result


def segment_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, segment), group in steps[steps["used"]].groupby(
        ["source", "segment"], sort=False
    ):
        row: dict[str, object] = {
            "source": source,
            "segment": segment,
            "label": group["label"].iloc[0],
            "cycle": int(group["cycle"].iloc[0]),
            "n": len(group),
        }
        for feature in FEATURES:
            row[f"{feature}_mean"] = float(group[feature].mean())
            row[f"{feature}_median"] = float(group[feature].median())
        rows.append(row)
    order = {name: index for index, (name, *_rest) in enumerate(SEGMENTS)}
    result = pd.DataFrame(rows)
    result["segment_order"] = result["segment"].map(order)
    return result.sort_values(["source", "segment_order"]).drop(
        columns="segment_order"
    )


def binary_feature_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, source_steps in steps[steps["used"]].groupby("source"):
        target = (source_steps["label"] == "fine").astype(int)
        for feature in FEATURES:
            raw_auc = float(roc_auc_score(target, source_steps[feature]))
            rows.append(
                {
                    "source": source,
                    "feature": feature,
                    "raw_auc_fine_high": raw_auc,
                    "separation_auc": max(raw_auc, 1.0 - raw_auc),
                    "fine_direction": "higher" if raw_auc >= 0.5 else "lower",
                }
            )
    return pd.DataFrame(rows)


def cycle_transfer_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, source_steps in steps[steps["used"]].groupby("source"):
        source_steps = source_steps.copy()
        source_steps["target"] = (source_steps["label"] == "fine").astype(int)
        for train_cycle, test_cycle in ((1, 2), (2, 1)):
            train = source_steps[source_steps["cycle"] == train_cycle]
            test = source_steps[source_steps["cycle"] == test_cycle]
            for group, columns in GROUPS.items():
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=1000,
                                random_state=20260719,
                            ),
                        ),
                    ]
                )
                model.fit(train[list(columns)], train["target"])
                probability = model.predict_proba(test[list(columns)])[:, 1]
                prediction = (probability >= 0.5).astype(int)
                rows.append(
                    {
                        "source": source,
                        "train_cycle": train_cycle,
                        "test_cycle": test_cycle,
                        "group": group,
                        "train_n": len(train),
                        "test_n": len(test),
                        "auroc": roc_auc_score(test["target"], probability),
                        "balanced_accuracy": balanced_accuracy_score(
                            test["target"], prediction
                        ),
                        "macro_f1": f1_score(
                            test["target"], prediction, average="macro"
                        ),
                        "predicted_fine_rate": float(prediction.mean()),
                    }
                )
    return pd.DataFrame(rows)


def plot_timeline(steps: pd.DataFrame, output: Path) -> None:
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
    fig, axes = plt.subplots(len(shown), 1, figsize=(13, 10), sharex=True)
    for axis, feature in zip(axes, shown):
        for name, _kind, _cycle, start, end in SEGMENTS:
            axis.axvspan(start / FPS, end / FPS, color=COLORS[name], alpha=0.8)
        for source, color, width in (("gt", "#94a3b8", 1.0), ("wam", "#111827", 1.3)):
            source_steps = steps[steps["source"] == source]
            axis.plot(
                source_steps["seconds"],
                source_steps[feature],
                color=color,
                linewidth=width,
                label=source.upper(),
            )
        axis.set_ylabel(labels[feature])
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("video time (seconds)")
    axes[-1].set_xlim(23 / FPS, 190 / FPS)
    fig.suptitle(
        "Episode 50458: simple1 / fine1 / simple2 / fine2 video labels",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned = load_aligned_sequences(
        Path(args.predictions), {EPISODE}, args.schedule
    )[EPISODE]
    joint_range = load_joint_range(Path(args.dataset))
    all_steps = []
    for source in ("gt", "wam"):
        features = simple_features(
            aligned[source], joint_range, aligned["chunk_starts"]
        )
        all_steps.append(build_steps(source, aligned["frame"], features))
    steps = pd.concat(all_steps, ignore_index=True)

    segments = segment_metrics(steps)
    feature_metrics = binary_feature_metrics(steps)
    transfer = cycle_transfer_metrics(steps)
    steps.to_csv(output_dir / "per_step_features.csv", index=False)
    segments.to_csv(output_dir / "segment_metrics.csv", index=False)
    feature_metrics.to_csv(output_dir / "binary_feature_metrics.csv", index=False)
    transfer.to_csv(output_dir / "cycle_transfer_metrics.csv", index=False)
    plot_timeline(steps, output_dir / "timeline.png")

    manifest = {
        "status": "exploratory_video_labelled_multiphase_check",
        "episode": EPISODE,
        "fps": FPS,
        "segments": [
            {
                "name": name,
                "label": label,
                "cycle": cycle,
                "frame_start_inclusive": start,
                "frame_end_exclusive": end,
                "seconds": [start / FPS, end / FPS],
            }
            for name, label, cycle, start, end in SEGMENTS
        ],
        "wam_coverage_frames": [
            int(aligned["frame"][0]),
            int(aligned["frame"][-1]),
        ],
        "excluded_chunk_offsets": [0, 1, 2],
        "warning": "One episode and visually approximate boundaries; exploratory only.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("WAM segment means")
    wam_segments = segments[segments["source"] == "wam"]
    print(
        wam_segments[
            [
                "segment",
                "n",
                "joint_step_mean",
                "joint_step_change_mean",
                "direction_change_mean",
                "gripper_value_mean",
                "gripper_change_mean",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nWAM cycle transfer")
    print(
        transfer[transfer["source"] == "wam"].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )


if __name__ == "__main__":
    main()
