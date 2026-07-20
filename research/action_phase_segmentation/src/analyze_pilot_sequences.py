#!/usr/bin/env python3
"""Compare GT and WAM step-level action changes with video-only boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_GROUPS = (
    "joint_position",
    "joint_d1",
    "joint_scalar",
    "joint_dynamics",
    "gripper_only",
    "joint_gripper",
)
CLASSIFICATION_GROUPS = FEATURE_GROUPS + (
    "joint_window9",
    "joint_gripper_window9",
    "time_only",
    "time_gripper",
)
EVENT_COLORS = {
    "approach_onset": "#3b82f6",
    "contact_onset": "#ef4444",
    "object_motion_onset": "#f97316",
    "grasp_lift_onset": "#a855f7",
    "transport_onset": "#14b8a6",
    "placement_contact": "#eab308",
    "release_onset": "#ec4899",
    "retreat_onset": "#64748b",
}
EVENT_SHORT = {
    "approach_onset": "A",
    "contact_onset": "C",
    "object_motion_onset": "O",
    "grasp_lift_onset": "L",
    "transport_onset": "T",
    "placement_contact": "P",
    "release_onset": "R",
    "retreat_onset": "X",
}
PHASE_ORDER = (
    "free_pre",
    "terminal_approach",
    "contact_manipulation",
    "stable_transport",
    "placement_release",
)
PHASE_FEATURES = (
    "speed",
    "accel_norm",
    "jerk_norm",
    "direction_change",
    "gripper",
    "gripper_abs_d1",
)
PHASE_TRANSITIONS = (
    ("free_pre", "terminal_approach"),
    ("terminal_approach", "contact_manipulation"),
    ("contact_manipulation", "stable_transport"),
    ("stable_transport", "placement_release"),
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
        "--labels",
        default="research/action_phase_segmentation/pilot/video_labels_pilot.json",
    )
    parser.add_argument(
        "--dataset", default="/home/admin/.cache/DreamZero-DROID-Data"
    )
    parser.add_argument(
        "--output-dir", default="research/action_phase_segmentation/pilot/analysis"
    )
    parser.add_argument("--schedule", type=int, default=16)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--peak-distance", type=int, default=6)
    parser.add_argument("--random-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def load_joint_range(dataset: Path) -> np.ndarray:
    stats = json.loads((dataset / "meta/stats.json").read_text())
    q01 = np.asarray(stats["action"]["q01"], dtype=np.float64)[14:21]
    q99 = np.asarray(stats["action"]["q99"], dtype=np.float64)[14:21]
    return np.maximum(q99 - q01, 1e-6)


def load_aligned_sequences(
    predictions: Path,
    episodes: set[int],
    schedule: int,
) -> dict[int, dict[str, np.ndarray]]:
    selected: dict[int, list[dict[str, object]]] = {episode: [] for episode in episodes}
    with predictions.open() as handle:
        for line in handle:
            row = json.loads(line)
            episode = int(row.get("episode", -1))
            if (
                episode in selected
                and row.get("record_type") == "chunk"
                and int(row["schedule"]) == schedule
                and int(row["run_rep"]) == 0
            ):
                selected[episode].append(row)

    result: dict[int, dict[str, np.ndarray]] = {}
    for episode, rows in selected.items():
        rows.sort(key=lambda row: int(row["anchor"]))
        if not rows:
            raise RuntimeError(f"No WAM rows for episode {episode}, schedule {schedule}")
        anchors = np.asarray([int(row["anchor"]) for row in rows], dtype=int)
        if len(anchors) > 1 and not np.all(np.diff(anchors) == 24):
            raise RuntimeError(f"Non-contiguous anchors for episode {episode}: {anchors}")
        frame_index = np.concatenate(
            [np.arange(anchor, anchor + 24, dtype=int) for anchor in anchors]
        )
        gt = np.concatenate(
            [np.asarray(row["gt_action"], dtype=np.float64) for row in rows], axis=0
        )
        wam = np.concatenate(
            [np.asarray(row["prediction"], dtype=np.float64) for row in rows], axis=0
        )
        chunk_starts = np.arange(len(rows), dtype=int) * 24
        expected = (len(frame_index), 8)
        if gt.shape != expected or wam.shape != expected:
            raise RuntimeError(
                f"Shape mismatch episode {episode}: frames={frame_index.shape}, "
                f"gt={gt.shape}, wam={wam.shape}"
            )
        result[episode] = {
            "frame": frame_index,
            "gt": gt,
            "wam": wam,
            "chunk_edges": anchors[1:],
            "chunk_starts": chunk_starts,
        }
    return result


def smooth_columns(values: np.ndarray) -> np.ndarray:
    if len(values) < 5:
        return values.copy()
    return savgol_filter(values, window_length=5, polyorder=2, axis=0, mode="interp")


def difference(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    result[1:] = np.diff(values, axis=0)
    result[0] = result[1]
    return result


def robust_standardize(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)
    standardized = (values - median) / scale
    return np.clip(standardized, -10.0, 10.0)


def rolling_statistics(
    values: np.ndarray, chunk_starts: np.ndarray, radius: int = 4
) -> np.ndarray:
    result = np.empty((len(values), values.shape[1] * 3), dtype=np.float64)
    for chunk_index, start in enumerate(chunk_starts):
        end = int(chunk_starts[chunk_index + 1]) if chunk_index + 1 < len(chunk_starts) else len(values)
        for index in range(int(start), end):
            left = max(int(start), index - radius)
            right = min(end, index + radius + 1)
            window = values[left:right]
            slope = (window[-1] - window[0]) / max(len(window) - 1, 1)
            result[index] = np.concatenate([window.mean(axis=0), window.std(axis=0), slope])
    return result


def action_features(
    action: np.ndarray,
    joint_range: np.ndarray,
    chunk_starts: np.ndarray,
) -> dict[str, np.ndarray]:
    joint = np.empty((len(action), 7), dtype=np.float64)
    gripper = np.empty((len(action), 1), dtype=np.float64)
    velocity = np.empty((len(action), 7), dtype=np.float64)
    acceleration = np.empty((len(action), 7), dtype=np.float64)
    jerk = np.empty((len(action), 7), dtype=np.float64)
    gripper_d1 = np.empty((len(action), 1), dtype=np.float64)
    direction_change = np.zeros(len(action), dtype=np.float64)

    for chunk_index, start in enumerate(chunk_starts):
        end = int(chunk_starts[chunk_index + 1]) if chunk_index + 1 < len(chunk_starts) else len(action)
        chunk_joint = smooth_columns(action[start:end, :7] / joint_range[None, :])
        chunk_gripper = smooth_columns(action[start:end, 7:8])
        chunk_velocity = difference(chunk_joint)
        chunk_acceleration = difference(chunk_velocity)
        chunk_jerk = difference(chunk_acceleration)
        chunk_gripper_d1 = difference(chunk_gripper)

        previous_velocity = np.vstack([chunk_velocity[0], chunk_velocity[:-1]])
        denominator = np.linalg.norm(chunk_velocity, axis=1) * np.linalg.norm(
            previous_velocity, axis=1
        )
        cosine = np.ones(len(chunk_velocity), dtype=np.float64)
        valid = denominator > 1e-8
        cosine[valid] = np.sum(
            chunk_velocity[valid] * previous_velocity[valid], axis=1
        ) / denominator[valid]

        joint[start:end] = chunk_joint
        gripper[start:end] = chunk_gripper
        velocity[start:end] = chunk_velocity
        acceleration[start:end] = chunk_acceleration
        jerk[start:end] = chunk_jerk
        gripper_d1[start:end] = chunk_gripper_d1
        direction_change[start:end] = 1.0 - np.clip(cosine, -1.0, 1.0)

    speed = np.linalg.norm(velocity, axis=1)
    accel_norm = np.linalg.norm(acceleration, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1)
    joint_scalar = np.column_stack(
        [speed, accel_norm, jerk_norm, direction_change]
    )
    joint_window9 = np.column_stack(
        [
            rolling_statistics(velocity, chunk_starts, radius=4),
            rolling_statistics(joint_scalar, chunk_starts, radius=4),
        ]
    )
    gripper_window9 = rolling_statistics(
        np.column_stack([gripper[:, 0], np.abs(gripper_d1[:, 0])]),
        chunk_starts,
        radius=4,
    )
    return {
        "joint": joint,
        "gripper": gripper[:, 0],
        "velocity": velocity,
        "acceleration": acceleration,
        "jerk": jerk,
        "gripper_d1": gripper_d1[:, 0],
        "speed": speed,
        "accel_norm": accel_norm,
        "jerk_norm": jerk_norm,
        "direction_change": direction_change,
        "joint_window9": joint_window9,
        "joint_gripper_window9": np.column_stack(
            [joint_window9, gripper_window9]
        ),
        "time_only": np.linspace(0.0, 1.0, len(action), dtype=np.float64)[:, None],
        "time_gripper": np.column_stack(
            [
                np.linspace(0.0, 1.0, len(action), dtype=np.float64),
                gripper[:, 0],
                gripper_d1[:, 0],
            ]
        ),
    }


def feature_matrix(features: dict[str, np.ndarray], group: str) -> np.ndarray:
    if group == "joint_position":
        return features["joint"]
    if group == "joint_d1":
        return features["velocity"]
    if group == "joint_scalar":
        return np.column_stack(
            [
                features["speed"],
                features["accel_norm"],
                features["jerk_norm"],
                features["direction_change"],
            ]
        )
    if group == "joint_dynamics":
        return np.column_stack(
            [
                features["velocity"],
                features["acceleration"],
                features["speed"],
                features["direction_change"],
            ]
        )
    if group == "gripper_only":
        return np.column_stack([features["gripper"], features["gripper_d1"]])
    if group == "joint_gripper":
        return np.column_stack(
            [
                feature_matrix(features, "joint_dynamics"),
                features["gripper"],
                features["gripper_d1"],
            ]
        )
    if group == "joint_window9":
        return features["joint_window9"]
    if group == "joint_gripper_window9":
        return features["joint_gripper_window9"]
    if group == "time_only":
        return features["time_only"]
    if group == "time_gripper":
        return features["time_gripper"]
    raise ValueError(group)


def window_change_score(values: np.ndarray, window: int) -> np.ndarray:
    values = robust_standardize(values)
    score = np.zeros(len(values), dtype=np.float64)
    for index in range(window, len(values) - window):
        before = values[index - window : index]
        after = values[index : index + window]
        mean_shift = np.linalg.norm(before.mean(axis=0) - after.mean(axis=0))
        variance_shift = np.mean(
            np.abs(np.log(before.var(axis=0) + 0.1) - np.log(after.var(axis=0) + 0.1))
        )
        score[index] = mean_shift / np.sqrt(values.shape[1]) + 0.25 * variance_shift
    return gaussian_filter1d(score, sigma=1.0)


def candidate_peaks(score: np.ndarray, distance: int, margin: int) -> np.ndarray:
    peaks, _ = find_peaks(score, distance=distance, prominence=0.05)
    return peaks[(peaks >= margin) & (peaks < len(score) - margin)]


def top_k_peaks(score: np.ndarray, k: int, distance: int, margin: int) -> np.ndarray:
    peaks = candidate_peaks(score, distance, margin)
    if len(peaks) == 0 or k == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(score[peaks])[::-1]
    return np.sort(peaks[order[:k]])


def match_count(predicted: np.ndarray, truth: np.ndarray, tolerance: int) -> int:
    pairs = sorted(
        (
            (abs(int(prediction) - int(target)), int(prediction), int(target))
            for prediction in predicted
            for target in truth
            if abs(int(prediction) - int(target)) <= tolerance
        ),
        key=lambda item: item[0],
    )
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    for _, prediction, target in pairs:
        if prediction not in used_predictions and target not in used_truth:
            used_predictions.add(prediction)
            used_truth.add(target)
    return len(used_truth)


def random_hit_rate(
    valid_frames: np.ndarray,
    truth: np.ndarray,
    k: int,
    tolerance: int,
    reps: int,
    rng: np.random.Generator,
) -> float:
    if k == 0 or len(valid_frames) < k:
        return 0.0
    rates = []
    for _ in range(reps):
        predicted = rng.choice(valid_frames, size=k, replace=False)
        rates.append(match_count(predicted, truth, tolerance) / len(truth))
    return float(np.mean(rates))


def event_percentiles(
    frames: np.ndarray,
    score: np.ndarray,
    events: list[dict[str, object]],
    tolerance: int,
) -> list[float]:
    valid_score = score[np.isfinite(score)]
    percentiles = []
    for event in events:
        center = int(event["frame"])
        local = score[np.abs(frames - center) <= tolerance]
        if len(local) == 0:
            continue
        value = float(np.max(local))
        percentiles.append(float(np.mean(valid_score <= value)))
    return percentiles


def phase_intervals(
    events: list[dict[str, object]], x_min: int, x_max: int
) -> list[tuple[str, int, int]]:
    centers = {str(event["name"]): int(event["frame"]) for event in events}
    required = ("approach_onset", "contact_onset", "placement_contact")
    if any(name not in centers for name in required):
        return []
    approach = centers["approach_onset"]
    contact = centers["contact_onset"]
    placement = centers["placement_contact"]
    transport = centers.get("transport_onset")
    end = min(centers.get("retreat_onset", x_max + 1), x_max + 1)
    intervals = [
        ("free_pre", x_min, approach),
        ("terminal_approach", approach, contact),
    ]
    if transport is None:
        intervals.append(("contact_manipulation", contact, placement))
    else:
        intervals.extend(
            [
                ("contact_manipulation", contact, transport),
                ("stable_transport", transport, placement),
            ]
        )
    intervals.append(("placement_release", placement, end))
    return [interval for interval in intervals if interval[2] - interval[1] >= 3]


def phase_feature_rows(
    episode: int,
    source: str,
    frames: np.ndarray,
    features: dict[str, np.ndarray],
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    scalar = np.column_stack(
        [
            features["speed"],
            features["accel_norm"],
            features["jerk_norm"],
            features["direction_change"],
            features["gripper"],
            np.abs(features["gripper_d1"]),
        ]
    )
    standardized = robust_standardize(scalar)
    unambiguous = np.ones(len(frames), dtype=bool)
    for event in events:
        lo, hi = int(event["frame_lo"]), int(event["frame_hi"])
        unambiguous &= ~((frames >= lo) & (frames <= hi))
    rows = []
    for phase, start, end in phase_intervals(events, int(frames[0]), int(frames[-1])):
        mask = (frames >= start) & (frames < end) & unambiguous
        values = scalar[mask]
        z_values = standardized[mask]
        if len(values) == 0:
            continue
        row: dict[str, object] = {
            "episode": episode,
            "source": source,
            "phase": phase,
            "start": start,
            "end": end,
            "steps": int(mask.sum()),
        }
        for column, name in enumerate(PHASE_FEATURES):
            row[f"{name}_mean"] = float(values[:, column].mean())
            row[f"{name}_median"] = float(np.median(values[:, column]))
            row[f"{name}_z_mean"] = float(z_values[:, column].mean())
        speed_mean = float(values[:, 0].mean())
        row["speed_cv"] = float(values[:, 0].std() / max(speed_mean, 1e-8))
        rows.append(row)
    return rows


def plot_phase_signatures(phase_rows: pd.DataFrame, output: Path) -> None:
    labels = (
        "speed",
        "accel",
        "jerk",
        "direction",
        "gripper",
        "gripper delta",
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, source in zip(axes, ("gt", "wam"), strict=True):
        subset = phase_rows[phase_rows["source"] == source]
        matrix = np.full((len(PHASE_ORDER), len(PHASE_FEATURES)), np.nan)
        counts = np.zeros(len(PHASE_ORDER), dtype=int)
        for row_index, phase in enumerate(PHASE_ORDER):
            phase_data = subset[subset["phase"] == phase]
            counts[row_index] = phase_data["episode"].nunique()
            for column, feature in enumerate(PHASE_FEATURES):
                if not phase_data.empty:
                    matrix[row_index, column] = phase_data[f"{feature}_z_mean"].mean()
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(
            range(len(PHASE_ORDER)),
            [f"{phase} (n={counts[i]})" for i, phase in enumerate(PHASE_ORDER)],
        )
        axis.set_title(f"{source.upper()} phase signature\nwithin-episode robust z-score")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if np.isfinite(matrix[row, column]):
                    axis.text(
                        column,
                        row,
                        f"{matrix[row, column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
    figure.colorbar(image, ax=axes, shrink=0.8, label="phase mean robust z")
    figure.suptitle("Video-defined phase signatures (3-episode exploratory pilot)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def build_phase_contrasts(phases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    median_features = (
        "speed",
        "accel_norm",
        "jerk_norm",
        "direction_change",
        "gripper",
        "gripper_abs_d1",
    )
    for (episode, source), group in phases.groupby(["episode", "source"]):
        by_phase = group.set_index("phase")
        for before, after in PHASE_TRANSITIONS:
            if before not in by_phase.index or after not in by_phase.index:
                continue
            row: dict[str, object] = {
                "episode": int(episode),
                "source": source,
                "transition": f"{before}->{after}",
            }
            for feature in median_features:
                left = float(by_phase.loc[before, f"{feature}_median"])
                right = float(by_phase.loc[after, f"{feature}_median"])
                row[f"{feature}_before"] = left
                row[f"{feature}_after"] = right
                row[f"{feature}_delta"] = right - left
                row[f"{feature}_ratio"] = right / max(abs(left), 1e-8)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_phase_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (source, transition), group in contrasts.groupby(["source", "transition"]):
        row: dict[str, object] = {
            "source": source,
            "transition": transition,
            "episodes": group["episode"].nunique(),
        }
        for feature in PHASE_FEATURES:
            delta = group[f"{feature}_delta"]
            ratio = group[f"{feature}_ratio"]
            row[f"{feature}_median_ratio"] = float(np.median(ratio))
            row[f"{feature}_increase_count"] = int((delta > 0).sum())
            row[f"{feature}_decrease_count"] = int((delta < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def phase_labels(
    frames: np.ndarray,
    events: list[dict[str, object]],
    chunk_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(len(frames), "", dtype=object)
    for phase, start, end in phase_intervals(events, int(frames[0]), int(frames[-1])):
        labels[(frames >= start) & (frames < end)] = phase
    valid = labels != ""
    for event in events:
        lo, hi = int(event["frame_lo"]), int(event["frame_hi"])
        valid &= ~((frames >= lo) & (frames <= hi))
    for start in chunk_starts:
        valid[int(start) : min(int(start) + 3, len(valid))] = False
    return labels, valid


def evaluate_phase_classification(
    blocks: dict[tuple[str, str], list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]],
    output_dir: Path,
) -> pd.DataFrame:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for (source, feature_group), group_blocks in blocks.items():
        episodes = sorted(block[0] for block in group_blocks)
        all_truth: list[np.ndarray] = []
        all_prediction: list[np.ndarray] = []
        for test_episode in episodes:
            train = [block for block in group_blocks if block[0] != test_episode]
            test = next(block for block in group_blocks if block[0] == test_episode)
            x_train = np.concatenate([block[1] for block in train], axis=0)
            y_train = np.concatenate([block[2] for block in train], axis=0)
            _, x_test, y_test, frame_test = test
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            C=1.0,
                            max_iter=5000,
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            all_truth.append(y_test)
            all_prediction.append(prediction)
            metric_rows.append(
                {
                    "source": source,
                    "feature_group": feature_group,
                    "scope": f"episode_{test_episode:06d}",
                    "steps": len(y_test),
                    "accuracy": accuracy_score(y_test, prediction),
                    "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                    "macro_f1_global_labels": f1_score(
                        y_test,
                        prediction,
                        labels=list(PHASE_ORDER),
                        average="macro",
                        zero_division=0,
                    ),
                }
            )
            for frame_value, truth_value, prediction_value in zip(
                frame_test, y_test, prediction, strict=True
            ):
                prediction_rows.append(
                    {
                        "episode": test_episode,
                        "frame": int(frame_value),
                        "source": source,
                        "feature_group": feature_group,
                        "truth": truth_value,
                        "prediction": prediction_value,
                    }
                )
        truth = np.concatenate(all_truth)
        prediction = np.concatenate(all_prediction)
        metric_rows.append(
            {
                "source": source,
                "feature_group": feature_group,
                "scope": "all_oof",
                "steps": len(truth),
                "accuracy": accuracy_score(truth, prediction),
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "macro_f1_global_labels": f1_score(
                    truth,
                    prediction,
                    labels=list(PHASE_ORDER),
                    average="macro",
                    zero_division=0,
                ),
            }
        )

        matrix = confusion_matrix(truth, prediction, labels=list(PHASE_ORDER), normalize="true")
        figure, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
        image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(PHASE_ORDER)), PHASE_ORDER, rotation=35, ha="right")
        axis.set_yticks(range(len(PHASE_ORDER)), PHASE_ORDER)
        axis.set_xlabel("predicted")
        axis.set_ylabel("video-defined phase")
        axis.set_title(f"{source.upper()} {feature_group}\nleave-one-episode-out")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, label="row-normalized fraction")
        figure.savefig(
            output_dir / f"phase_confusion_{source}_{feature_group}.png", dpi=150
        )
        plt.close(figure)

    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "phase_classification_predictions.csv", index=False
    )
    return pd.DataFrame(metric_rows)


def add_event_overlay(
    axes: np.ndarray,
    events: list[dict[str, object]],
    x_min: int,
    x_max: int,
) -> None:
    for event in events:
        center = int(event["frame"])
        if not x_min <= center <= x_max:
            continue
        name = str(event["name"])
        color = EVENT_COLORS[name]
        lo, hi = int(event["frame_lo"]), int(event["frame_hi"])
        for axis in axes.flat:
            axis.axvspan(lo, hi, color=color, alpha=0.07, linewidth=0)
            axis.axvline(center, color=color, alpha=0.55, linewidth=0.8)


def plot_episode(
    episode: int,
    task: str,
    sequence: dict[str, np.ndarray],
    features_by_source: dict[str, dict[str, np.ndarray]],
    scores_by_source: dict[str, dict[str, np.ndarray]],
    top_peaks: dict[str, dict[str, np.ndarray]],
    events: list[dict[str, object]],
    output: Path,
) -> None:
    frame = sequence["frame"]
    sources = ("gt", "wam")
    rows = 7
    figure, axes = plt.subplots(
        rows,
        2,
        figsize=(18, 15),
        sharex=True,
        constrained_layout=True,
    )
    for column, source in enumerate(sources):
        features = features_by_source[source]
        axes[0, column].plot(frame, features["speed"], color="#2563eb")
        axes[0, column].set_ylabel("joint speed")
        axes[1, column].plot(frame, features["accel_norm"], color="#dc2626", label="accel")
        axes[1, column].plot(frame, features["jerk_norm"], color="#f59e0b", alpha=0.8, label="jerk")
        axes[1, column].set_ylabel("accel / jerk")
        axes[1, column].legend(loc="upper right", fontsize=8)
        axes[2, column].plot(frame, features["direction_change"], color="#7c3aed")
        axes[2, column].set_ylabel("direction change")
        axes[3, column].plot(frame, features["gripper"], color="#0891b2", label="position")
        axes[3, column].plot(
            frame,
            np.abs(features["gripper_d1"]),
            color="#db2777",
            alpha=0.8,
            label="abs delta",
        )
        axes[3, column].set_ylabel("gripper")
        axes[3, column].legend(loc="upper right", fontsize=8)
        for row, group in enumerate(("joint_d1", "joint_dynamics", "joint_gripper"), start=4):
            score = scores_by_source[source][group]
            axes[row, column].plot(frame, score, color="#111827")
            peaks = top_peaks[source][group]
            axes[row, column].scatter(
                frame[peaks], score[peaks], marker="v", color="#16a34a", s=28, zorder=4
            )
            axes[row, column].set_ylabel(f"score\n{group}")
        axes[0, column].set_title(f"{source.upper()} action")
        for edge in sequence["chunk_edges"]:
            for axis in axes[:, column]:
                axis.axvline(edge, color="#94a3b8", linestyle=":", linewidth=0.7)
        axes[-1, column].set_xlabel("DROID frame (15 Hz)")

    add_event_overlay(axes, events, int(frame[0]), int(frame[-1]))
    label_axis = axes[0, 0]
    y_top = label_axis.get_ylim()[1]
    for event in events:
        center = int(event["frame"])
        if frame[0] <= center <= frame[-1]:
            label_axis.text(
                center,
                y_top,
                EVENT_SHORT[str(event["name"])],
                color=EVENT_COLORS[str(event["name"])],
                fontsize=8,
                ha="center",
                va="bottom",
            )
    figure.suptitle(f"Episode {episode}: {task}\nGreen triangles are top-K ranking diagnostics", fontsize=13)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    labels = json.loads(Path(args.labels).read_text())
    label_rows = {int(row["episode"]): row for row in labels["episodes"]}
    sequences = load_aligned_sequences(
        Path(args.predictions), set(label_rows), args.schedule
    )
    joint_range = load_joint_range(Path(args.dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    classification_blocks: dict[
        tuple[str, str], list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]
    ] = {}

    for episode, label_row in label_rows.items():
        sequence = sequences[episode]
        frame = sequence["frame"]
        events = [
            event
            for event in label_row["events"]
            if int(frame[0]) <= int(event["frame"]) <= int(frame[-1])
        ]
        truth = np.asarray([int(event["frame"]) for event in events], dtype=int)
        features_by_source = {
            source: action_features(
                sequence[source],
                joint_range,
                sequence["chunk_starts"],
            )
            for source in ("gt", "wam")
        }
        for source, features in features_by_source.items():
            phase_rows.extend(
                phase_feature_rows(episode, source, frame, features, events)
            )
            labels_for_steps, valid_steps = phase_labels(
                frame, events, sequence["chunk_starts"]
            )
            for group in CLASSIFICATION_GROUPS:
                classification_blocks.setdefault((source, group), []).append(
                    (
                        episode,
                        feature_matrix(features, group)[valid_steps],
                        labels_for_steps[valid_steps],
                        frame[valid_steps],
                    )
                )
        scores_by_source: dict[str, dict[str, np.ndarray]] = {}
        top_peaks: dict[str, dict[str, np.ndarray]] = {}
        for source, features in features_by_source.items():
            scores_by_source[source] = {}
            top_peaks[source] = {}
            for group in FEATURE_GROUPS:
                score = window_change_score(feature_matrix(features, group), args.window)
                scores_by_source[source][group] = score
                peak_index = top_k_peaks(
                    score, len(events), args.peak_distance, args.window
                )
                top_peaks[source][group] = peak_index
                predicted_frames = frame[peak_index]
                hits = match_count(predicted_frames, truth, tolerance=5)
                valid_frames = frame[args.window : -args.window]
                percentiles = event_percentiles(frame, score, events, tolerance=5)
                metric_rows.append(
                    {
                        "episode": episode,
                        "source": source,
                        "feature_group": group,
                        "n_events": len(events),
                        "n_candidate_peaks": len(
                            candidate_peaks(score, args.peak_distance, args.window)
                        ),
                        "top_k_hits_at_5": hits,
                        "top_k_hit_rate_at_5": hits / len(events),
                        "random_hit_rate_at_5": random_hit_rate(
                            valid_frames,
                            truth,
                            len(events),
                            5,
                            args.random_reps,
                            rng,
                        ),
                        "mean_event_score_percentile_at_5": float(np.mean(percentiles)),
                        "top_k_chunk_edge_fraction_at_2": float(
                            np.mean(
                                [
                                    np.min(np.abs(sequence["chunk_edges"] - value)) <= 2
                                    for value in predicted_frames
                                ]
                            )
                        )
                        if len(predicted_frames)
                        else 0.0,
                        "top_k_chunk_edge_count_at_2": int(
                            np.sum(
                                [
                                    np.min(np.abs(sequence["chunk_edges"] - value)) <= 2
                                    for value in predicted_frames
                                ]
                            )
                        ),
                        "top_k_selected": len(predicted_frames),
                    }
                )
                for event in events:
                    center = int(event["frame"])
                    local = np.flatnonzero(np.abs(frame - center) <= 5)
                    local_best = local[np.argmax(score[local])]
                    event_rows.append(
                        {
                            "episode": episode,
                            "source": source,
                            "feature_group": group,
                            "event": event["name"],
                            "event_frame": center,
                            "best_frame_within_5": int(frame[local_best]),
                            "best_score_within_5": float(score[local_best]),
                            "score_percentile": float(np.mean(score <= score[local_best])),
                        }
                    )

        plot_episode(
            episode,
            str(label_row["task"]),
            sequence,
            features_by_source,
            scores_by_source,
            top_peaks,
            events,
            output_dir / f"episode_{episode:06d}_alignment.png",
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics_by_episode.csv", index=False)
    events = pd.DataFrame(event_rows)
    events.to_csv(output_dir / "event_scores.csv", index=False)
    phases = pd.DataFrame(phase_rows)
    phases.to_csv(output_dir / "phase_features.csv", index=False)
    contrasts = build_phase_contrasts(phases)
    contrasts.to_csv(output_dir / "phase_contrasts_by_episode.csv", index=False)
    contrast_summary = summarize_phase_contrasts(contrasts)
    contrast_summary.to_csv(output_dir / "phase_contrast_summary.csv", index=False)
    classification_metrics = evaluate_phase_classification(
        classification_blocks, output_dir
    )
    classification_metrics.to_csv(
        output_dir / "phase_classification_metrics.csv", index=False
    )
    phase_aggregate = (
        phases.groupby(["source", "phase"], as_index=False)
        .agg(
            episodes=("episode", "nunique"),
            steps=("steps", "sum"),
            **{
                f"{feature}_z_mean": (f"{feature}_z_mean", "mean")
                for feature in PHASE_FEATURES
            },
        )
    )
    phase_aggregate.to_csv(output_dir / "phase_signature_aggregate.csv", index=False)
    plot_phase_signatures(phases, output_dir / "phase_signatures.png")
    aggregate = (
        metrics.groupby(["source", "feature_group"], as_index=False)
        .agg(
            episodes=("episode", "nunique"),
            events=("n_events", "sum"),
            top_k_hits_at_5=("top_k_hits_at_5", "sum"),
            random_expected_hits_at_5=("random_hit_rate_at_5", lambda values: 0.0),
            mean_event_score_percentile_at_5=(
                "mean_event_score_percentile_at_5",
                "mean",
            ),
            top_k_chunk_edge_count_at_2=("top_k_chunk_edge_count_at_2", "sum"),
            top_k_selected=("top_k_selected", "sum"),
        )
    )
    random_expected = (
        metrics.assign(
            random_expected_hits_at_5=lambda frame: frame["random_hit_rate_at_5"]
            * frame["n_events"]
        )
        .groupby(["source", "feature_group"])["random_expected_hits_at_5"]
        .sum()
    )
    aggregate = aggregate.drop(columns=["random_expected_hits_at_5"])
    aggregate["top_k_hit_rate_at_5"] = (
        aggregate["top_k_hits_at_5"] / aggregate["events"]
    )
    aggregate["random_hit_rate_at_5"] = [
        random_expected.loc[(row.source, row.feature_group)] / row.events
        for row in aggregate.itertuples(index=False)
    ]
    aggregate["top_k_chunk_edge_fraction_at_2"] = (
        aggregate["top_k_chunk_edge_count_at_2"] / aggregate["top_k_selected"]
    )
    aggregate = aggregate.sort_values(
        ["source", "top_k_hit_rate_at_5"], ascending=[True, False]
    )
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    manifest = {
        "status": "exploratory pilot; not confirmatory",
        "labels": str(Path(args.labels)),
        "predictions": str(Path(args.predictions)),
        "schedule": args.schedule,
        "window": args.window,
        "peak_distance": args.peak_distance,
        "top_k_definition": "K equals the number of in-coverage video events; ranking diagnostic only",
        "event_tolerance": 5,
        "derivative_definition": "Action-only, within-chunk differences. The undefined first derivative in each chunk is filled from that chunk's first forward difference; no current_state or previous rollout endpoint is used.",
        "joint_range": joint_range.tolist(),
        "episodes": sorted(label_rows),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
