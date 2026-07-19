#!/usr/bin/env python3
"""Analyze whether within/chunk-transition action geometry predicts gripper phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STAGES = ["free_open", "pre_close", "closing", "hold", "release"]
FINE_STAGES = {"pre_close", "closing", "hold", "release"}
JOINT_FEATURES = [
    "joint_path",
    "joint_step_mean",
    "joint_step_max",
    "joint_endpoint",
    "joint_accel_mean",
    "joint_accel_max",
    "joint_jerk_mean",
    "joint_jerk_max",
    "joint_direction_change",
    "joint_roughness",
] + [f"joint_abs_step_d{i}" for i in range(7)]
GRIPPER_FEATURES = [
    "gripper_start",
    "gripper_end",
    "gripper_mean",
    "gripper_range",
    "gripper_tv",
    "gripper_step_max",
    "gripper_endpoint",
    "gripper_high_fraction",
]


@dataclass(frozen=True)
class Config:
    dataset: str
    output_dir: str
    episodes: int
    seed: int
    chunk_horizon: int
    first_anchor: int
    bootstrap_reps: int
    min_gripper_span: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/admin/.cache/DreamZero-DROID-Data")
    parser.add_argument(
        "--output-dir",
        default="research/action_chunk_compute/experiments/h1_gt_chunk_phase/results/confirmatory_1200",
    )
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--chunk-horizon", type=int, default=24)
    parser.add_argument("--first-anchor", type=int, default=23)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--min-gripper-span", type=float, default=0.2)
    return parser.parse_args()


def git_info(repo: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, text=True, capture_output=True
        )
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "status_lines": len(status.splitlines()) if status else 0,
    }


def episode_path(dataset: Path, episode: int) -> Path:
    return dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"


def select_episodes(dataset: Path, count: int, seed: int, min_length: int) -> list[int]:
    candidates: list[int] = []
    with (dataset / "meta/episodes.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("success", False) and int(row["length"]) >= min_length:
                candidates.append(int(row["episode_index"]))
    if count > len(candidates):
        raise ValueError(f"Requested {count} episodes, only {len(candidates)} are eligible")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(candidates, size=count, replace=False).tolist())


def load_joint_ranges(dataset: Path) -> np.ndarray:
    stats = json.loads((dataset / "meta/stats.json").read_text())
    action_stats = stats["action"]
    q01 = np.asarray(action_stats["q01"], dtype=np.float64)[14:21]
    q99 = np.asarray(action_stats["q99"], dtype=np.float64)[14:21]
    return np.maximum(q99 - q01, 1e-6)


def hysteresis_gripper(
    gripper: np.ndarray, min_span: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    low = float(np.quantile(gripper, 0.05))
    high = float(np.quantile(gripper, 0.95))
    span = high - low
    if span < min_span:
        zeros = np.zeros(len(gripper), dtype=bool)
        normalized = np.zeros(len(gripper), dtype=np.float64)
        return normalized, zeros, zeros.copy(), zeros.copy(), False

    normalized = np.clip((gripper - low) / span, 0.0, 1.0)
    closed = np.zeros(len(gripper), dtype=bool)
    close_event = np.zeros(len(gripper), dtype=bool)
    release_event = np.zeros(len(gripper), dtype=bool)
    state = bool(normalized[0] >= 0.5)
    closed[0] = state
    for i in range(1, len(normalized)):
        if not state and normalized[i] >= 0.75:
            state = True
            close_event[i] = True
        elif state and normalized[i] <= 0.25:
            state = False
            release_event[i] = True
        closed[i] = state
    return normalized, closed, close_event, release_event, True


def phase_for_chunk(
    start: int,
    end: int,
    closed: np.ndarray,
    close_event: np.ndarray,
    release_event: np.ndarray,
    horizon: int,
    clear_gripper: bool,
) -> str:
    if not clear_gripper:
        return "ambiguous"
    has_close = bool(close_event[start:end].any())
    has_release = bool(release_event[start:end].any())
    if has_close and has_release:
        return "ambiguous"
    if has_close:
        return "closing"
    if has_release:
        return "release"
    state_before = bool(closed[start - 1] if start > 0 else closed[start])
    if state_before:
        return "hold"
    next_end = min(end + horizon, len(close_event))
    if bool(close_event[end:next_end].any()):
        return "pre_close"
    return "free_open"


def safe_norm_rows(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(values, axis=1)


def direction_change(steps: np.ndarray) -> float:
    if len(steps) < 2:
        return 0.0
    left = steps[:-1]
    right = steps[1:]
    denom = safe_norm_rows(left) * safe_norm_rows(right)
    valid = denom > 1e-8
    if not valid.any():
        return 0.0
    cosine = np.sum(left[valid] * right[valid], axis=1) / denom[valid]
    return float(np.mean(1.0 - np.clip(cosine, -1.0, 1.0)))


def chunk_features(
    action_joint: np.ndarray,
    current_joint: np.ndarray,
    gripper: np.ndarray,
) -> dict[str, float]:
    trajectory = np.vstack([current_joint[None, :], action_joint])
    steps = np.diff(trajectory, axis=0)
    accel = np.diff(steps, axis=0)
    jerk = np.diff(accel, axis=0)
    step_norm = safe_norm_rows(steps)
    accel_norm = safe_norm_rows(accel)
    jerk_norm = safe_norm_rows(jerk)
    path = float(step_norm.sum())
    result = {
        "joint_path": path,
        "joint_step_mean": float(step_norm.mean()),
        "joint_step_max": float(step_norm.max(initial=0.0)),
        "joint_endpoint": float(np.linalg.norm(action_joint[-1] - current_joint)),
        "joint_accel_mean": float(accel_norm.mean()) if len(accel_norm) else 0.0,
        "joint_accel_max": float(accel_norm.max(initial=0.0)),
        "joint_jerk_mean": float(jerk_norm.mean()) if len(jerk_norm) else 0.0,
        "joint_jerk_max": float(jerk_norm.max(initial=0.0)),
        "joint_direction_change": direction_change(steps),
        "joint_roughness": float(accel_norm.sum() / max(path, 1e-8)),
        "gripper_start": float(gripper[0]),
        "gripper_end": float(gripper[-1]),
        "gripper_mean": float(gripper.mean()),
        "gripper_range": float(np.ptp(gripper)),
        "gripper_tv": float(np.abs(np.diff(gripper)).sum()),
        "gripper_step_max": float(np.abs(np.diff(gripper)).max(initial=0.0)),
        "gripper_endpoint": float(abs(gripper[-1] - gripper[0])),
        "gripper_high_fraction": float(np.mean(gripper >= 0.5)),
    }
    for dim in range(7):
        result[f"joint_abs_step_d{dim}"] = float(np.mean(np.abs(steps[:, dim])))
    return result


def transition_features(
    current: dict[str, float],
    previous: dict[str, float] | None,
    current_joint: np.ndarray,
    previous_action_joint: np.ndarray | None,
    current_gripper: np.ndarray,
    previous_gripper: np.ndarray | None,
) -> dict[str, float]:
    if previous is None or previous_action_joint is None or previous_gripper is None:
        result = {f"trans_delta_{key}": 0.0 for key in JOINT_FEATURES + GRIPPER_FEATURES}
        result.update(
            {
                "trans_joint_boundary": 0.0,
                "trans_gripper_boundary": 0.0,
                "trans_endpoint_direction_change": 0.0,
                "trans_has_previous": 0.0,
            }
        )
        return result

    result = {
        f"trans_delta_{key}": float(current[key] - previous[key])
        for key in JOINT_FEATURES + GRIPPER_FEATURES
    }
    current_vector = current_joint[-1] - current_joint[0]
    previous_vector = previous_action_joint[-1] - previous_action_joint[0]
    denominator = float(np.linalg.norm(current_vector) * np.linalg.norm(previous_vector))
    cosine = float(np.dot(current_vector, previous_vector) / denominator) if denominator > 1e-8 else 1.0
    result.update(
        {
            "trans_joint_boundary": float(
                np.linalg.norm(current_joint[0] - previous_action_joint[-1])
            ),
            "trans_gripper_boundary": float(abs(current_gripper[0] - previous_gripper[-1])),
            "trans_endpoint_direction_change": float(1.0 - np.clip(cosine, -1.0, 1.0)),
            "trans_has_previous": 1.0,
        }
    )
    return result


def extract_episode(dataset: Path, episode: int, config: Config, joint_range: np.ndarray) -> list[dict]:
    frame = pd.read_parquet(
        episode_path(dataset, episode), columns=["observation.state", "action", "frame_index"]
    )
    actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
    states = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
    joints = actions[:, 14:21] / joint_range
    state_joints = states[:, 7:14] / joint_range
    gripper_raw = actions[:, 12]
    gripper, closed, close_event, release_event, clear_gripper = hysteresis_gripper(
        gripper_raw, config.min_gripper_span
    )

    rows: list[dict] = []
    previous_features: dict[str, float] | None = None
    previous_joint: np.ndarray | None = None
    previous_gripper: np.ndarray | None = None
    chunk_index = 0
    for start in range(config.first_anchor, len(actions) - config.chunk_horizon + 1, config.chunk_horizon):
        end = start + config.chunk_horizon
        action_joint = joints[start:end]
        current_state_joint = state_joints[start]
        current_gripper = gripper[start:end]
        features = chunk_features(action_joint, current_state_joint, current_gripper)
        transition = transition_features(
            features,
            previous_features,
            action_joint,
            previous_joint,
            current_gripper,
            previous_gripper,
        )
        stage = phase_for_chunk(
            start,
            end,
            closed,
            close_event,
            release_event,
            config.chunk_horizon,
            clear_gripper,
        )
        rows.append(
            {
                "episode": episode,
                "chunk_index": chunk_index,
                "anchor": start,
                "stage": stage,
                "fine": int(stage in FINE_STAGES),
                "clear_gripper": int(clear_gripper),
                "raw_gripper_q05": float(np.quantile(gripper_raw, 0.05)),
                "raw_gripper_q95": float(np.quantile(gripper_raw, 0.95)),
                **features,
                **transition,
            }
        )
        previous_features = features
        previous_joint = action_joint
        previous_gripper = current_gripper
        chunk_index += 1
    return rows


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    transition = sorted(column for column in columns if column.startswith("trans_"))
    return {
        "joint_only": JOINT_FEATURES,
        "gripper_only": GRIPPER_FEATURES,
        "joint_gripper": JOINT_FEATURES + GRIPPER_FEATURES,
        "joint_gripper_transition": JOINT_FEATURES + GRIPPER_FEATURES + transition,
    }


def build_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        C=1.0,
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def grouped_oof_binary(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = data[target].to_numpy(dtype=int)
    groups = data["episode"].to_numpy(dtype=int)
    x = data[features].to_numpy(dtype=np.float64)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probability = np.full(len(data), np.nan, dtype=np.float64)
    prediction = np.full(len(data), -1, dtype=int)
    for train, test in splitter.split(x, y, groups):
        model = build_model(model_name, seed)
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
        prediction[test] = model.predict(x[test])
    if np.isnan(probability).any() or (prediction < 0).any():
        raise RuntimeError("OOF prediction is incomplete")
    return probability, prediction


def metric_values(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
    }


def bootstrap_binary_metrics(
    data: pd.DataFrame,
    probability: np.ndarray,
    prediction: np.ndarray,
    target: str,
    reps: int,
    seed: int,
) -> dict[str, list[float]]:
    groups = data["episode"].to_numpy(dtype=int)
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }
    y_all = data[target].to_numpy(dtype=int)
    for _ in range(reps):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample_index = np.concatenate([indices[group] for group in sampled_groups])
        y = y_all[sample_index]
        if len(np.unique(y)) < 2:
            continue
        values = metric_values(y, probability[sample_index], prediction[sample_index])
        for key, value in values.items():
            samples[key].append(value)
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in samples.items()
    }


def bootstrap_paired_delta(
    data: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
    target: str,
    reps: int,
    seed: int,
) -> dict[str, object]:
    groups = data["episode"].to_numpy(dtype=int)
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    y_all = data[target].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    deltas = {"auroc": [], "auprc": []}
    for _ in range(reps):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample_index = np.concatenate([indices[group] for group in sampled_groups])
        y = y_all[sample_index]
        if len(np.unique(y)) < 2:
            continue
        deltas["auroc"].append(roc_auc_score(y, right[sample_index]) - roc_auc_score(y, left[sample_index]))
        deltas["auprc"].append(
            average_precision_score(y, right[sample_index])
            - average_precision_score(y, left[sample_index])
        )
    result: dict[str, object] = {}
    for key, values in deltas.items():
        result[key] = {
            "point": float(
                (roc_auc_score if key == "auroc" else average_precision_score)(y_all, right)
                - (roc_auc_score if key == "auroc" else average_precision_score)(y_all, left)
            ),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
    return result


def evaluate_binary_task(
    name: str,
    data: pd.DataFrame,
    target: str,
    sets: dict[str, list[str]],
    config: Config,
) -> tuple[list[dict], list[pd.DataFrame], dict[str, object]]:
    metrics: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    probabilities: dict[tuple[str, str], np.ndarray] = {}
    model_specs = [("logistic", key) for key in sets]
    model_specs.append(("random_forest", "joint_gripper_transition"))
    for model_name, feature_name in model_specs:
        probability, prediction = grouped_oof_binary(
            data, sets[feature_name], target, model_name, config.seed
        )
        probabilities[(model_name, feature_name)] = probability
        point = metric_values(data[target].to_numpy(dtype=int), probability, prediction)
        ci = bootstrap_binary_metrics(
            data,
            probability,
            prediction,
            target,
            config.bootstrap_reps,
            config.seed + len(metrics),
        )
        metrics.append(
            {
                "task": name,
                "model": model_name,
                "feature_group": feature_name,
                "samples": int(len(data)),
                "episodes": int(data["episode"].nunique()),
                **point,
                **{f"{key}_ci_low": value[0] for key, value in ci.items()},
                **{f"{key}_ci_high": value[1] for key, value in ci.items()},
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "task": name,
                    "model": model_name,
                    "feature_group": feature_name,
                    "episode": data["episode"].to_numpy(),
                    "anchor": data["anchor"].to_numpy(),
                    "stage": data["stage"].to_numpy(),
                    "target": data[target].to_numpy(dtype=int),
                    "probability": probability,
                    "prediction": prediction,
                }
            )
        )

    paired = bootstrap_paired_delta(
        data,
        probabilities[("logistic", "gripper_only")],
        probabilities[("logistic", "joint_gripper_transition")],
        target,
        config.bootstrap_reps,
        config.seed + 100,
    )
    return metrics, prediction_frames, paired


def grouped_multiclass(
    data: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = sorted(data["stage"].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    y = data["stage"].map(label_to_id).to_numpy(dtype=int)
    groups = data["episode"].to_numpy(dtype=int)
    x = data[features].to_numpy(dtype=np.float64)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probability = np.full((len(data), len(labels)), np.nan, dtype=np.float64)
    prediction = np.full(len(data), -1, dtype=int)
    for train, test in splitter.split(x, y, groups):
        model = build_model("logistic", seed)
        model.fit(x[train], y[train])
        fold_probability = model.predict_proba(x[test])
        classes = model.named_steps["model"].classes_.astype(int)
        probability[np.ix_(test, classes)] = fold_probability
        prediction[test] = model.predict(x[test])
    if np.isnan(probability).any():
        raise RuntimeError("Multiclass OOF prediction is incomplete")
    return probability, prediction, labels


def stage_summary(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    valid = frame[frame["stage"].isin(STAGES)]
    for feature in feature_names:
        rho, p_value = spearmanr(valid[feature], valid["fine"])
        for stage in STAGES:
            values = valid.loc[valid["stage"] == stage, feature].to_numpy(dtype=float)
            rows.append(
                {
                    "feature": feature,
                    "stage": stage,
                    "count": int(len(values)),
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "q25": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                    "q75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                    "fine_spearman": float(rho),
                    "fine_spearman_p": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    config: Config,
    chunks: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: dict[str, object],
    multiclass: dict[str, object],
) -> None:
    counts = chunks["stage"].value_counts().to_dict()
    preclose = metrics[(metrics.task == "pre_close_vs_free") & (metrics.model == "logistic")]
    fine = metrics[(metrics.task == "fine_vs_free") & (metrics.model == "logistic")]

    def row(table: pd.DataFrame, group: str) -> pd.Series:
        return table[table.feature_group == group].iloc[0]

    lines = [
        "# H1 GT 动作 chunk 间阶段实验结果",
        "",
        f"- episode 数：{config.episodes}",
        f"- 全部 chunk 数：{len(chunks)}",
        f"- 可用阶段 chunk 数：{int(chunks.stage.isin(STAGES).sum())}",
        f"- 阶段计数：`{json.dumps(counts, ensure_ascii=False)}`",
        "",
        "## 主结果",
        "",
        "| 任务 | 特征 | AUROC | AUPRC | Balanced Acc. |",
        "|---|---|---:|---:|---:|",
    ]
    for task_name, table in [("fine vs free", fine), ("pre-close vs free", preclose)]:
        for group in ["joint_only", "gripper_only", "joint_gripper", "joint_gripper_transition"]:
            value = row(table, group)
            lines.append(
                f"| {task_name} | {group} | {value.auroc:.4f} | {value.auprc:.4f} | {value.balanced_accuracy:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Transition 相对 gripper-only 的配对增量",
            "",
            f"- fine vs free：`{json.dumps(paired['fine_vs_free'], ensure_ascii=False)}`",
            f"- pre-close vs free：`{json.dumps(paired['pre_close_vs_free'], ensure_ascii=False)}`",
            "",
            "## 五分类",
            "",
            f"- macro-F1：{multiclass['macro_f1']:.4f}",
            f"- balanced accuracy：{multiclass['balanced_accuracy']:.4f}",
            "",
            "## 解释限制",
            "",
            "该实验的阶段由 GT 夹爪构造，因此 gripper-only 的 fine/free 结果主要是 sanity check。真正检验师兄想法的是 pre-close held-out 结果、transition 的增量，以及后续 WAM 配对实验中的 compute-benefit 预测。这里不能得出任何“应该多算”的结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    config = Config(
        dataset=args.dataset,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        chunk_horizon=args.chunk_horizon,
        first_anchor=args.first_anchor,
        bootstrap_reps=args.bootstrap_reps,
        min_gripper_span=args.min_gripper_span,
    )
    repo = Path(__file__).resolve().parents[3]
    dataset = Path(config.dataset)
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_episodes(
        dataset,
        config.episodes,
        config.seed,
        config.first_anchor + 2 * config.chunk_horizon,
    )
    joint_range = load_joint_ranges(dataset)
    rows: list[dict] = []
    for index, episode in enumerate(selected, start=1):
        rows.extend(extract_episode(dataset, episode, config, joint_range))
        if index % 100 == 0:
            print(f"processed {index}/{len(selected)} episodes", flush=True)
    chunks = pd.DataFrame(rows)
    sets = feature_sets(chunks.columns.tolist())

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "selected_episodes": selected,
        "joint_q99_minus_q01": joint_range.tolist(),
        "git": git_info(repo),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "feature_sets": sets,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    chunks.to_parquet(output_dir / "chunks.parquet", index=False)

    valid = chunks[chunks["stage"].isin(STAGES)].reset_index(drop=True)
    fine_data = valid.copy()
    preclose_data = valid[valid["stage"].isin(["free_open", "pre_close"])].copy().reset_index(drop=True)
    preclose_data["pre_close_target"] = (preclose_data["stage"] == "pre_close").astype(int)

    all_metrics: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    fine_metrics, fine_predictions, fine_delta = evaluate_binary_task(
        "fine_vs_free", fine_data, "fine", sets, config
    )
    all_metrics.extend(fine_metrics)
    prediction_frames.extend(fine_predictions)
    pre_metrics, pre_predictions, pre_delta = evaluate_binary_task(
        "pre_close_vs_free", preclose_data, "pre_close_target", sets, config
    )
    all_metrics.extend(pre_metrics)
    prediction_frames.extend(pre_predictions)

    full_features = sets["joint_gripper_transition"]
    multi_probability, multi_prediction, labels = grouped_multiclass(valid, full_features, config.seed)
    true_ids = valid["stage"].map({label: i for i, label in enumerate(labels)}).to_numpy(dtype=int)
    multiclass = {
        "labels": labels,
        "samples": int(len(valid)),
        "episodes": int(valid["episode"].nunique()),
        "macro_f1": float(f1_score(true_ids, multi_prediction, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(true_ids, multi_prediction)),
        "ovr_macro_auroc": float(roc_auc_score(true_ids, multi_probability, multi_class="ovr", average="macro")),
    }
    multi_frame = valid[["episode", "anchor", "stage"]].copy()
    multi_frame["prediction"] = [labels[index] for index in multi_prediction]
    for index, label in enumerate(labels):
        multi_frame[f"probability_{label}"] = multi_probability[:, index]
    multi_frame.to_parquet(output_dir / "multiclass_predictions.parquet", index=False)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        output_dir / "binary_predictions.parquet", index=False
    )
    summary = stage_summary(valid, JOINT_FEATURES + GRIPPER_FEATURES + sets["joint_gripper_transition"][-4:])
    summary.to_csv(output_dir / "stage_summary.csv", index=False)
    paired = {"fine_vs_free": fine_delta, "pre_close_vs_free": pre_delta}
    (output_dir / "paired_deltas.json").write_text(json.dumps(paired, indent=2, ensure_ascii=False))
    (output_dir / "multiclass_metrics.json").write_text(
        json.dumps(multiclass, indent=2, ensure_ascii=False)
    )
    write_report(output_dir / "report.md", config, chunks, metrics, paired, multiclass)
    print((output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
