#!/usr/bin/env python3
"""Analyze causal prefix-stop traces from full DreamZero denoising trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_gt_chunk_phase import (
    GRIPPER_FEATURES,
    JOINT_FEATURES,
    chunk_features,
    transition_features,
)


CHECKPOINTS = [2, 4, 8]
STAGES = ["free_open", "pre_close", "closing", "hold", "release"]
TOLERANCES = [0.0, 0.002, 0.005]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="research/action_chunk_compute/experiments/h3_nested_provisional_compute/results/prefix_trace_12",
    )
    parser.add_argument("--dataset", default="/home/admin/.cache/DreamZero-DROID-Data")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def action_ranges(dataset: Path) -> tuple[np.ndarray, float]:
    stats = json.loads((dataset / "meta/stats.json").read_text())["action"]
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    return np.maximum(q99[14:21] - q01[14:21], 1e-6), float(
        max(q99[12] - q01[12], 1e-6)
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denominator <= 1e-12:
        return 1.0
    return float(np.dot(left_flat, right_flat) / denominator)


def build_samples(input_dir: Path, joint_range: np.ndarray, gripper_range: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prefix16_max_diff = 0.0
    for path in sorted(input_dir.glob("episode_*.npz")):
        trace = np.load(path)
        episode = int(trace["episode"])
        previous: dict[int, tuple[dict[str, float], np.ndarray, np.ndarray]] = {}
        prefix16_max_diff = max(
            prefix16_max_diff,
            float(np.max(np.abs(trace["prefix_actions"][:, -1] - trace["final_actions"]))),
        )
        for sample_index, anchor in enumerate(trace["anchors"]):
            target = trace["gt_actions"][sample_index].astype(np.float64)
            state = trace["current_states"][sample_index].astype(np.float64)
            prefixes = trace["prefix_actions"][sample_index].astype(np.float64)
            flows = trace["action_flows"][sample_index, :, :, :8].astype(np.float64)
            row: dict[str, object] = {
                "episode": episode,
                "anchor": int(anchor),
                "chunk_index": sample_index,
                "stage": str(trace["stages"][sample_index]),
                "fine": int(trace["fine"][sample_index]),
                "latency": float(trace["latencies"][sample_index]),
            }

            current_joint = state[:7]
            current_gripper = float(state[7])
            persistence_joint = np.repeat(current_joint[None, :], 24, axis=0)
            persistence_gripper = np.repeat(current_gripper, 24)
            for horizon in [8, 24]:
                row[f"persistence_joint_error_h{horizon}"] = float(
                    np.mean(np.abs(persistence_joint[:horizon] - target[:horizon, :7]) / joint_range)
                )
                row[f"persistence_gripper_error_h{horizon}"] = float(
                    np.mean(np.abs(persistence_gripper[:horizon] - target[:horizon, 7]) / gripper_range)
                )

            gt_trajectory = np.vstack([current_joint, target[:8, :7]]) / joint_range
            row["gt_motion_h8"] = float(np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1).mean())

            for stop_index in range(16):
                calls = stop_index + 1
                prediction = prefixes[stop_index]
                for horizon in [8, 24]:
                    row[f"e{calls}_joint_h{horizon}"] = float(
                        np.mean(np.abs(prediction[:horizon, :7] - target[:horizon, :7]) / joint_range)
                    )
                    row[f"e{calls}_gripper_h{horizon}"] = float(
                        np.mean(np.abs(prediction[:horizon, 7] - target[:horizon, 7]) / gripper_range)
                    )
                pred_trajectory = np.vstack([current_joint, prediction[:8, :7]]) / joint_range
                row[f"k{calls}_pred_motion_h8"] = float(
                    np.linalg.norm(np.diff(pred_trajectory, axis=0), axis=1).mean()
                )

                if calls not in CHECKPOINTS:
                    continue
                joint = prediction[:, :7] / joint_range
                gripper = prediction[:, 7] / gripper_range
                state_joint = current_joint / joint_range
                geometry = chunk_features(joint, state_joint, gripper)
                prior = previous.get(calls)
                transition = transition_features(
                    geometry,
                    prior[0] if prior else None,
                    joint,
                    prior[1] if prior else None,
                    gripper,
                    prior[2] if prior else None,
                )
                for key, value in geometry.items():
                    row[f"k{calls}_{key}"] = value
                for key, value in transition.items():
                    row[f"k{calls}_{key}"] = value
                previous[calls] = (geometry, joint, gripper)

                flow = flows[stop_index]
                previous_flow = flows[stop_index - 1]
                flow_delta = flow - previous_flow
                row[f"k{calls}_flow_cosine"] = cosine(flow, previous_flow)
                row[f"k{calls}_flow_norm"] = float(np.linalg.norm(flow) / np.sqrt(flow.size))
                row[f"k{calls}_flow_delta_l2"] = float(
                    np.linalg.norm(flow_delta) / np.sqrt(flow_delta.size)
                )
                row[f"k{calls}_flow_relative_l2"] = float(
                    np.linalg.norm(flow_delta) / max(np.linalg.norm(previous_flow), 1e-12)
                )

                previous_prefix = prefixes[stop_index - 1]
                joint_delta = np.abs(prediction[:, :7] - previous_prefix[:, :7]) / joint_range
                gripper_delta = np.abs(prediction[:, 7] - previous_prefix[:, 7]) / gripper_range
                row[f"k{calls}_prefix_joint_delta_mean"] = float(joint_delta.mean())
                row[f"k{calls}_prefix_joint_delta_max"] = float(joint_delta.max())
                row[f"k{calls}_prefix_gripper_delta_mean"] = float(gripper_delta.mean())
                row[f"k{calls}_prefix_gripper_delta_max"] = float(gripper_delta.max())

            rows.append(row)

    if not rows:
        raise RuntimeError(f"No episode trace files found in {input_dir}")
    if prefix16_max_diff > 1e-6:
        raise RuntimeError(f"prefix-16 invariant failed: max_abs_diff={prefix16_max_diff}")
    samples = pd.DataFrame(rows).sort_values(["episode", "anchor"]).reset_index(drop=True)
    for calls in CHECKPOINTS:
        samples[f"benefit_k{calls}_to_16"] = (
            samples[f"e{calls}_joint_h8"] - samples["e16_joint_h8"]
        )
        combined = feature_sets(samples, calls)["combined"]
        lagged = samples.groupby("episode", sort=False)[combined].shift(1)
        lagged.columns = [f"lag_{column}" for column in combined]
        samples = pd.concat([samples, lagged], axis=1)
    samples.attrs["prefix16_max_diff"] = prefix16_max_diff
    return samples


def feature_sets(samples: pd.DataFrame, calls: int) -> dict[str, list[str]]:
    flow = [
        f"k{calls}_flow_cosine",
        f"k{calls}_flow_norm",
        f"k{calls}_flow_delta_l2",
        f"k{calls}_flow_relative_l2",
    ]
    convergence = [
        f"k{calls}_prefix_joint_delta_mean",
        f"k{calls}_prefix_joint_delta_max",
        f"k{calls}_prefix_gripper_delta_mean",
        f"k{calls}_prefix_gripper_delta_max",
    ]
    geometry = [f"k{calls}_{feature}" for feature in JOINT_FEATURES + GRIPPER_FEATURES]
    transition = sorted(column for column in samples if column.startswith(f"k{calls}_trans_"))
    combined = flow + convergence + geometry + transition
    return {
        "flow_only": flow,
        "prefix_convergence": convergence,
        "action_geometry": geometry,
        "chunk_transition": transition,
        "combined": combined,
        "lagged_previous_chunk": [f"lag_{column}" for column in combined],
    }


def bootstrap_episode_mean(
    frame: pd.DataFrame, column: str, reps: int, seed: int
) -> tuple[float, list[float]]:
    episode_means = frame.groupby("episode")[column].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(episode_means, len(episode_means), replace=True).mean() for _ in range(reps)]
    )
    return float(episode_means.mean()), [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def quality_curve(samples: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for calls in range(1, 17):
        for metric in ["joint_h8", "joint_h24", "gripper_h8", "gripper_h24"]:
            column = f"e{calls}_{metric}"
            mean, ci = bootstrap_episode_mean(samples, column, reps, seed + calls)
            rows.append(
                {
                    "calls": calls,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "median": float(samples[column].median()),
                }
            )
    return pd.DataFrame(rows)


def stage_benefits(samples: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for calls in CHECKPOINTS:
        target = f"benefit_k{calls}_to_16"
        for stage in STAGES:
            subset = samples[samples["stage"] == stage]
            mean, ci = bootstrap_episode_mean(subset, target, reps, seed + calls)
            rows.append(
                {
                    "calls": calls,
                    "stage": stage,
                    "samples": len(subset),
                    "mean_benefit": mean,
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "positive_fraction": float((subset[target] > 0).mean()),
                    "gt_0.002_fraction": float((subset[target] > 0.002).mean()),
                }
            )
    return pd.DataFrame(rows)


def oracle_summary(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = samples[[f"e{k}_joint_h8" for k in range(1, 17)]].to_numpy()
    overall = []
    by_stage = []
    for tolerance in TOLERANCES:
        earliest = np.asarray(
            [
                next(k + 1 for k, value in enumerate(row) if value <= row[-1] + tolerance)
                for row in errors
            ]
        )
        overall.append(
            {
                "tolerance": tolerance,
                "mean_calls": float(earliest.mean()),
                "median_calls": float(np.median(earliest)),
                "fraction_le_2": float(np.mean(earliest <= 2)),
                "fraction_le_4": float(np.mean(earliest <= 4)),
                "fraction_le_8": float(np.mean(earliest <= 8)),
                "fraction_gt_8": float(np.mean(earliest > 8)),
            }
        )
        for stage in STAGES:
            mask = samples["stage"].to_numpy() == stage
            by_stage.append(
                {
                    "tolerance": tolerance,
                    "stage": stage,
                    "mean_calls": float(earliest[mask].mean()),
                    "median_calls": float(np.median(earliest[mask])),
                }
            )
    return pd.DataFrame(overall), pd.DataFrame(by_stage)


def ridge_oof(frame: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    x = frame[features].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    groups = frame["episode"].to_numpy(dtype=int)
    prediction = np.full(len(frame), np.nan)
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        )
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    return prediction


def compute_prediction(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_columns = pd.get_dummies(samples["stage"], prefix="phase", dtype=float)
    frame = pd.concat([samples, phase_columns], axis=1)
    metrics = []
    prediction_rows = []
    for calls in CHECKPOINTS:
        target = f"benefit_k{calls}_to_16"
        sets = feature_sets(frame, calls)
        sets["gt_phase_oracle"] = phase_columns.columns.tolist()
        y = frame[target].to_numpy(dtype=float)
        for name, features in sets.items():
            prediction = ridge_oof(frame, features, target)
            rho, p_value = spearmanr(y, prediction)
            row: dict[str, object] = {
                "calls": calls,
                "feature_group": name,
                "mae": float(mean_absolute_error(y, prediction)),
                "r2": float(r2_score(y, prediction)),
                "spearman": float(rho),
                "spearman_p": float(p_value),
            }
            for tolerance in TOLERANCES:
                label = y > tolerance
                suffix = str(tolerance).replace(".", "p")
                row[f"positive_fraction_{suffix}"] = float(label.mean())
                if len(np.unique(label)) == 2:
                    row[f"auroc_{suffix}"] = float(roc_auc_score(label, prediction))
                    row[f"auprc_{suffix}"] = float(average_precision_score(label, prediction))
                else:
                    row[f"auroc_{suffix}"] = np.nan
                    row[f"auprc_{suffix}"] = np.nan
            metrics.append(row)
            prediction_rows.extend(
                {
                    "episode": int(frame.iloc[index]["episode"]),
                    "anchor": int(frame.iloc[index]["anchor"]),
                    "stage": frame.iloc[index]["stage"],
                    "calls": calls,
                    "feature_group": name,
                    "actual": float(y[index]),
                    "prediction": float(prediction[index]),
                }
                for index in range(len(frame))
            )
    return pd.DataFrame(metrics), pd.DataFrame(prediction_rows)


def feature_correlations(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calls in CHECKPOINTS:
        target = f"benefit_k{calls}_to_16"
        for feature in feature_sets(samples, calls)["combined"]:
            rho, p_value = spearmanr(samples[feature], samples[target], nan_policy="omit")
            rows.append(
                {
                    "calls": calls,
                    "feature": feature,
                    "spearman": float(rho),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["calls", "spearman"], key=lambda values: values.abs() if values.name == "spearman" else values,
        ascending=[True, False],
    )


def motion_summary(samples: pd.DataFrame) -> pd.DataFrame:
    frame = samples.copy()
    frame["motion_bin"] = pd.qcut(frame["gt_motion_h8"], 3, labels=["low", "mid", "high"])
    rows = []
    for calls in CHECKPOINTS:
        target = f"benefit_k{calls}_to_16"
        for motion_bin, subset in frame.groupby("motion_bin", observed=True):
            rows.append(
                {
                    "calls": calls,
                    "motion_bin": str(motion_bin),
                    "samples": len(subset),
                    "mean_benefit": float(subset[target].mean()),
                    "positive_fraction": float((subset[target] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    samples: pd.DataFrame,
    curve: pd.DataFrame,
    stages: pd.DataFrame,
    oracle: pd.DataFrame,
    prediction: pd.DataFrame,
    motion: pd.DataFrame,
) -> None:
    joint_curve = curve[curve.metric == "joint_h8"].set_index("calls")
    lines = [
        "# H3 同一去噪轨迹 prefix-stop 因果实验",
        "",
        f"- episode：{samples.episode.nunique()}",
        f"- 连续 chunk：{len(samples)}",
        f"- prefix-16 与正常 final 最大差：{samples.attrs['prefix16_max_diff']:.1g}",
        "",
        "## Quality-compute 曲线",
        "",
        "| Calls | H8 joint MAE | 95% CI | Pred. motion H8 |",
        "|---:|---:|---:|---:|",
    ]
    for calls in [1, 2, 4, 8, 12, 16]:
        row = joint_curve.loc[calls]
        lines.append(
            f"| {calls} | {row['mean']:.5f} | [{row.ci95_low:.5f}, {row.ci95_high:.5f}] | "
            f"{samples[f'k{calls}_pred_motion_h8'].mean():.5f} |"
        )
    lines.extend(
        [
            "",
            f"Persistence baseline H8 joint MAE：{samples.persistence_joint_error_h8.mean():.5f}",
            f"GT motion H8：{samples.gt_motion_h8.mean():.5f}",
            "",
            "## 阶段额外计算收益（E_k - E_16）",
            "",
            "| 阶段 | 2→16 | 4→16 | 8→16 |",
            "|---|---:|---:|---:|",
        ]
    )
    for stage in STAGES:
        values = [
            stages[(stages.calls == calls) & (stages.stage == stage)].iloc[0].mean_benefit
            for calls in CHECKPOINTS
        ]
        lines.append(f"| {stage} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} |")
    lines.extend(
        [
            "",
            "## Full-16 非劣的最早 prefix oracle",
            "",
            "| tolerance | Mean calls | <=2 | <=4 | <=8 | >8 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in oracle.itertuples(index=False):
        lines.append(
            f"| {row.tolerance:.3f} | {row.mean_calls:.2f} | {row.fraction_le_2:.1%} | "
            f"{row.fraction_le_4:.1%} | {row.fraction_le_8:.1%} | {row.fraction_gt_8:.1%} |"
        )
    lines.extend(["", "## Held-out compute-benefit 预测", ""])
    for calls in CHECKPOINTS:
        subset = prediction[prediction.calls == calls]
        for group in ["flow_only", "prefix_convergence", "action_geometry", "chunk_transition", "combined", "lagged_previous_chunk", "gt_phase_oracle"]:
            row = subset[subset.feature_group == group].iloc[0]
            lines.append(
                f"- k={calls} / `{group}`：Spearman {row.spearman:.3f}，R2 {row.r2:.3f}，"
                f"benefit>0.002 AUROC {row.auroc_0p002:.3f}。"
            )
    high_motion = motion[(motion.calls == 2) & (motion.motion_bin == "high")].iloc[0]
    lines.extend(
        [
            "",
            "## 运动幅度敏感性",
            "",
            f"GT motion 最高三分位中，2→16 mean benefit 为 {high_motion.mean_benefit:.5f}，"
            f"正收益比例 {high_motion.positive_fraction:.1%}。",
            "",
            "## 判定",
            "",
            "动作输出可以识别自由移动与精细操作阶段，但阶段、动作几何、chunk 间变化和 GT phase oracle 均未证明能够预测继续执行更多 DiT 的收益。当前离线 MAE 甚至随调用数增加而上升，因此不能据此实现“精细阶段多算”。",
            "",
            "## 限制",
            "",
            "DROID 只有成功示范，离线单轨迹 MAE 会惩罚不同但同样可行的多模态动作，也可能偏好保守动作。prefix-stop 结论足以否定当前 MAE 下的阶段门控依据，但不能证明 1-2 call 闭环成功率优于 full-16；上线前仍需仿真或真机闭环评估。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = repo / input_dir
    output_dir = input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_range, gripper_range = action_ranges(Path(args.dataset))
    samples = build_samples(input_dir, joint_range, gripper_range)
    curve = quality_curve(samples, args.bootstrap_reps, args.seed)
    stages = stage_benefits(samples, args.bootstrap_reps, args.seed)
    oracle, oracle_stages = oracle_summary(samples)
    prediction, prediction_rows = compute_prediction(samples)
    correlations = feature_correlations(samples)
    motion = motion_summary(samples)

    samples.to_parquet(output_dir / "samples.parquet", index=False)
    curve.to_csv(output_dir / "quality_curve.csv", index=False)
    stages.to_csv(output_dir / "stage_benefits.csv", index=False)
    oracle.to_csv(output_dir / "oracle_prefix_summary.csv", index=False)
    oracle_stages.to_csv(output_dir / "oracle_prefix_by_stage.csv", index=False)
    prediction.to_csv(output_dir / "compute_prediction_metrics.csv", index=False)
    prediction_rows.to_parquet(output_dir / "compute_predictions.parquet", index=False)
    correlations.to_csv(output_dir / "feature_correlations.csv", index=False)
    motion.to_csv(output_dir / "motion_sensitivity.csv", index=False)
    write_report(
        output_dir / "report.md",
        samples,
        curve,
        stages,
        oracle,
        prediction,
        motion,
    )
    print((output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
