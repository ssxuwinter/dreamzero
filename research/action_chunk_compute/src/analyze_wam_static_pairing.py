#!/usr/bin/env python3
"""Analyze paired static-schedule WAM predictions and compute-benefit proxies."""

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
    bootstrap_binary_metrics,
    chunk_features,
    grouped_oof_binary,
    metric_values,
    transition_features,
)


SCHEDULES = [5, 8, 16]
STAGES = ["free_open", "pre_close", "closing", "hold", "release"]
TOLERANCES = [0.0, 0.002, 0.005]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="research/action_chunk_compute/experiments/h2_wam_static_compute/results/diagnostic_contiguous_12/raw_predictions.jsonl",
    )
    parser.add_argument("--dataset", default="/home/admin/.cache/DreamZero-DROID-Data")
    parser.add_argument("--run-rep", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def action_ranges(dataset: Path) -> tuple[np.ndarray, float]:
    stats = json.loads((dataset / "meta/stats.json").read_text())["action"]
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    joint_range = np.maximum(q99[14:21] - q01[14:21], 1e-6)
    gripper_range = float(max(q99[12] - q01[12], 1e-6))
    return joint_range, gripper_range


def load_records(path: Path, run_rep: int) -> dict[tuple[int, int, int], dict]:
    records: dict[tuple[int, int, int], dict] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["record_type"] != "chunk" or int(row["run_rep"]) != run_rep:
                continue
            key = (int(row["episode"]), int(row["anchor"]), int(row["schedule"]))
            if key in records:
                raise RuntimeError(f"Duplicate record {key}")
            records[key] = row
    if not records:
        raise RuntimeError(f"No chunk records found in {path}")
    return records


def error_metrics(prediction: np.ndarray, target: np.ndarray, joint_range: np.ndarray, gripper_range: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for horizon in [1, 8, 24]:
        joint_error = np.abs(prediction[:horizon, :7] - target[:horizon, :7])
        gripper_error = np.abs(prediction[:horizon, 7] - target[:horizon, 7])
        result[f"joint_raw_mae_h{horizon}"] = float(joint_error.mean())
        result[f"joint_norm_mae_h{horizon}"] = float((joint_error / joint_range).mean())
        result[f"joint_norm_max_h{horizon}"] = float((joint_error / joint_range).max())
        result[f"gripper_raw_mae_h{horizon}"] = float(gripper_error.mean())
        result[f"gripper_norm_mae_h{horizon}"] = float((gripper_error / gripper_range).mean())
    return result


def build_samples(
    records: dict[tuple[int, int, int], dict],
    joint_range: np.ndarray,
    gripper_range: float,
) -> pd.DataFrame:
    sample_keys = sorted({(episode, anchor) for episode, anchor, _ in records})
    rows: list[dict] = []
    previous: dict[tuple[int, int], tuple[dict[str, float], np.ndarray, np.ndarray]] = {}
    for episode, anchor in sample_keys:
        missing = [schedule for schedule in SCHEDULES if (episode, anchor, schedule) not in records]
        if missing:
            raise RuntimeError(f"Missing schedules {missing} for episode={episode}, anchor={anchor}")
        base = records[(episode, anchor, 5)]
        state = np.asarray(base["current_state"], dtype=np.float64)
        target = np.asarray(base["gt_action"], dtype=np.float64)
        row: dict[str, object] = {
            "episode": episode,
            "anchor": anchor,
            "chunk_index": int(base["chunk_index"]),
            "stage": base["stage"],
            "fine": int(base["fine"]),
        }
        for schedule in SCHEDULES:
            record = records[(episode, anchor, schedule)]
            prediction = np.asarray(record["prediction"], dtype=np.float64)
            row[f"latency_s{schedule}"] = float(record["latency_seconds"])
            for key, value in error_metrics(prediction, target, joint_range, gripper_range).items():
                row[f"s{schedule}_{key}"] = value

            normalized_joint = prediction[:, :7] / joint_range
            normalized_state_joint = state[:7] / joint_range
            normalized_gripper = prediction[:, 7] / gripper_range
            features = chunk_features(normalized_joint, normalized_state_joint, normalized_gripper)
            prior = previous.get((episode, schedule))
            transition = transition_features(
                features,
                prior[0] if prior else None,
                normalized_joint,
                prior[1] if prior else None,
                normalized_gripper,
                prior[2] if prior else None,
            )
            for key, value in {**features, **transition}.items():
                row[f"s{schedule}_{key}"] = value
            previous[(episode, schedule)] = (features, normalized_joint, normalized_gripper)
        rows.append(row)

    samples = pd.DataFrame(rows).sort_values(["episode", "anchor"]).reset_index(drop=True)
    for low, high in [(5, 8), (5, 16), (8, 16)]:
        samples[f"benefit_{low}_to_{high}"] = (
            samples[f"s{low}_joint_norm_mae_h8"] - samples[f"s{high}_joint_norm_mae_h8"]
        )
    feature_columns = [
        column
        for column in samples.columns
        if column.startswith("s5_")
        and "mae_" not in column
        and "norm_max" not in column
        and not column.startswith("s5_latency")
    ]
    lagged = samples.groupby("episode", sort=False)[feature_columns].shift(1)
    lagged.columns = [f"lag_{column}" for column in feature_columns]
    return pd.concat([samples, lagged], axis=1)


def bootstrap_episode_mean(
    frame: pd.DataFrame,
    column: str,
    reps: int,
    seed: int,
) -> tuple[float, list[float]]:
    episode_means = frame.groupby("episode")[column].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [rng.choice(episode_means, size=len(episode_means), replace=True).mean() for _ in range(reps)]
    )
    return float(episode_means.mean()), [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def schedule_summary(samples: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for schedule in SCHEDULES:
        for metric in ["joint_norm_mae_h8", "joint_norm_mae_h24", "gripper_norm_mae_h8", "latency"]:
            column = f"latency_s{schedule}" if metric == "latency" else f"s{schedule}_{metric}"
            usable = samples.copy()
            if metric == "latency":
                usable = usable[usable["chunk_index"] > 0]
            mean, ci = bootstrap_episode_mean(usable, column, reps, seed + schedule)
            rows.append(
                {
                    "schedule": schedule,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "median": float(usable[column].median()),
                    "p95": float(usable[column].quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def benefit_summary(samples: pd.DataFrame, reps: int, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict] = []
    result: dict[str, object] = {}
    for name in ["benefit_5_to_8", "benefit_5_to_16", "benefit_8_to_16"]:
        mean, ci = bootstrap_episode_mean(samples, name, reps, seed)
        entry = {
            "mean": mean,
            "ci95": ci,
            "fraction_positive": float((samples[name] > 0).mean()),
            "fraction_gt_0.002": float((samples[name] > 0.002).mean()),
            "fraction_gt_0.005": float((samples[name] > 0.005).mean()),
        }
        result[name] = entry
        for stage in STAGES:
            subset = samples[samples["stage"] == stage]
            stage_mean, stage_ci = bootstrap_episode_mean(subset, name, reps, seed + STAGES.index(stage))
            rows.append(
                {
                    "benefit": name,
                    "stage": stage,
                    "samples": len(subset),
                    "mean": stage_mean,
                    "ci95_low": stage_ci[0],
                    "ci95_high": stage_ci[1],
                    "median": float(subset[name].median()),
                    "fraction_positive": float((subset[name] > 0).mean()),
                }
            )
    return pd.DataFrame(rows), result


def acceptable_schedule_summary(samples: pd.DataFrame) -> pd.DataFrame:
    errors = samples[[f"s{schedule}_joint_norm_mae_h8" for schedule in SCHEDULES]].to_numpy()
    rows: list[dict] = []
    for tolerance in TOLERANCES:
        best = errors.min(axis=1)
        selected = []
        for row, row_best in zip(errors, best):
            acceptable = [schedule for schedule, error in zip(SCHEDULES, row) if error <= row_best + tolerance]
            selected.append(min(acceptable))
        selected_array = np.asarray(selected)
        rows.append(
            {
                "tolerance": tolerance,
                "mean_calls": float(selected_array.mean()),
                **{f"fraction_s{schedule}": float(np.mean(selected_array == schedule)) for schedule in SCHEDULES},
            }
        )
    return pd.DataFrame(rows)


def phase_feature_sets(samples: pd.DataFrame) -> dict[str, list[str]]:
    transition = sorted(column for column in samples if column.startswith("s5_trans_"))
    current_joint = [f"s5_{feature}" for feature in JOINT_FEATURES]
    current_gripper = [f"s5_{feature}" for feature in GRIPPER_FEATURES]
    current_full = current_joint + current_gripper + transition
    lagged_full = [f"lag_{feature}" for feature in current_full]
    return {
        "current_joint": current_joint,
        "current_gripper": current_gripper,
        "current_joint_gripper_transition": current_full,
        "lagged_previous_chunk": lagged_full,
    }


def evaluate_phase(samples: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    feature_sets = phase_feature_sets(samples)
    tasks = {
        "fine_vs_free": (samples.copy(), "fine"),
        "pre_close_vs_free": (
            samples[samples["stage"].isin(["free_open", "pre_close"])].copy(),
            "pre_close_target",
        ),
    }
    tasks["pre_close_vs_free"][0]["pre_close_target"] = (
        tasks["pre_close_vs_free"][0]["stage"] == "pre_close"
    ).astype(int)
    rows: list[dict] = []
    for task, (data, target) in tasks.items():
        data = data.reset_index(drop=True)
        for feature_group, features in feature_sets.items():
            probability, prediction = grouped_oof_binary(
                data, features, target, "logistic", seed
            )
            point = metric_values(data[target].to_numpy(dtype=int), probability, prediction)
            ci = bootstrap_binary_metrics(data, probability, prediction, target, reps, seed)
            rows.append(
                {
                    "task": task,
                    "feature_group": feature_group,
                    "samples": len(data),
                    "episodes": data["episode"].nunique(),
                    **point,
                    **{f"{key}_ci_low": value[0] for key, value in ci.items()},
                    **{f"{key}_ci_high": value[1] for key, value in ci.items()},
                }
            )
    return pd.DataFrame(rows)


def ridge_oof(data: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    x = data[features].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    groups = data["episode"].to_numpy(dtype=int)
    prediction = np.full(len(data), np.nan)
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(x, y, groups):
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


def compute_prediction(samples: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = phase_feature_sets(samples)
    phase_oracle = pd.get_dummies(samples["stage"], prefix="phase", dtype=float)
    augmented = pd.concat([samples, phase_oracle], axis=1)
    sets["gt_phase_oracle"] = phase_oracle.columns.tolist()
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    y = augmented[target].to_numpy(dtype=float)
    for name, features in sets.items():
        prediction = ridge_oof(augmented, features, target)
        rho, rho_p = spearmanr(y, prediction)
        row: dict[str, object] = {
            "target": target,
            "feature_group": name,
            "mae": float(mean_absolute_error(y, prediction)),
            "r2": float(r2_score(y, prediction)),
            "spearman": float(rho),
            "spearman_p": float(rho_p),
        }
        for tolerance in TOLERANCES:
            binary = y > tolerance
            suffix = str(tolerance).replace(".", "p")
            row[f"positive_fraction_{suffix}"] = float(binary.mean())
            if len(np.unique(binary)) == 2:
                row[f"auroc_{suffix}"] = float(roc_auc_score(binary, prediction))
                row[f"auprc_{suffix}"] = float(average_precision_score(binary, prediction))
            else:
                row[f"auroc_{suffix}"] = np.nan
                row[f"auprc_{suffix}"] = np.nan
        rows.append(row)
        predictions.append(
            pd.DataFrame(
                {
                    "episode": augmented["episode"],
                    "anchor": augmented["anchor"],
                    "stage": augmented["stage"],
                    "target": target,
                    "feature_group": name,
                    "actual": y,
                    "prediction": prediction,
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def feature_correlations(samples: pd.DataFrame, target: str) -> pd.DataFrame:
    features = phase_feature_sets(samples)["current_joint_gripper_transition"]
    rows = []
    for feature in features:
        rho, p_value = spearmanr(samples[feature], samples[target])
        rows.append({"feature": feature, "target": target, "spearman": rho, "p_value": p_value})
    return pd.DataFrame(rows).sort_values("spearman", key=lambda values: values.abs(), ascending=False)


def write_report(
    path: Path,
    samples: pd.DataFrame,
    schedules: pd.DataFrame,
    benefits: pd.DataFrame,
    benefit_overall: dict[str, object],
    acceptable: pd.DataFrame,
    phase: pd.DataFrame,
    prediction: pd.DataFrame,
) -> None:
    lines = [
        "# H2 WAM 静态 5/8/16-call 连续 chunk 配对结果",
        "",
        f"- episode：{samples.episode.nunique()}",
        f"- 连续 chunk：{len(samples)}",
        f"- 阶段计数：`{json.dumps(samples.stage.value_counts().to_dict(), ensure_ascii=False)}`",
        "",
        "## Schedule 总体",
        "",
        "| Calls | Joint MAE H8 (q99 norm) | Joint MAE H24 | Gripper MAE H8 | Warm latency |",
        "|---:|---:|---:|---:|---:|",
    ]
    for schedule in SCHEDULES:
        table = schedules[schedules.schedule == schedule].set_index("metric")
        lines.append(
            f"| {schedule} | {table.loc['joint_norm_mae_h8','mean']:.5f} | "
            f"{table.loc['joint_norm_mae_h24','mean']:.5f} | "
            f"{table.loc['gripper_norm_mae_h8','mean']:.5f} | "
            f"{table.loc['latency','mean']:.2f}s |"
        )
    lines.extend(["", "## 额外计算收益", ""])
    for name, value in benefit_overall.items():
        lines.append(
            f"- `{name}`：均值 {value['mean']:.5f}，95% CI "
            f"[{value['ci95'][0]:.5f}, {value['ci95'][1]:.5f}]；"
            f"正收益比例 {value['fraction_positive']:.1%}，>0.002 比例 {value['fraction_gt_0.002']:.1%}。"
        )
    lines.extend(
        [
            "",
            "## 阶段与收益",
            "",
            "| 阶段 | 5→16 mean benefit | 正收益比例 |",
            "|---|---:|---:|",
        ]
    )
    stage_table = benefits[benefits.benefit == "benefit_5_to_16"].set_index("stage")
    for stage in STAGES:
        row = stage_table.loc[stage]
        lines.append(f"| {stage} | {row['mean']:.5f} | {row['fraction_positive']:.1%} |")
    lines.extend(["", "## 阶段识别（episode-held-out）", ""])
    for task in ["fine_vs_free", "pre_close_vs_free"]:
        subset = phase[phase.task == task]
        for feature_group in ["current_joint", "current_gripper", "current_joint_gripper_transition", "lagged_previous_chunk"]:
            row = subset[subset.feature_group == feature_group].iloc[0]
            lines.append(
                f"- `{task}` / `{feature_group}`：AUROC {row.auroc:.3f}，AUPRC {row.auprc:.3f}。"
            )
    lines.extend(["", "## Compute-benefit held-out 预测", ""])
    for row in prediction.itertuples(index=False):
        lines.append(
            f"- `{row.feature_group}`：Spearman {row.spearman:.3f}，R2 {row.r2:.3f}，"
            f"benefit>0.002 AUROC {getattr(row, 'auroc_0p002'):.3f}。"
        )
    lines.extend(
        [
            "",
            "## 最低可接受静态 schedule（oracle 诊断）",
            "",
            "| tolerance | mean calls | s5 | s8 | s16 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in acceptable.itertuples(index=False):
        lines.append(
            f"| {row.tolerance:.3f} | {row.mean_calls:.2f} | "
            f"{row.fraction_s5:.1%} | {row.fraction_s8:.1%} | {row.fraction_s16:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "5/8/16-call mask 不嵌套，本实验的 benefit 是静态 schedule-policy 差异，不是从同一中间状态继续计算的因果收益。当前最终输出可用于下一 chunk 的 lagged 调度；只有 H3 的 provisional-action/action-flow 轨迹才能决定当前请求是否早停。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = repo / input_path
    output_dir = input_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_range, gripper_range = action_ranges(Path(args.dataset))
    records = load_records(input_path, args.run_rep)
    samples = build_samples(records, joint_range, gripper_range)
    samples.to_parquet(output_dir / "samples.parquet", index=False)

    schedules = schedule_summary(samples, args.bootstrap_reps, args.seed)
    benefits, benefit_overall = benefit_summary(samples, args.bootstrap_reps, args.seed)
    acceptable = acceptable_schedule_summary(samples)
    phase = evaluate_phase(samples, min(args.bootstrap_reps, 1000), args.seed)
    compute_metrics, compute_predictions = compute_prediction(samples, "benefit_5_to_16")
    correlations = feature_correlations(samples, "benefit_5_to_16")

    schedules.to_csv(output_dir / "schedule_summary.csv", index=False)
    benefits.to_csv(output_dir / "stage_benefit_summary.csv", index=False)
    acceptable.to_csv(output_dir / "acceptable_schedules.csv", index=False)
    phase.to_csv(output_dir / "phase_metrics.csv", index=False)
    compute_metrics.to_csv(output_dir / "compute_prediction_metrics.csv", index=False)
    compute_predictions.to_parquet(output_dir / "compute_predictions.parquet", index=False)
    correlations.to_csv(output_dir / "feature_correlations.csv", index=False)
    samples[
        [
            "episode",
            "chunk_index",
            "anchor",
            "stage",
            "benefit_5_to_16",
            "s5_joint_step_mean",
            "s5_joint_accel_mean",
            "s5_joint_jerk_mean",
            "s5_gripper_step_max",
        ]
    ].to_csv(output_dir / "trajectory_table.csv", index=False)
    (output_dir / "overall_benefits.json").write_text(
        json.dumps(benefit_overall, indent=2, ensure_ascii=False)
    )
    write_report(
        output_dir / "report.md",
        samples,
        schedules,
        benefits,
        benefit_overall,
        acceptable,
        phase,
        compute_metrics,
    )
    print((output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
