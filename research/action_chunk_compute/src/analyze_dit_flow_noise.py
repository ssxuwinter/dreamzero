#!/usr/bin/env python3

from __future__ import annotations

import argparse
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

from analyze_wam_prefix_trace import CHECKPOINTS, action_ranges, build_samples

TOLERANCES = [0.0, 0.002, 0.005]
COMPLEXITY_TARGETS = [
    "gt_motion_h8",
    "fine",
    "k2_pred_motion_h8",
    "k4_pred_motion_h8",
    "k8_pred_motion_h8",
    "k2_prefix_joint_delta_mean",
    "k4_prefix_joint_delta_mean",
    "k8_prefix_joint_delta_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="research/action_chunk_compute/experiments/h3_nested_provisional_compute/results/prefix_trace_12",
    )
    parser.add_argument("--dataset", default="/home/admin/.cache/DreamZero-DROID-Data")
    parser.add_argument(
        "--output-dir",
        default="research/action_chunk_compute/experiments/h4_attention_noise_diagnostics/results/flow_noise_existing_trace",
    )
    return parser.parse_args()


def flow_noise_rows(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("episode_*.npz")):
        trace = np.load(path)
        episode = int(trace["episode"])
        anchors = trace["anchors"].astype(int)
        flows = trace["action_flows"].astype(np.float64)
        for sample_index, anchor in enumerate(anchors):
            row: dict[str, object] = {
                "episode": episode,
                "anchor": int(anchor),
                "chunk_index": sample_index,
            }
            for calls in [1, *CHECKPOINTS, 16]:
                index = calls - 1
                current = flows[sample_index, index]
                previous = flows[sample_index, max(index - 1, 0)]
                delta = current - previous
                groups = {
                    "all32": current,
                    "action8": current[:, :8],
                    "joint7": current[:, :7],
                    "gripper": current[:, 7:8],
                }
                delta_groups = {
                    "all32": delta,
                    "action8": delta[:, :8],
                    "joint7": delta[:, :7],
                    "gripper": delta[:, 7:8],
                }
                for name, value in groups.items():
                    row[f"k{calls}_dit_flow_{name}_rms"] = float(np.sqrt(np.mean(value**2)))
                    row[f"k{calls}_dit_flow_{name}_mean_abs"] = float(np.mean(np.abs(value)))
                    row[f"k{calls}_dit_flow_{name}_max_abs"] = float(np.max(np.abs(value)))
                for name, value in delta_groups.items():
                    row[f"k{calls}_dit_flow_delta_{name}_rms"] = float(np.sqrt(np.mean(value**2)))
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No trace files found in {input_dir}")
    return pd.DataFrame(rows)


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


def flow_feature_sets(calls: int) -> dict[str, list[str]]:
    scalar = [
        f"k{calls}_dit_flow_all32_rms",
        f"k{calls}_dit_flow_action8_rms",
        f"k{calls}_dit_flow_joint7_rms",
        f"k{calls}_dit_flow_gripper_rms",
        f"k{calls}_dit_flow_delta_all32_rms",
        f"k{calls}_dit_flow_delta_action8_rms",
        f"k{calls}_dit_flow_delta_joint7_rms",
        f"k{calls}_dit_flow_delta_gripper_rms",
    ]
    expanded = []
    for group in ["all32", "action8", "joint7", "gripper"]:
        expanded.extend(
            [
                f"k{calls}_dit_flow_{group}_rms",
                f"k{calls}_dit_flow_{group}_mean_abs",
                f"k{calls}_dit_flow_{group}_max_abs",
                f"k{calls}_dit_flow_delta_{group}_rms",
            ]
        )
    return {"dit_flow_scalar": scalar, "dit_flow_expanded": expanded}


def complexity_correlations(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calls in [1, *CHECKPOINTS, 16]:
        features = flow_feature_sets(calls if calls in CHECKPOINTS else calls)["dit_flow_expanded"]
        for feature in features:
            for target in COMPLEXITY_TARGETS:
                if target not in samples:
                    continue
                rho, p_value = spearmanr(samples[feature], samples[target], nan_policy="omit")
                rows.append(
                    {
                        "calls": calls,
                        "feature": feature,
                        "target": target,
                        "spearman": float(rho),
                        "p_value": float(p_value),
                    }
                )
    return pd.DataFrame(rows).sort_values("spearman", key=lambda values: values.abs(), ascending=False)


def compute_prediction(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calls in CHECKPOINTS:
        target = f"benefit_k{calls}_to_16"
        y = samples[target].to_numpy(dtype=float)
        for group, features in flow_feature_sets(calls).items():
            prediction = ridge_oof(samples, features, target)
            rho, p_value = spearmanr(y, prediction)
            row: dict[str, object] = {
                "calls": calls,
                "feature_group": group,
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
            rows.append(row)
    return pd.DataFrame(rows)


def stage_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calls in [1, *CHECKPOINTS, 16]:
        for stage, subset in samples.groupby("stage"):
            rows.append(
                {
                    "calls": calls,
                    "stage": stage,
                    "samples": len(subset),
                    "action8_rms_mean": float(subset[f"k{calls}_dit_flow_action8_rms"].mean()),
                    "joint7_rms_mean": float(subset[f"k{calls}_dit_flow_joint7_rms"].mean()),
                    "delta_action8_rms_mean": float(subset[f"k{calls}_dit_flow_delta_action8_rms"].mean()),
                    "gt_motion_h8_mean": float(subset["gt_motion_h8"].mean()),
                }
            )
    return pd.DataFrame(rows)


def write_report(path: Path, samples: pd.DataFrame, complexity: pd.DataFrame, prediction: pd.DataFrame, stages: pd.DataFrame) -> None:
    lines = [
        "# H4b DiT 输出 noise/flow 既有 trace 分析",
        "",
        f"- episode：{samples.episode.nunique()}",
        f"- chunk：{len(samples)}",
        "- 数据：H3 full-16 causal prefix trace 中已有 `action_flows`，因此本轮为探索性离线分析。",
        "",
        "## 与动作复杂度的 Spearman 相关",
        "",
        "| Calls | Feature | Target | Spearman | p |",
        "|---:|---|---|---:|---:|",
    ]
    for row in complexity.head(20).itertuples(index=False):
        lines.append(
            f"| {row.calls} | `{row.feature}` | `{row.target}` | {row.spearman:.3f} | {row.p_value:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 按阶段的 action-flow RMS",
            "",
            "| Calls | Stage | n | action8 RMS | joint7 RMS | delta action8 RMS | GT motion |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stages[(stages.calls == 8)].itertuples(index=False):
        lines.append(
            f"| {row.calls} | {row.stage} | {row.samples} | {row.action8_rms_mean:.4f} | "
            f"{row.joint7_rms_mean:.4f} | {row.delta_action8_rms_mean:.4f} | {row.gt_motion_h8_mean:.4f} |"
        )
    lines.extend(["", "## Held-out compute-benefit 预测", ""])
    for row in prediction.itertuples(index=False):
        lines.append(
            f"- k={row.calls} / `{row.feature_group}`：Spearman {row.spearman:.3f}，R2 {row.r2:.3f}，"
            f"benefit>0.002 AUROC {row.auroc_0p002:.3f}，AUPRC {row.auprc_0p002:.3f}。"
        )
    best_complexity = complexity.iloc[0]
    best_prediction = prediction.sort_values("auroc_0p002", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## 初步判定",
            "",
            f"最强复杂度相关为 `{best_complexity.feature}` vs `{best_complexity.target}`，Spearman {best_complexity.spearman:.3f}。",
            f"最佳 compute-benefit AUROC 为 k={int(best_prediction.calls)} / `{best_prediction.feature_group}` 的 {best_prediction.auroc_0p002:.3f}。",
            "若方向与假设不一致或不能超过 H3 既有 flow/convergence 结果，则 DiT 输出 flow 只能作为弱诊断，不足以支持在线多算 gate。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = repo / input_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_samples = input_dir / "analysis" / "samples.parquet"
    if prior_samples.exists():
        samples = pd.read_parquet(prior_samples)
        samples.attrs["prefix16_max_diff"] = 0.0
    else:
        joint_range, gripper_range = action_ranges(Path(args.dataset))
        samples = build_samples(input_dir, joint_range, gripper_range)
    flow_noise = flow_noise_rows(input_dir)
    samples = samples.merge(flow_noise, on=["episode", "anchor", "chunk_index"], how="left")
    complexity = complexity_correlations(samples)
    prediction = compute_prediction(samples)
    stages = stage_summary(samples)

    samples.to_parquet(output_dir / "samples.parquet", index=False)
    complexity.to_csv(output_dir / "complexity_correlations.csv", index=False)
    prediction.to_csv(output_dir / "compute_prediction_metrics.csv", index=False)
    stages.to_csv(output_dir / "stage_flow_summary.csv", index=False)
    write_report(output_dir / "report.md", samples, complexity, prediction, stages)
    print((output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
