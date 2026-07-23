#!/usr/bin/env python3
"""Council-requested rigor checks: benefit label learnability, bootstrap CIs, trivial baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CHECKPOINTS = [2, 4, 8]
TOLERANCE = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h3-analysis",
        default="research/action_chunk_compute/experiments/h3_nested_provisional_compute/results/prefix_trace_12/analysis",
    )
    parser.add_argument(
        "--h4b-analysis",
        default="research/action_chunk_compute/experiments/h4_attention_noise_diagnostics/results/flow_noise_existing_trace",
    )
    parser.add_argument(
        "--output-dir",
        default="research/action_chunk_compute/experiments/h4_attention_noise_diagnostics/results/benefit_rigor",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def episode_bootstrap_auroc(
    frame: pd.DataFrame, score_col: str, label_col: str, reps: int, seed: int
) -> dict[str, float]:
    """Bootstrap over episodes; recompute pooled AUROC each draw."""
    episodes = frame["episode"].unique()
    rng = np.random.default_rng(seed)
    point = np.nan
    labels = frame[label_col].to_numpy(dtype=bool)
    if len(np.unique(labels)) == 2:
        point = float(roc_auc_score(labels, frame[score_col]))
    draws = []
    for _ in range(reps):
        chosen = rng.choice(episodes, len(episodes), replace=True)
        sample = pd.concat([frame[frame.episode == ep] for ep in chosen])
        y = sample[label_col].to_numpy(dtype=bool)
        if len(np.unique(y)) < 2:
            continue
        draws.append(roc_auc_score(y, sample[score_col]))
    draws = np.asarray(draws)
    return {
        "auroc": point,
        "ci_low": float(np.quantile(draws, 0.025)) if len(draws) else np.nan,
        "ci_high": float(np.quantile(draws, 0.975)) if len(draws) else np.nan,
        "valid_draws": int(len(draws)),
    }


def episode_bootstrap_spearman(
    frame: pd.DataFrame, score_col: str, target_col: str, reps: int, seed: int
) -> dict[str, float]:
    episodes = frame["episode"].unique()
    rng = np.random.default_rng(seed)
    point = float(spearmanr(frame[score_col], frame[target_col])[0])
    draws = []
    for _ in range(reps):
        chosen = rng.choice(episodes, len(episodes), replace=True)
        sample = pd.concat([frame[frame.episode == ep] for ep in chosen])
        rho = spearmanr(sample[score_col], sample[target_col])[0]
        if np.isfinite(rho):
            draws.append(rho)
    draws = np.asarray(draws)
    return {
        "spearman": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


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


def benefit_distribution(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calls in CHECKPOINTS:
        b = samples[f"benefit_k{calls}_to_16"].to_numpy(dtype=float)
        e16 = samples["e16_joint_h8"].to_numpy(dtype=float)
        rows.append(
            {
                "calls": calls,
                "mean": float(b.mean()),
                "std": float(b.std()),
                "q05": float(np.quantile(b, 0.05)),
                "median": float(np.median(b)),
                "q95": float(np.quantile(b, 0.95)),
                "frac_pos": float((b > 0).mean()),
                "frac_gt_0.002": float((b > 0.002).mean()),
                "frac_gt_0.005": float((b > 0.005).mean()),
                "n_pos_0.002": int((b > 0.002).sum()),
                "mean_relative_to_e16": float(b.mean() / e16.mean()),
                "q95_relative_to_e16": float(np.quantile(b, 0.95) / e16.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    h3_dir = repo / args.h3_analysis
    h4b_dir = repo / args.h4b_analysis
    out_dir = repo / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(h4b_dir / "samples.parquet")
    h3_preds = pd.read_parquet(h3_dir / "compute_predictions.parquet")

    dist = benefit_distribution(samples)

    auroc_rows = []
    spearman_rows = []
    for calls in CHECKPOINTS:
        target = f"benefit_k{calls}_to_16"
        label = f"label_{calls}"
        samples[label] = samples[target] > TOLERANCE

        # 1) H3 feature groups from saved OOF predictions
        for group in sorted(h3_preds.feature_group.unique()):
            sub = h3_preds[(h3_preds.calls == calls) & (h3_preds.feature_group == group)].copy()
            sub["label"] = sub["actual"] > TOLERANCE
            res = episode_bootstrap_auroc(sub, "prediction", "label", args.bootstrap_reps, args.seed + calls)
            auroc_rows.append({"calls": calls, "predictor": f"h3_{group}", **res})
            res_s = episode_bootstrap_spearman(sub, "prediction", "actual", args.bootstrap_reps, args.seed + calls)
            spearman_rows.append({"calls": calls, "predictor": f"h3_{group}", **res_s})

        # 2) H4b flow feature groups (recompute OOF)
        flow_scalar = [
            f"k{calls}_dit_flow_all32_rms", f"k{calls}_dit_flow_action8_rms",
            f"k{calls}_dit_flow_joint7_rms", f"k{calls}_dit_flow_gripper_rms",
            f"k{calls}_dit_flow_delta_all32_rms", f"k{calls}_dit_flow_delta_action8_rms",
            f"k{calls}_dit_flow_delta_joint7_rms", f"k{calls}_dit_flow_delta_gripper_rms",
        ]
        pred = ridge_oof(samples, flow_scalar, target)
        tmp = samples[["episode", label, target]].copy()
        tmp["score"] = pred
        res = episode_bootstrap_auroc(tmp, "score", label, args.bootstrap_reps, args.seed + calls)
        auroc_rows.append({"calls": calls, "predictor": "h4b_dit_flow_scalar", **res})
        res_s = episode_bootstrap_spearman(tmp, "score", target, args.bootstrap_reps, args.seed + calls)
        spearman_rows.append({"calls": calls, "predictor": "h4b_dit_flow_scalar", **res_s})

        # 3) Trivial single-feature baselines (no fitting; raw feature as score)
        for feat, name in [
            (f"k{calls}_pred_motion_h8", "raw_pred_motion"),
            (f"k{calls}_dit_flow_joint7_rms", "raw_flow_rms"),
            (f"k{calls}_dit_flow_delta_joint7_rms", "raw_flow_delta_rms"),
            (f"e{calls}_joint_h8", "oracle_current_error"),
        ]:
            tmp = samples[["episode", label, target, feat]].copy()
            res = episode_bootstrap_auroc(tmp, feat, label, args.bootstrap_reps, args.seed + calls)
            auroc_rows.append({"calls": calls, "predictor": name, **res})
            res_s = episode_bootstrap_spearman(tmp, feat, target, args.bootstrap_reps, args.seed + calls)
            spearman_rows.append({"calls": calls, "predictor": name, **res_s})

    auroc_df = pd.DataFrame(auroc_rows)
    spearman_df = pd.DataFrame(spearman_rows)

    dist.to_csv(out_dir / "benefit_distribution.csv", index=False)
    auroc_df.to_csv(out_dir / "auroc_bootstrap.csv", index=False)
    spearman_df.to_csv(out_dir / "spearman_bootstrap.csv", index=False)

    lines = [
        "# Benefit 标签严谨性检查（council 要求）",
        "",
        f"- 样本：{samples.episode.nunique()} episode / {len(samples)} chunk；tolerance={TOLERANCE}",
        "",
        "## 1. Benefit 分布（正收益天花板）",
        "",
        dist.round(5).to_markdown(index=False),
        "",
        "## 2. Episode-bootstrap AUROC（benefit>0.002）",
        "",
        auroc_df.round(3).to_markdown(index=False),
        "",
        "## 3. Episode-bootstrap Spearman（连续 benefit）",
        "",
        spearman_df.round(3).to_markdown(index=False),
        "",
        "## 判读要点",
        "",
        "- 若所有 CI 均横跨 0.5（AUROC）或 0（Spearman），包括 oracle_current_error，则标签本身在该样本量下不可学，flow 的 0.484 不能解释为特异性失败。",
        "- 若 oracle_current_error（当前误差，作弊特征）显著 >0.5 而 flow 仍在 0.5 附近，则可下 flow-specific no-go。",
        "- benefit 相对 e16 的量级（mean_relative_to_e16）指示即便完美 gate 可省的误差上限。",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
