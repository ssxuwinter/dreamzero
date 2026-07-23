#!/usr/bin/env python3
"""Analyze attention entropy across manipulation stages, joined with H3 benefit/motion data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STAGE_ORDER = ["free_open", "pre_close", "closing", "hold", "release"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="research/action_chunk_compute/experiments/h4_attention_noise_diagnostics/results/attention_entropy_stages_21",
    )
    parser.add_argument(
        "--h3-samples",
        default="research/action_chunk_compute/experiments/h3_nested_provisional_compute/results/prefix_trace_12/analysis/samples.parquet",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def load_chunk_records(input_dir: Path) -> pd.DataFrame:
    """Load per-chunk entropy JSONs into long and summary frames."""
    rows = []
    for path in sorted(input_dir.glob("episode_*_chunk_*_entropy.json")):
        payload = json.loads(path.read_text())
        records = payload["records"]
        if not records:
            continue
        frame = pd.DataFrame(records)
        # KV-cache 推理路径记录 kv_action_register；每次 DiT call 每层一条
        for label, sub in frame.groupby("label"):
            means = sub["mean"].to_numpy(dtype=float)
            rows.append(
                {
                    "episode": payload["episode"],
                    "chunk_index": payload["chunk_index"],
                    "anchor": payload["anchor"],
                    "stage": payload["stage"],
                    "attn_label": label,
                    "n_records": len(sub),
                    "entropy_mean": float(means.mean()),
                    "entropy_std": float(means.std()),
                    "entropy_p10": float(np.quantile(means, 0.10)),
                    "entropy_p90": float(np.quantile(means, 0.90)),
                    "entropy_min": float(sub["min"].min()),
                    "entropy_max": float(sub["max"].max()),
                }
            )
    if not rows:
        raise RuntimeError(f"No entropy records under {input_dir}")
    return pd.DataFrame(rows)


def episode_bootstrap_diff(
    frame: pd.DataFrame, value_col: str, group_a: list[str], group_b: list[str], reps: int, seed: int
) -> dict[str, float]:
    """Bootstrap (over chunks within episode-stage cells) difference fine-vs-free."""
    a = frame[frame.stage.isin(group_a)][value_col].to_numpy(dtype=float)
    b = frame[frame.stage.isin(group_b)][value_col].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    draws = [
        rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
        for _ in range(reps)
    ]
    return {
        "diff": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    input_dir = repo / args.input_dir
    out_dir = input_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunk_records(input_dir)
    # 只分析 KV-cache action-register 路径（实际推理路径）
    main_frame = chunks[chunks.attn_label == "kv_action_register"].copy()
    if main_frame.empty:
        main_frame = chunks.copy()

    h3 = pd.read_parquet(repo / args.h3_samples)
    joined = main_frame.merge(
        h3[
            [
                "episode", "anchor", "gt_motion_h8", "fine",
                "benefit_k2_to_16", "benefit_k4_to_16", "benefit_k8_to_16",
                "e16_joint_h8", "k8_pred_motion_h8",
            ]
        ],
        on=["episode", "anchor"],
        how="left",
        suffixes=("", "_h3"),
    )

    stage_summary = (
        joined.groupby("stage")
        .agg(
            chunks=("episode", "count"),
            entropy_mean=("entropy_mean", "mean"),
            entropy_sd=("entropy_mean", "std"),
            gt_motion=("gt_motion_h8", "mean"),
        )
        .reindex(STAGE_ORDER)
        .reset_index()
    )

    fine_vs_free = episode_bootstrap_diff(
        joined, "entropy_mean",
        ["pre_close", "closing", "hold", "release"], ["free_open"],
        args.bootstrap_reps, args.seed,
    )

    corr_rows = []
    for target in ["gt_motion_h8", "k8_pred_motion_h8", "benefit_k2_to_16", "benefit_k4_to_16", "benefit_k8_to_16", "e16_joint_h8"]:
        valid = joined.dropna(subset=[target])
        rho, p = spearmanr(valid["entropy_mean"], valid[target])
        corr_rows.append({"target": target, "n": len(valid), "spearman": float(rho), "p": float(p)})
    corr = pd.DataFrame(corr_rows)

    joined.to_parquet(out_dir / "entropy_chunks.parquet", index=False)
    stage_summary.to_csv(out_dir / "stage_summary.csv", index=False)
    corr.to_csv(out_dir / "entropy_correlations.csv", index=False)

    lines = [
        "# H5 attention entropy 跨阶段分析（21 chunk / 3 episode / 5 stage）",
        "",
        f"- attn 路径：kv_action_register（实际推理路径）；每 chunk 记录数 ≈ {int(main_frame.n_records.mean())}",
        "",
        "## 按阶段的 entropy",
        "",
        stage_summary.round(5).to_markdown(index=False),
        "",
        "## fine(pre_close+closing+hold+release) − free_open 差值",
        "",
        f"diff = {fine_vs_free['diff']:.5f}, 95% CI [{fine_vs_free['ci_low']:.5f}, {fine_vs_free['ci_high']:.5f}]",
        "",
        "## entropy 与复杂度/收益的 Spearman",
        "",
        corr.round(4).to_markdown(index=False),
        "",
        "## 判读",
        "",
        "- 假设 H5a 预测 fine 阶段 entropy 更高。若 CI 含 0 或方向相反，则该样本不支持。",
        "- 若 entropy 与 benefit 的相关 CI 含 0，则 entropy 也不能作为 compute gate 信号。",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
