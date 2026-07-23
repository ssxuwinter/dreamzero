#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="research/action_chunk_compute/experiments/h4_attention_noise_diagnostics/results/attention_entropy_image_pilot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = repo / input_dir
    summary_path = input_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    frame = pd.read_csv(summary_path)
    lines = [
        "# H5 attention entropy image pilot",
        "",
        f"- episode：{frame.episode.nunique() if not frame.empty else 0}",
        f"- chunk：{len(frame)}",
        "- 数据：`image/` 下三个 episode 的三视角 mp4。",
        "",
        "## Summary",
        "",
        frame.to_markdown(index=False),
        "",
    ]
    if not frame.empty and "mean_entropy" in frame:
        by_stage = frame.groupby("stage", dropna=False).agg(
            chunks=("episode", "count"),
            mean_entropy=("mean_entropy", "mean"),
            std_entropy=("mean_entropy", "std"),
            mean_records=("record_count", "mean"),
        ).reset_index()
        lines.extend(
            [
                "## By stage",
                "",
                by_stage.to_markdown(index=False),
                "",
                "## Pilot interpretation",
                "",
                "该 pilot 只验证 attention entropy 插桩与数据通路是否可用，样本数不足以支持 compute gate 结论。若 record_count 为 0，说明模型路径未触发当前插桩标签或服务端未返回诊断字段。",
                "",
            ]
        )
    report = input_dir / "report.md"
    report.write_text("\n".join(lines))
    print(report.read_text())


if __name__ == "__main__":
    main()
