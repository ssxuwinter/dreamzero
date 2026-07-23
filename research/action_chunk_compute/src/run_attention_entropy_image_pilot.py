#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eval_utils.policy_client import WebsocketClientPolicy

CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--image-dir", default="image")
    parser.add_argument(
        "--phase-table",
        default="research/action_chunk_compute/experiments/h1_gt_chunk_phase/results/confirmatory_1200/chunks.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="research/action_chunk_compute/experiments/h4_attention_noise_diagnostics/results/attention_entropy_image_pilot",
    )
    parser.add_argument("--chunks-per-episode", type=int, default=1)
    parser.add_argument("--prompt", default="pick up the object")
    return parser.parse_args()


def load_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {path}")
    return np.stack(frames)


def load_episode_views(episode_dir: Path) -> dict[str, np.ndarray]:
    return {key: load_video(episode_dir / filename) for key, filename in CAMERA_FILES.items()}


def make_obs(views: dict[str, np.ndarray], indices: list[int], prompt: str, session_id: str) -> dict[str, object]:
    obs: dict[str, object] = {}
    for key, frames in views.items():
        clipped = [min(max(index, 0), len(frames) - 1) for index in indices]
        selected = frames[clipped]
        obs[key] = selected[0] if len(selected) == 1 else selected
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def summarize_entropy(records: list[dict[str, object]]) -> dict[str, float | int]:
    if not records:
        return {"record_count": 0}
    means = np.asarray([float(record["mean"]) for record in records], dtype=np.float64)
    return {
        "record_count": int(len(records)),
        "mean_entropy": float(means.mean()),
        "std_entropy": float(means.std()),
        "min_entropy": float(means.min()),
        "max_entropy": float(means.max()),
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    image_dir = Path(args.image_dir)
    phase_table = Path(args.phase_table)
    output_dir = Path(args.output_dir)
    if not image_dir.is_absolute():
        image_dir = repo / image_dir
    if not phase_table.is_absolute():
        phase_table = repo / phase_table
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = pd.read_parquet(phase_table)
    client = WebsocketClientPolicy(args.host, args.port)
    rows = []
    for episode_dir in sorted(image_dir.glob("episode_*")):
        episode = int(episode_dir.name.split("_")[-1])
        views = load_episode_views(episode_dir)
        frame_count = min(len(frames) for frames in views.values())
        episode_phases = phases[phases.episode == episode].sort_values("anchor")
        if episode_phases.empty:
            anchors = [23]
            stage_by_anchor = {23: "unknown"}
        else:
            valid = episode_phases[episode_phases.anchor < frame_count]
            if valid.empty:
                valid = episode_phases.head(1).copy()
                valid["anchor"] = min(23, frame_count - 1)
            anchors = [int(value) for value in valid.anchor.head(args.chunks_per_episode)]
            stage_by_anchor = {int(row.anchor): str(row.stage) for row in valid.itertuples(index=False)}

        session_id = f"attention-pilot-e{episode}-{uuid.uuid4()}"
        client.infer(make_obs(views, [0], args.prompt, session_id))
        for chunk_index, anchor in enumerate(anchors):
            indices = [anchor - 23, anchor - 16, anchor - 8, anchor]
            start = time.perf_counter()
            response = client.infer(make_obs(views, indices, args.prompt, session_id))
            latency = time.perf_counter() - start
            records = response.get("attention_entropy_trace", []) if isinstance(response, dict) else []
            summary = summarize_entropy(records)
            rows.append(
                {
                    "episode": episode,
                    "chunk_index": chunk_index,
                    "anchor": anchor,
                    "stage": stage_by_anchor.get(anchor, "unknown"),
                    "latency": latency,
                    **summary,
                }
            )
            (output_dir / f"episode_{episode:06d}_chunk_{chunk_index:02d}_entropy.json").write_text(
                json.dumps(
                    {
                        "episode": episode,
                        "chunk_index": chunk_index,
                        "anchor": anchor,
                        "stage": stage_by_anchor.get(anchor, "unknown"),
                        "latency": latency,
                        "records": records,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print(f"episode={episode} chunk={chunk_index} anchor={anchor} records={summary['record_count']} latency={latency:.2f}s", flush=True)
        client.reset({})

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.md").write_text(frame.to_markdown(index=False))
    print(frame.to_markdown(index=False))


if __name__ == "__main__":
    main()
