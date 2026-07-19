#!/usr/bin/env python3
"""Run paired DreamZero inference against several static DiT schedules."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eval_utils.policy_client import WebsocketClientPolicy


CAMERAS = {
    "observation/exterior_image_0_left": "exterior_image_1_left",
    "observation/exterior_image_1_left": "exterior_image_2_left",
    "observation/wrist_image_left": "wrist_image_left",
}
FRAME_OFFSETS = [-23, -16, -8, 0]
SCHEDULE_PORTS = {5: 8105, 8: 8108, 16: 8116}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--dataset", default="/home/admin/.cache/DreamZero-DROID-Data")
    parser.add_argument(
        "--sample-manifest",
        default="research/action_chunk_compute/experiments/h2_wam_static_compute/sample_manifest.json",
    )
    parser.add_argument(
        "--phase-table",
        default="research/action_chunk_compute/experiments/h1_gt_chunk_phase/results/confirmatory_1200/chunks.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="research/action_chunk_compute/experiments/h2_wam_static_compute/results/diagnostic_contiguous_12",
    )
    parser.add_argument("--schedules", type=int, nargs="+", default=[5, 8, 16])
    parser.add_argument("--ports", type=int, nargs="+", default=[8105, 8108, 8116])
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--repeat-first", action="store_true")
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


def parquet_path(dataset: Path, episode: int) -> Path:
    return dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"


def video_path(dataset: Path, episode: int, camera: str) -> Path:
    return dataset / f"videos/chunk-{episode // 1000:03d}/{camera}/episode_{episode:06d}.mp4"


def load_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def load_episode(dataset: Path, episode: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    table = pd.read_parquet(parquet_path(dataset, episode))
    cameras = {
        observation_key: load_video(video_path(dataset, episode, dataset_key))
        for observation_key, dataset_key in CAMERAS.items()
    }
    lengths = {len(table), *(len(frames) for frames in cameras.values())}
    if len(lengths) != 1:
        raise RuntimeError(f"Episode {episode} has inconsistent lengths: {sorted(lengths)}")
    return table, cameras


def observation(
    cameras: dict[str, np.ndarray],
    states: np.ndarray,
    indices: list[int],
    prompt: str,
    session: str,
) -> dict[str, object]:
    anchor = indices[-1]
    obs: dict[str, object] = {}
    for key, frames in cameras.items():
        selected = frames[indices]
        obs[key] = selected[0] if len(indices) == 1 else selected
    obs["observation/cartesian_position"] = states[anchor, 0:6].astype(np.float32)
    obs["observation/gripper_position"] = states[anchor, 6:7].astype(np.float32)
    obs["observation/joint_position"] = states[anchor, 7:14].astype(np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session
    return obs


def infer_one(client: WebsocketClientPolicy, obs: dict[str, object]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    action = client.infer(dict(obs))
    elapsed = time.perf_counter() - start
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (24, 8):
        raise RuntimeError(f"Expected (24, 8), received {action.shape}")
    return action, elapsed


def parallel_infer(
    executor: concurrent.futures.ThreadPoolExecutor,
    clients: dict[int, WebsocketClientPolicy],
    observations: dict[int, dict[str, object]],
) -> dict[int, tuple[np.ndarray, float]]:
    futures = {
        schedule: executor.submit(infer_one, clients[schedule], observations[schedule])
        for schedule in clients
    }
    return {schedule: future.result() for schedule, future in futures.items()}


def close_clients(clients: dict[int, WebsocketClientPolicy]) -> None:
    for client in clients.values():
        try:
            client._ws.close()  # Closing avoids expensive generated-video decoding on reset.
        except Exception:
            pass


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_episode(
    host: str,
    ports: dict[int, int],
    dataset: Path,
    episode_spec: dict[str, object],
    phases: pd.DataFrame,
    output: Path,
    run_rep: int,
) -> None:
    episode = int(episode_spec["episode"])
    prompt = str(episode_spec["prompt"])
    table, cameras = load_episode(dataset, episode)
    states = np.stack(table["observation.state"].to_numpy()).astype(np.float64)
    raw_actions = np.stack(table["action"].to_numpy()).astype(np.float64)
    phase_rows = phases[phases["episode"] == episode].sort_values("anchor")
    if phase_rows.empty:
        raise RuntimeError(f"No phase rows found for episode {episode}")

    clients = {schedule: WebsocketClientPolicy(host, port) for schedule, port in ports.items()}
    sessions = {
        schedule: f"wam-static-e{episode}-r{run_rep}-s{schedule}-{uuid.uuid4()}"
        for schedule in ports
    }
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
            initial_obs = {
                schedule: observation(cameras, states, [0], prompt, sessions[schedule])
                for schedule in ports
            }
            initial = parallel_infer(executor, clients, initial_obs)
            append_rows(
                output,
                [
                    {
                        "record_type": "initial",
                        "run_rep": run_rep,
                        "schedule": schedule,
                        "episode": episode,
                        "prompt": prompt,
                        "latency_seconds": elapsed,
                        "prediction": action.tolist(),
                    }
                    for schedule, (action, elapsed) in initial.items()
                ],
            )

            for chunk_index, phase_row in enumerate(phase_rows.itertuples(index=False)):
                anchor = int(phase_row.anchor)
                indices = [anchor + offset for offset in FRAME_OFFSETS]
                end = anchor + 24
                if min(indices) < 0 or end > len(table):
                    raise RuntimeError(f"Invalid anchor {anchor} for episode {episode}")
                observations = {
                    schedule: observation(cameras, states, indices, prompt, sessions[schedule])
                    for schedule in ports
                }
                results = parallel_infer(executor, clients, observations)
                current_state = np.concatenate([states[anchor, 7:14], states[anchor, 6:7]])
                gt_action = np.concatenate(
                    [raw_actions[anchor:end, 14:21], raw_actions[anchor:end, 12:13]], axis=1
                )
                rows = []
                for schedule, (prediction, elapsed) in results.items():
                    rows.append(
                        {
                            "record_type": "chunk",
                            "run_rep": run_rep,
                            "schedule": schedule,
                            "episode": episode,
                            "prompt": prompt,
                            "chunk_index": chunk_index,
                            "anchor": anchor,
                            "frame_indices": indices,
                            "stage": str(phase_row.stage),
                            "fine": int(phase_row.fine),
                            "latency_seconds": elapsed,
                            "current_state": current_state.astype(np.float32).tolist(),
                            "gt_action": gt_action.astype(np.float32).tolist(),
                            "prediction": prediction.tolist(),
                        }
                    )
                append_rows(output, rows)
                print(
                    f"episode={episode} chunk={chunk_index + 1}/{len(phase_rows)} "
                    + " ".join(
                        f"s{schedule}={results[schedule][1]:.2f}s" for schedule in sorted(results)
                    ),
                    flush=True,
                )
    finally:
        close_clients(clients)


def main() -> None:
    args = parse_args()
    if len(args.schedules) != len(args.ports):
        raise ValueError("--schedules and --ports must have equal lengths")
    ports = dict(zip(args.schedules, args.ports, strict=True))
    unexpected = set(ports) - set(SCHEDULE_PORTS)
    if unexpected:
        raise ValueError(f"Unsupported schedules: {sorted(unexpected)}")

    repo = Path(__file__).resolve().parents[3]
    dataset = Path(args.dataset)
    sample_manifest = Path(args.sample_manifest)
    phase_table = Path(args.phase_table)
    output_dir = Path(args.output_dir)
    if not sample_manifest.is_absolute():
        sample_manifest = repo / sample_manifest
    if not phase_table.is_absolute():
        phase_table = repo / phase_table
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "raw_predictions.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    manifest = json.loads(sample_manifest.read_text())
    episodes = manifest["episodes"]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    phases = pd.read_parquet(phase_table)
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "ports": ports,
        "dataset": str(dataset),
        "sample_manifest": str(sample_manifest),
        "phase_table": str(phase_table),
        "episodes": episodes,
        "git": git_info(repo),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False)
    )

    for index, episode_spec in enumerate(episodes):
        run_episode(args.host, ports, dataset, episode_spec, phases, output, run_rep=0)
        if index == 0 and args.repeat_first:
            run_episode(args.host, ports, dataset, episode_spec, phases, output, run_rep=1)
    print(f"completed {len(episodes)} episodes; output={output}")


if __name__ == "__main__":
    main()
