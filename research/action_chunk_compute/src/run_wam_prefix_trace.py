#!/usr/bin/env python3
"""Collect causal prefix-stop action traces from a full 16-call WAM run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eval_utils.policy_client import WebsocketClientPolicy
from run_wam_static_pairing import load_episode, observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8120)
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
        default="research/action_chunk_compute/experiments/h3_nested_provisional_compute/results/prefix_trace_12",
    )
    parser.add_argument("--max-episodes", type=int)
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


def infer_trace(client: WebsocketClientPolicy, obs: dict[str, object]) -> tuple[dict, float]:
    start = time.perf_counter()
    response = client.infer(dict(obs))
    elapsed = time.perf_counter() - start
    if not isinstance(response, dict):
        raise RuntimeError(f"Expected trace response dict, received {type(response)}")

    arrays = {
        "action": np.asarray(response["action"], dtype=np.float32),
        "prefix_actions": np.asarray(response["prefix_actions"], dtype=np.float32),
        "normalized_prefix_actions": np.asarray(
            response["normalized_prefix_actions"], dtype=np.float32
        ),
        "action_flows": np.asarray(response["action_flows"], dtype=np.float32),
        "attention_entropy_trace": response.get("attention_entropy_trace", []),
    }
    expected = {
        "action": (24, 8),
        "prefix_actions": (16, 24, 8),
        "normalized_prefix_actions": (16, 24, 32),
        "action_flows": (16, 24, 32),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise RuntimeError(f"{key}: expected {shape}, received {arrays[key].shape}")
    max_diff = float(np.max(np.abs(arrays["prefix_actions"][-1] - arrays["action"])))
    if max_diff > 1e-6:
        raise RuntimeError(f"prefix-16 mismatch: max_abs_diff={max_diff}")
    return arrays, elapsed


def close_client(client: WebsocketClientPolicy) -> None:
    try:
        client._ws.close()
    except Exception:
        pass


def run_episode(
    host: str,
    port: int,
    dataset: Path,
    episode_spec: dict[str, object],
    phases: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    episode = int(episode_spec["episode"])
    prompt = str(episode_spec["prompt"])
    table, cameras = load_episode(dataset, episode)
    states = np.stack(table["observation.state"].to_numpy()).astype(np.float64)
    raw_actions = np.stack(table["action"].to_numpy()).astype(np.float64)
    phase_rows = phases[phases["episode"] == episode].sort_values("anchor")
    if phase_rows.empty:
        raise RuntimeError(f"No phase rows for episode {episode}")

    session = f"wam-prefix-e{episode}-{uuid.uuid4()}"
    client = WebsocketClientPolicy(host, port)
    try:
        initial_obs = observation(cameras, states, [0], prompt, session)
        _, initial_latency = infer_trace(client, initial_obs)

        anchors: list[int] = []
        stages: list[str] = []
        fine: list[int] = []
        current_states: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        final_actions: list[np.ndarray] = []
        prefix_actions: list[np.ndarray] = []
        normalized_prefix_actions: list[np.ndarray] = []
        action_flows: list[np.ndarray] = []
        attention_entropy_traces: list[dict[str, object]] = []
        latencies: list[float] = []

        for chunk_index, phase_row in enumerate(phase_rows.itertuples(index=False)):
            anchor = int(phase_row.anchor)
            indices = [anchor - 23, anchor - 16, anchor - 8, anchor]
            obs = observation(cameras, states, indices, prompt, session)
            trace, latency = infer_trace(client, obs)
            end = anchor + 24
            target = np.concatenate(
                [raw_actions[anchor:end, 14:21], raw_actions[anchor:end, 12:13]], axis=1
            )
            current_state = np.concatenate([states[anchor, 7:14], states[anchor, 6:7]])
            anchors.append(anchor)
            stages.append(str(phase_row.stage))
            fine.append(int(phase_row.fine))
            current_states.append(current_state.astype(np.float32))
            targets.append(target.astype(np.float32))
            final_actions.append(trace["action"])
            prefix_actions.append(trace["prefix_actions"])
            normalized_prefix_actions.append(trace["normalized_prefix_actions"])
            action_flows.append(trace["action_flows"])
            attention_entropy_traces.append(
                {
                    "anchor": int(anchor),
                    "chunk_index": int(chunk_index),
                    "records": trace.get("attention_entropy_trace", []),
                }
            )
            latencies.append(latency)
            print(
                f"episode={episode} chunk={chunk_index + 1}/{len(phase_rows)} "
                f"latency={latency:.2f}s",
                flush=True,
            )
    finally:
        close_client(client)

    output_path = output_dir / f"episode_{episode:06d}.npz"
    np.savez_compressed(
        output_path,
        episode=np.asarray(episode, dtype=np.int64),
        prompt=np.asarray(prompt),
        anchors=np.asarray(anchors, dtype=np.int64),
        stages=np.asarray(stages),
        fine=np.asarray(fine, dtype=np.int8),
        current_states=np.stack(current_states),
        gt_actions=np.stack(targets),
        final_actions=np.stack(final_actions),
        prefix_actions=np.stack(prefix_actions),
        normalized_prefix_actions=np.stack(normalized_prefix_actions),
        action_flows=np.stack(action_flows),
        latencies=np.asarray(latencies, dtype=np.float64),
        initial_latency=np.asarray(initial_latency, dtype=np.float64),
    )
    entropy_path = output_dir / f"episode_{episode:06d}_attention_entropy.json"
    entropy_path.write_text(
        json.dumps(
            {
                "episode": episode,
                "prompt": prompt,
                "chunks": attention_entropy_traces,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return {
        "episode": episode,
        "chunks": len(anchors),
        "path": str(output_path),
        "initial_latency": initial_latency,
        "mean_latency": float(np.mean(latencies)),
    }


def main() -> None:
    args = parse_args()
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

    manifest = json.loads(sample_manifest.read_text())
    episodes = manifest["episodes"]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    expected_paths = [output_dir / f"episode_{int(row['episode']):06d}.npz" for row in episodes]
    existing = [str(path) for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite trace files: {existing}")
    phases = pd.read_parquet(phase_table)
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "port": args.port,
        "dataset": str(dataset),
        "sample_manifest": str(sample_manifest),
        "phase_table": str(phase_table),
        "prefix_stops": list(range(1, 17)),
        "semantics": "first k full-trajectory action flows, then reuse flow k through scheduler step 16",
        "episodes": episodes,
        "git": git_info(repo),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False)
    )

    completed = []
    for episode_spec in episodes:
        completed.append(
            run_episode(args.host, args.port, dataset, episode_spec, phases, output_dir)
        )
    (output_dir / "completed.json").write_text(
        json.dumps(completed, indent=2, ensure_ascii=False)
    )
    print(f"completed {len(completed)} episodes; output={output_dir}")


if __name__ == "__main__":
    main()
