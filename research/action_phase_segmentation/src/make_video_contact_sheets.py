#!/usr/bin/env python3
"""Build video-only contact sheets for blind pilot boundary annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="/home/admin/.cache/DreamZero-DROID-Data"
    )
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--samples-per-sheet", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        default="research/action_phase_segmentation/pilot/contact_sheets",
    )
    return parser.parse_args()


def video_path(dataset: Path, episode: int, camera: str) -> Path:
    return (
        dataset
        / f"videos/chunk-{episode // 1000:03d}"
        / camera
        / f"episode_{episode:06d}.mp4"
    )


def load_episode_metadata(dataset: Path) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    with (dataset / "meta/episodes.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["episode_index"])] = row
    return rows


def open_videos(dataset: Path, episode: int) -> tuple[list[cv2.VideoCapture], int]:
    captures = []
    frame_counts = []
    for camera in CAMERAS:
        path = video_path(dataset, episode, camera)
        if not path.exists():
            raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        captures.append(capture)
        frame_counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    return captures, min(frame_counts)


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index}")
    return cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)


def sample_panel(
    captures: list[cv2.VideoCapture], episode: int, frame_index: int
) -> np.ndarray:
    images = [read_frame(capture, frame_index) for capture in captures]
    panel = np.concatenate(images, axis=1)
    header = np.full((32, panel.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        f"episode {episode:06d} | frame {frame_index:04d} | {frame_index / 15.0:5.2f}s",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, panel], axis=0)


def tiled_sheet(panels: list[np.ndarray], columns: int = 2) -> np.ndarray:
    panel_h, panel_w = panels[0].shape[:2]
    rows = (len(panels) + columns - 1) // columns
    canvas = np.full((rows * panel_h, columns * panel_w, 3), 230, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        y, x = row * panel_h, column * panel_w
        canvas[y : y + panel_h, x : x + panel_w] = panel
    return canvas


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_episode_metadata(dataset)
    manifest = {
        "source": "video-only; no action data read",
        "fps": 15,
        "stride": args.stride,
        "samples_per_sheet": args.samples_per_sheet,
        "camera_order": list(CAMERAS),
        "episodes": [],
    }

    for episode in args.episodes:
        captures, frame_count = open_videos(dataset, episode)
        try:
            indices = list(range(0, frame_count, args.stride))
            episode_outputs = []
            for sheet_index, start in enumerate(
                range(0, len(indices), args.samples_per_sheet)
            ):
                selected = indices[start : start + args.samples_per_sheet]
                panels = [sample_panel(captures, episode, index) for index in selected]
                sheet = tiled_sheet(panels)
                output = output_dir / f"episode_{episode:06d}_sheet_{sheet_index:02d}.jpg"
                if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"Cannot write {output}")
                episode_outputs.append(
                    {"path": str(output), "sampled_frames": selected}
                )
        finally:
            for capture in captures:
                capture.release()

        row = metadata.get(episode, {})
        manifest["episodes"].append(
            {
                "episode": episode,
                "frame_count": frame_count,
                "tasks": row.get("tasks", []),
                "sheets": episode_outputs,
            }
        )
        print(f"episode={episode} frames={frame_count} sheets={len(episode_outputs)}")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
