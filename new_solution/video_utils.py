from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    video_path: Path
    fps: float
    num_frames: int
    width: int
    height: int
    duration_sec: float


def find_mp4_videos(video_root: Path) -> list[Path]:
    videos: list[Path] = []
    for path in sorted(video_root.rglob("*.mp4")):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if any(part == "__MACOSX" or part.startswith("._") for part in path.parts):
            continue
        videos.append(path)
    return videos


def _count_frames_fallback(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for frame counting: {video_path}")

    count = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    return count


def load_video_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if num_frames <= 0:
            num_frames = _count_frames_fallback(video_path)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    duration_sec = float(num_frames / fps) if fps > 0 else 0.0
    return VideoMetadata(
        video_path=video_path,
        fps=fps,
        num_frames=num_frames,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def _unique_preserve_order(indices: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for index in indices:
        index = int(index)
        if index in seen:
            continue
        ordered.append(index)
        seen.add(index)
    return ordered


def sample_frame_indices(
    start_frame: int,
    end_frame: int,
    num_samples: int,
) -> list[int]:
    if num_samples <= 0 or end_frame < start_frame:
        return []

    segment_length = end_frame - start_frame + 1
    if segment_length <= num_samples:
        return list(range(start_frame, end_frame + 1))

    indices = np.rint(np.linspace(start_frame, end_frame, num_samples)).astype(np.int64)
    sampled = _unique_preserve_order(indices.tolist())
    if len(sampled) >= num_samples:
        return sampled[:num_samples]

    full_range = list(range(start_frame, end_frame + 1))
    for index in full_range:
        if index not in sampled:
            sampled.append(index)
        if len(sampled) >= num_samples:
            break
    return sorted(sampled[:num_samples])


def extract_rgb_frames_by_indices(
    video_path: Path,
    frame_indices: Iterable[int],
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        capture.release()

    return frames


def extract_segment_rgb_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    num_samples: int,
) -> tuple[list[np.ndarray], list[int]]:
    sampled_indices = sample_frame_indices(
        start_frame=start_frame,
        end_frame=end_frame,
        num_samples=num_samples,
    )
    frames = extract_rgb_frames_by_indices(video_path, sampled_indices)
    return frames, sampled_indices[: len(frames)]

