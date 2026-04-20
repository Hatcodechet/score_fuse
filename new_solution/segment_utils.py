from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def sanitize_scores(anomaly_scores: np.ndarray | Iterable[float]) -> np.ndarray:
    scores = np.asarray(list(anomaly_scores) if not isinstance(anomaly_scores, np.ndarray) else anomaly_scores)
    scores = scores.astype(np.float32).reshape(-1)
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    values = sanitize_scores(values)
    if values.size == 0:
        return values
    window_size = max(1, int(window_size))
    if window_size == 1 or values.size == 1:
        return values.copy()

    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones((window_size,), dtype=np.float32) / float(window_size)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = sanitize_scores(values)
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return np.zeros_like(values)
    scale = max_value - min_value
    if scale <= 1e-6:
        return np.zeros_like(values)
    return ((values - min_value) / scale).astype(np.float32)


def resample_scores_to_length(scores: np.ndarray, target_length: int) -> np.ndarray:
    scores = sanitize_scores(scores)
    target_length = max(0, int(target_length))
    if target_length == 0:
        return np.zeros((0,), dtype=np.float32)
    if scores.size == 0:
        return np.zeros((target_length,), dtype=np.float32)
    if scores.size == target_length:
        return scores.astype(np.float32, copy=True)
    if scores.size == 1:
        return np.full((target_length,), float(scores[0]), dtype=np.float32)

    source_x = np.linspace(0.0, 1.0, scores.size, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    resampled = np.interp(target_x, source_x, scores.astype(np.float32))
    return resampled.astype(np.float32)


def _choose_uniform_segment_count(
    num_frames: int,
    fps: float,
    min_uniform_segments: int,
    max_uniform_segments: int,
) -> int:
    if num_frames <= 0:
        return 0

    fps = max(float(fps), 1e-6)
    duration_sec = num_frames / fps

    if num_frames < 24 or duration_sec < 2.0:
        return 1
    if num_frames < 80 or duration_sec < 6.0:
        return 2
    if num_frames < 180 or duration_sec < 12.0:
        return 3
    if num_frames < 450 or duration_sec < 30.0:
        return max(3, min(4, max_uniform_segments, max(1, min_uniform_segments)))
    return max(4, min(5, max_uniform_segments))


def _build_uniform_segments(
    num_frames: int,
    fps: float,
    min_uniform_segments: int,
    max_uniform_segments: int,
) -> list[dict]:
    if num_frames <= 0:
        return []

    segment_count = _choose_uniform_segment_count(
        num_frames=num_frames,
        fps=fps,
        min_uniform_segments=min_uniform_segments,
        max_uniform_segments=max_uniform_segments,
    )
    if segment_count <= 1:
        return [{"start_frame": 0, "end_frame": max(0, num_frames - 1), "segment_type": "uniform"}]

    boundaries = np.linspace(0, num_frames, segment_count + 1, dtype=np.int64)
    segments: list[dict] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        start_frame = int(start)
        end_frame = int(max(start_frame, end - 1))
        segments.append(
            {
                "start_frame": start_frame,
                "end_frame": min(num_frames - 1, end_frame),
                "segment_type": "uniform",
            }
        )
    return segments


def _find_contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start_index: int | None = None

    for index, active in enumerate(mask.tolist()):
        if active and start_index is None:
            start_index = index
            continue
        if not active and start_index is not None:
            regions.append((start_index, index - 1))
            start_index = None

    if start_index is not None:
        regions.append((start_index, int(mask.size - 1)))
    return regions


def _score_index_to_frame(index: int, score_length: int, num_frames: int, is_end: bool) -> int:
    if score_length <= 1:
        return max(0, num_frames - 1 if is_end else 0)

    frame_value = (float(index + (1 if is_end else 0)) / float(score_length)) * float(num_frames)
    if is_end:
        frame_index = int(math.ceil(frame_value) - 1)
    else:
        frame_index = int(math.floor(frame_value))
    return min(max(frame_index, 0), max(0, num_frames - 1))


def _merge_frame_segments(segments: list[dict], max_gap_frames: int) -> list[dict]:
    if not segments:
        return []

    merged: list[dict] = [segments[0].copy()]
    for segment in segments[1:]:
        previous = merged[-1]
        if segment["start_frame"] <= previous["end_frame"] + max_gap_frames:
            previous["end_frame"] = max(previous["end_frame"], segment["end_frame"])
            previous["segment_type"] = "event"
            continue
        merged.append(segment.copy())
    return merged


def propose_segments(
    anomaly_scores: np.ndarray,
    num_frames: int,
    fps: float,
    min_uniform_segments: int = 4,
    max_uniform_segments: int = 6,
    max_event_segments: int = 5,
    smooth_window: int = 5,
    low_peak_threshold: float = 0.35,
    relative_event_threshold: float = 0.5,
    min_event_len: int = 4,
    context_expand: int = 3,
) -> list[dict]:
    num_frames = max(0, int(num_frames))
    if num_frames == 0:
        return []

    raw_scores = sanitize_scores(anomaly_scores)
    if raw_scores.size == 0:
        return _build_uniform_segments(
            num_frames=num_frames,
            fps=fps,
            min_uniform_segments=min_uniform_segments,
            max_uniform_segments=max_uniform_segments,
        )

    normalized_scores = normalize_scores(raw_scores)
    smoothed_scores = moving_average(normalized_scores, window_size=smooth_window)

    peak_value = float(smoothed_scores.max()) if smoothed_scores.size else 0.0
    score_std = float(smoothed_scores.std()) if smoothed_scores.size else 0.0
    median_value = float(np.median(smoothed_scores)) if smoothed_scores.size else 0.0
    weak_curve = peak_value < low_peak_threshold or score_std < 0.08 or (peak_value - median_value) < 0.12
    if weak_curve:
        return _build_uniform_segments(
            num_frames=num_frames,
            fps=fps,
            min_uniform_segments=min_uniform_segments,
            max_uniform_segments=max_uniform_segments,
        )

    event_threshold = max(
        float(low_peak_threshold),
        float(peak_value * relative_event_threshold),
        float(np.percentile(smoothed_scores, 75)) if smoothed_scores.size >= 4 else 0.0,
    )
    event_mask = smoothed_scores >= event_threshold
    regions = _find_contiguous_regions(event_mask)

    if not regions:
        return _build_uniform_segments(
            num_frames=num_frames,
            fps=fps,
            min_uniform_segments=min_uniform_segments,
            max_uniform_segments=max_uniform_segments,
        )

    candidate_segments: list[dict] = []
    score_length = max(1, raw_scores.size)

    for start_idx, end_idx in regions:
        region_len = end_idx - start_idx + 1
        if region_len < max(1, int(min_event_len)):
            local_max = float(smoothed_scores[start_idx : end_idx + 1].max())
            if local_max < max(low_peak_threshold, peak_value * 0.75):
                continue

        expanded_start = max(0, start_idx - int(context_expand))
        expanded_end = min(score_length - 1, end_idx + int(context_expand))
        start_frame = _score_index_to_frame(expanded_start, score_length=score_length, num_frames=num_frames, is_end=False)
        end_frame = _score_index_to_frame(expanded_end, score_length=score_length, num_frames=num_frames, is_end=True)
        if end_frame < start_frame:
            continue

        region_scores = smoothed_scores[start_idx : end_idx + 1]
        candidate_segments.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "segment_type": "event",
                "_score_mean": float(region_scores.mean()),
                "_score_max": float(region_scores.max()),
            }
        )

    if not candidate_segments:
        return _build_uniform_segments(
            num_frames=num_frames,
            fps=fps,
            min_uniform_segments=min_uniform_segments,
            max_uniform_segments=max_uniform_segments,
        )

    candidate_segments = sorted(
        candidate_segments,
        key=lambda item: (item["_score_max"], item["_score_mean"], -(item["end_frame"] - item["start_frame"])),
        reverse=True,
    )[: max(1, int(max_event_segments))]
    candidate_segments = sorted(candidate_segments, key=lambda item: (item["start_frame"], item["end_frame"]))
    merged_segments = _merge_frame_segments(
        candidate_segments,
        max_gap_frames=max(1, int(round(max(float(fps), 1.0) * 0.35))),
    )

    cleaned_segments: list[dict] = []
    for segment in merged_segments[: max(1, int(max_event_segments))]:
        cleaned_segments.append(
            {
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "segment_type": str(segment["segment_type"]),
            }
        )
    return cleaned_segments


def compute_segment_score_stats(
    frame_aligned_scores: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> dict[str, float]:
    scores = resample_scores_to_length(frame_aligned_scores, target_length=max(int(frame_aligned_scores.size), int(end_frame) + 1))
    if scores.size == 0:
        return {"mean": 0.0, "max": 0.0}

    start_frame = max(0, int(start_frame))
    end_frame = min(int(end_frame), int(scores.size - 1))
    if end_frame < start_frame:
        return {"mean": 0.0, "max": 0.0}

    window = scores[start_frame : end_frame + 1]
    if window.size == 0:
        return {"mean": 0.0, "max": 0.0}
    return {
        "mean": round(float(window.mean()), 6),
        "max": round(float(window.max()), 6),
    }

