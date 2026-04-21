"""Generate segment-level HolmesVAU descriptions for videos under /workspace/test.

Example:
  python3 /workspace/score_fuse/new_solution/run_segment_description.py \
    --video-root /workspace/test \
    --output-root /workspace/score_fuse/new_solution/outputs \
    --checkpoint /workspace/model_ucf.pth \
    --feature-root /workspace/UCFClipFeatures \
    --scorer-source-root /workspace/VadCLIP/src \
    --holmes-model-path ppxin321/HolmesVAU-2B
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from anomaly_wrapper import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_SCORER_SOURCE_ROOT,
    AnomalyScoreService,
)
from holmes_wrapper import DEFAULT_HOLMES_MODEL, load_holmes_model, generate_segment_description
from segment_utils import compute_segment_score_stats, propose_segments, resample_scores_to_length
from video_utils import extract_segment_rgb_frames, find_mp4_videos, load_video_metadata


PROMPT = (
    "Describe the key action happening in this short video segment.\n"
    "Focus on visible actions, interactions, and any unusual or suspicious behavior.\n"
    "Be concise, specific, and avoid hallucinating details that are not clearly visible."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one HolmesVAU description per anomaly-driven segment.")
    parser.add_argument("--video-root", type=Path, default=Path("/workspace/test"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/score_fuse/new_solution/outputs"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--score-root", type=Path, default=None, help="Optional directory of precomputed .npy/.json score files.")
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
        help="Directory of feature .npy files for checkpoint scoring.",
    )
    parser.add_argument(
        "--scorer-source-root",
        type=Path,
        default=DEFAULT_SCORER_SOURCE_ROOT,
        help="Source root containing option.py/ucf_option.py, model.py, and utils/tools.py for the checkpoint scorer.",
    )
    parser.add_argument("--holmes-model-path", type=str, default=DEFAULT_HOLMES_MODEL)
    parser.add_argument("--holmes-device", type=str, default="auto")
    parser.add_argument("--scorer-device", type=str, default="auto")
    parser.add_argument("--segment-frames", type=int, default=12, help="Max frames sampled per segment for HolmesVAU.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--allow-uniform-score-fallback", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-uniform-segments", type=int, default=4)
    parser.add_argument("--max-uniform-segments", type=int, default=6)
    parser.add_argument("--max-event-segments", type=int, default=5)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--low-peak-threshold", type=float, default=0.35)
    parser.add_argument("--relative-event-threshold", type=float, default=0.5)
    parser.add_argument("--min-event-len", type=int, default=4)
    parser.add_argument("--context-expand", type=int, default=3)
    return parser.parse_args()


def _relative_video_json_path(video_path: Path, video_root: Path, output_root: Path) -> Path:
    try:
        relative = video_path.relative_to(video_root)
        return output_root / relative.parent / f"{relative.stem}.segment_descriptions.json"
    except ValueError:
        return output_root / f"{video_path.stem}.segment_descriptions.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_segment_entry(
    segment_id: int,
    segment: dict[str, Any],
    fps: float,
    frame_aligned_scores,
    sampled_frame_indices: list[int],
    description: str,
) -> dict[str, Any]:
    start_frame = int(segment["start_frame"])
    end_frame = int(segment["end_frame"])
    return {
        "segment_id": int(segment_id),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": round(float(start_frame / fps), 6) if fps > 0 else 0.0,
        "end_time_sec": round(float(end_frame / fps), 6) if fps > 0 else 0.0,
        "segment_type": str(segment["segment_type"]),
        "sampled_frame_indices": [int(index) for index in sampled_frame_indices],
        "score_stats": compute_segment_score_stats(
            frame_aligned_scores=frame_aligned_scores,
            start_frame=start_frame,
            end_frame=end_frame,
        ),
        "description": description,
    }


def process_video(
    video_path: Path,
    args: argparse.Namespace,
    score_service: AnomalyScoreService,
    holmes_bundle,
) -> dict[str, Any]:
    metadata = load_video_metadata(video_path)
    score_result = score_service.get_anomaly_scores_for_video(video_path, num_frames=metadata.num_frames)

    # The scorer may operate on coarse temporal snippets. We resample to one score per
    # decoded frame before computing segment statistics so frame indices stay aligned.
    frame_aligned_scores = resample_scores_to_length(score_result.scores, metadata.num_frames)
    segments = propose_segments(
        anomaly_scores=score_result.scores,
        num_frames=metadata.num_frames,
        fps=metadata.fps,
        min_uniform_segments=args.min_uniform_segments,
        max_uniform_segments=args.max_uniform_segments,
        max_event_segments=args.max_event_segments,
        smooth_window=args.smooth_window,
        low_peak_threshold=args.low_peak_threshold,
        relative_event_threshold=args.relative_event_threshold,
        min_event_len=args.min_event_len,
        context_expand=args.context_expand,
    )

    if not segments:
        segments = [{"start_frame": 0, "end_frame": max(0, metadata.num_frames - 1), "segment_type": "uniform"}]

    segment_entries: list[dict[str, Any]] = []
    for segment_id, segment in enumerate(segments):
        frames, sampled_frame_indices = extract_segment_rgb_frames(
            video_path=video_path,
            start_frame=int(segment["start_frame"]),
            end_frame=int(segment["end_frame"]),
            num_samples=args.segment_frames,
        )
        if not frames:
            raise ValueError(
                f"Failed to decode frames for segment {segment_id} of {video_path.name}: "
                f"{segment['start_frame']}..{segment['end_frame']}"
            )

        description = generate_segment_description(
            holmes_bundle.model,
            holmes_bundle.processor,
            frames,
            prompt=PROMPT,
            generation_config=holmes_bundle.generation_config,
        )
        segment_entries.append(
            _build_segment_entry(
                segment_id=segment_id,
                segment=segment,
                fps=metadata.fps,
                frame_aligned_scores=frame_aligned_scores,
                sampled_frame_indices=sampled_frame_indices,
                description=description,
            )
        )

    return {
        "video_path": str(video_path),
        "relative_video_path": str(video_path.relative_to(args.video_root)),
        "fps": round(float(metadata.fps), 6),
        "num_frames": int(metadata.num_frames),
        "duration_sec": round(float(metadata.duration_sec), 6),
        "score_source": score_result.source,
        "score_metadata": score_result.metadata,
        "score_alignment": {
            "original_score_length": int(score_result.scores.size),
            "aligned_to_num_frames": int(metadata.num_frames),
            "resampled": bool(score_result.scores.size != metadata.num_frames),
        },
        "prompt": PROMPT,
        "segments": segment_entries,
    }


def main() -> None:
    args = parse_args()
    video_paths = find_mp4_videos(args.video_root)
    if args.limit is not None and args.limit >= 0:
        video_paths = video_paths[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)

    score_service = AnomalyScoreService(
        video_root=args.video_root,
        checkpoint_path=args.checkpoint,
        score_root=args.score_root,
        feature_root=args.feature_root,
        scorer_source_root=args.scorer_source_root,
        scorer_device=args.scorer_device,
        allow_uniform_fallback=args.allow_uniform_score_fallback,
    )
    holmes_bundle = load_holmes_model(
        model_path=args.holmes_model_path,
        device=args.holmes_device,
        max_new_tokens=args.max_new_tokens,
    )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for video_path in video_paths:
        output_path = _relative_video_json_path(video_path, args.video_root, args.output_root)
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {video_path} -> {output_path} already exists")
            continue

        print(f"[start] {video_path}")
        try:
            payload = process_video(
                video_path=video_path,
                args=args,
                score_service=score_service,
                holmes_bundle=holmes_bundle,
            )
            _write_json(output_path, payload)
            results.append(payload)
            print(f"[ok] {video_path} -> {output_path}")
        except Exception as exc:
            error_payload = {
                "video_path": str(video_path),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
            }
            errors.append(error_payload)
            print(f"[error] {video_path}: {exc}")

    combined_payload = {
        "video_root": str(args.video_root),
        "output_root": str(args.output_root),
        "num_success": len(results),
        "num_error": len(errors),
        "results": results,
        "errors": errors,
    }
    _write_json(args.output_root / "combined_segment_descriptions.json", combined_payload)


if __name__ == "__main__":
    main()
