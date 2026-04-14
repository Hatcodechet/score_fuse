from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline_utils import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCORER_CHECKPOINT,
    DEFAULT_VIDEO_ROOT,
    build_feature_index,
    infer_scores_for_feature_file,
    iter_video_paths,
    limit_paths,
    load_custom_scorer_model,
    match_feature_path,
    video_output_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer temporal anomaly scores from UCF feature files.")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SCORER_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    videos = limit_paths(iter_video_paths(args.video_root), args.limit)
    feature_index = build_feature_index(args.feature_root, clip_index=0)
    model, model_args, device, missing, unexpected = load_custom_scorer_model(
        args.checkpoint,
        device_arg=args.device,
    )

    print(f"Loaded scorer checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    for video_path in videos:
        feature_path = match_feature_path(video_path, feature_index)
        if feature_path is None:
            print(f"[skip] feature not found for {video_path.name}")
            continue

        scores = infer_scores_for_feature_file(
            model,
            feature_path,
            device=device,
            max_length=model_args.visual_length,
        )

        out_dir = video_output_dir(args.output_root, video_path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "scores.npy", scores)
        print(f"[ok] {video_path.name} -> {out_dir / 'scores.npy'}")


if __name__ == "__main__":
    main()

