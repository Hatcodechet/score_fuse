from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu

from pipeline_utils import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VIDEO_ROOT,
    iter_video_paths,
    limit_paths,
    sample_snippet_indices_from_scores,
    snippet_indices_to_frame_indices,
    video_output_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ATS-style frame selection from precomputed temporal scores.")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--select-frames", type=int, default=12)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--snippet-stride", type=int, default=16)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    videos = limit_paths(iter_video_paths(args.video_root), args.limit)

    for video_path in videos:
        out_dir = video_output_dir(args.output_root, video_path.stem)
        score_path = out_dir / "scores.npy"
        if not score_path.exists():
            print(f"[skip] scores not found for {video_path.name}")
            continue

        scores = np.load(score_path)
        _, snippet_indices = sample_snippet_indices_from_scores(
            scores,
            select_frames=args.select_frames,
            tau=args.tau,
        )

        video_reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        frame_indices = snippet_indices_to_frame_indices(
            snippet_indices,
            num_frames=len(video_reader),
            snippet_stride=args.snippet_stride,
            frame_offset=args.frame_offset,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "selected_indices.npy", frame_indices)
        np.save(out_dir / "selected_snippet_indices.npy", np.asarray(snippet_indices, dtype=np.int64))
        print(f"[ok] {video_path.name} -> {out_dir / 'selected_indices.npy'}")


if __name__ == "__main__":
    main()

