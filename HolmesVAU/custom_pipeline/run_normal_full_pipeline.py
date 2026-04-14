from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu

from pipeline_utils import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_HOLMES_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCORER_CHECKPOINT,
    DEFAULT_VIDEO_ROOT,
    build_feature_index,
    choose_prompt_for_video,
    generate_description_from_frames,
    infer_scores_for_feature_file,
    is_normal_video,
    iter_video_paths,
    limit_paths,
    load_custom_scorer_model,
    load_holmes_model,
    match_feature_path,
    sample_snippet_indices_from_scores,
    save_anomaly_scores_plot,
    save_sampled_frames_grid,
    save_text,
    snippet_indices_to_frame_indices,
    video_output_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline only for normal videos."
    )
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SCORER_CHECKPOINT)
    parser.add_argument("--holmes-model-path", type=Path, default=DEFAULT_HOLMES_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--select-frames", type=int, default=12)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--snippet-stride", type=int, default=16)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--scorer-device", type=str, default="auto")
    parser.add_argument("--holmes-device", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--description-name",
        type=str,
        default="description_normal.txt",
        help="Description filename written for each normal video.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    videos = limit_paths(iter_video_paths(args.video_root), args.limit)
    normal_videos = [video for video in videos if is_normal_video(video)]
    feature_index = build_feature_index(args.feature_root, clip_index=0)

    scorer_model, scorer_args, scorer_device, missing, unexpected = load_custom_scorer_model(
        args.checkpoint,
        device_arg=args.scorer_device,
    )
    holmes_model, tokenizer, generation_config, holmes_device = load_holmes_model(
        args.holmes_model_path,
        device_arg=args.holmes_device,
    )
    generation_config["max_new_tokens"] = args.max_new_tokens

    print(f"Normal videos to process: {len(normal_videos)}")
    print(f"Custom scorer: {args.checkpoint}")
    print(f"Scorer device: {scorer_device}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    print(f"Holmes model: {args.holmes_model_path}")
    print(f"Holmes device: {holmes_device}")

    for video_path in normal_videos:
        feature_path = match_feature_path(video_path, feature_index)
        if feature_path is None:
            print(f"[skip] feature not found for {video_path.name}")
            continue

        out_dir = video_output_dir(args.output_root, video_path.stem)
        description_path = out_dir / args.description_name
        if description_path.exists() and not args.overwrite:
            print(f"[skip] {args.description_name} already exists for {video_path.name}")
            continue

        scores = infer_scores_for_feature_file(
            scorer_model,
            feature_path,
            device=scorer_device,
            max_length=scorer_args.visual_length,
        )
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
        prompt = choose_prompt_for_video(video_path, rng=rng)

        description = generate_description_from_frames(
            holmes_model,
            tokenizer,
            generation_config,
            video_path=video_path,
            frame_indices=frame_indices.tolist(),
            prompt=prompt,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "scores.npy", scores)
        np.save(out_dir / "selected_indices.npy", frame_indices)
        np.save(out_dir / "selected_snippet_indices.npy", np.asarray(snippet_indices, dtype=np.int64))
        save_text(out_dir / "prompt_normal.txt", prompt + "\n")
        save_text(description_path, description + "\n")
        save_sampled_frames_grid(video_path, frame_indices.tolist(), out_dir / "sampled_frames.png")
        save_anomaly_scores_plot(scores, snippet_indices, out_dir / "anomaly_scores.png")

        print(f"[ok] {video_path.name}")
        print(f"prompt: {prompt}")
        print(description)


if __name__ == "__main__":
    main()
