from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline_utils import (
    DEFAULT_HOLMES_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VIDEO_ROOT,
    build_label_guided_anomaly_prompt,
    generate_description_from_frames,
    is_normal_video,
    iter_video_paths,
    limit_paths,
    load_holmes_model,
    save_text,
    video_output_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate description2.txt only for anomaly videos using existing selected frames "
            "and label-guided anomaly prompts."
        )
    )
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--holmes-model-path", type=Path, default=DEFAULT_HOLMES_MODEL_PATH)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    videos = limit_paths(iter_video_paths(args.video_root), args.limit)
    anomaly_videos = [video for video in videos if not is_normal_video(video)]

    model, tokenizer, generation_config, device = load_holmes_model(
        args.holmes_model_path,
        device_arg=args.device,
    )
    generation_config["max_new_tokens"] = args.max_new_tokens

    print(f"Loaded HolmesVAU model: {args.holmes_model_path}")
    print(f"Device: {device}")
    print(f"Anomaly videos to process: {len(anomaly_videos)}")

    for video_path in anomaly_videos:
        out_dir = video_output_dir(args.output_root, video_path.stem)
        frame_index_path = out_dir / "selected_indices.npy"
        description2_path = out_dir / "description2.txt"

        if not frame_index_path.exists():
            print(f"[skip] selected indices not found for {video_path.name}")
            continue

        if description2_path.exists() and not args.overwrite:
            print(f"[skip] description2 already exists for {video_path.name}")
            continue

        frame_indices = np.load(frame_index_path).tolist()
        prompt = build_label_guided_anomaly_prompt(video_path.stem)
        description = generate_description_from_frames(
            model,
            tokenizer,
            generation_config,
            video_path=video_path,
            frame_indices=frame_indices,
            prompt=prompt,
        )

        save_text(description2_path, description + "\n")
        save_text(out_dir / "prompt2.txt", prompt + "\n")

        print(f"[ok] {video_path.name}")
        print(f"prompt: {prompt}")
        print(description)


if __name__ == "__main__":
    main()
