from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from pipeline_utils import (
    DEFAULT_HOLMES_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VIDEO_ROOT,
    choose_prompt_for_video,
    generate_description_from_frames,
    iter_video_paths,
    limit_paths,
    load_holmes_model,
    save_anomaly_scores_plot,
    save_sampled_frames_grid,
    save_text,
    video_output_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HolmesVAU descriptions from preselected frame indices.")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--holmes-model-path", type=Path, default=DEFAULT_HOLMES_MODEL_PATH)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    videos = limit_paths(iter_video_paths(args.video_root), args.limit)
    model, tokenizer, generation_config, device = load_holmes_model(
        args.holmes_model_path,
        device_arg=args.device,
    )
    generation_config["max_new_tokens"] = args.max_new_tokens

    print(f"Loaded HolmesVAU model: {args.holmes_model_path}")
    print(f"Device: {device}")

    for video_path in videos:
        out_dir = video_output_dir(args.output_root, video_path.stem)
        frame_index_path = out_dir / "selected_indices.npy"
        snippet_index_path = out_dir / "selected_snippet_indices.npy"
        score_path = out_dir / "scores.npy"

        if not frame_index_path.exists():
            print(f"[skip] selected indices not found for {video_path.name}")
            continue

        frame_indices = np.load(frame_index_path).tolist()
        snippet_indices = np.load(snippet_index_path).tolist() if snippet_index_path.exists() else []
        scores = np.load(score_path) if score_path.exists() else None
        prompt = choose_prompt_for_video(
            video_path,
            rng=rng,
            override_prompt=args.prompt,
        )

        description = generate_description_from_frames(
            model,
            tokenizer,
            generation_config,
            video_path=video_path,
            frame_indices=frame_indices,
            prompt=prompt,
        )
        save_text(out_dir / "prompt.txt", prompt + "\n")
        save_text(out_dir / "description.txt", description + "\n")
        save_sampled_frames_grid(video_path, frame_indices, out_dir / "sampled_frames.png")
        if scores is not None:
            save_anomaly_scores_plot(scores, snippet_indices, out_dir / "anomaly_scores.png")

        print(f"[ok] {video_path.name}")
        print(f"prompt: {prompt}")
        print(description)


if __name__ == "__main__":
    main()
