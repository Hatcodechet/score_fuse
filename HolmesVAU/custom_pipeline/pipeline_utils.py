from __future__ import annotations

import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


CUSTOM_PIPELINE_DIR = Path(__file__).resolve().parent
HOLMES_PROJECT_ROOT = CUSTOM_PIPELINE_DIR.parent
VHUNG_SRC_ROOT = Path("/workspace/VHung/src")
DEFAULT_VIDEO_ROOT = Path("/workspace/test")
DEFAULT_FEATURE_ROOT = Path("/workspace/VHung/data/UCFClipFeatures")
DEFAULT_SCORER_CHECKPOINT = Path("/workspace/VHung/model/model_ucf.pth")
DEFAULT_HOLMES_MODEL_PATH = Path("/workspace/HolmesVAU-2B")
DEFAULT_OUTPUT_ROOT = CUSTOM_PIPELINE_DIR / "outputs"
DEFAULT_PROMPT = "Could you specify the anomaly events present in the video?"

PROMPT_LIST = [
    "Describe the anomaly events observed in the video.",
    "Could you describe the anomaly events observed in the video?",
    "Could you specify the anomaly events present in the video?",
    "Give a description of the detected anomaly events in this video.",
    "Could you give a description of the anomaly events in the video?",
    "Provide a summary of the anomaly events in the video.",
    "Could you provide a summary of the anomaly events in this video?",
    "What details can you provide about the anomaly in the video?",
    "How would you detail the anomaly events found in the video?",
    "How would you describe the particular anomaly events in the video?",
]

NORMAL_PROMPT_LIST = [
    "Describe the events occurring in this video.",
    "Could you describe the events observed in the video?",
    "Provide a detailed description of the actions in this video.",
    "Give a summary of what is happening in this video.",
    "Could you specify the actions and subjects present in the video?",
    "What details can you provide about the events in the video?",
    "How would you describe the scene in this video?",
    "Provide a video-level description of this video.",
    "Summarize the main subjects and key actions in this video.",
    "Describe the main activities unfolding in the video.",
]


def ensure_import_paths() -> None:
    for candidate in (str(HOLMES_PROJECT_ROOT), str(VHUNG_SRC_ROOT)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def resolve_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def iter_video_paths(video_root: Path) -> list[Path]:
    video_paths = []
    for path in video_root.rglob("*.mp4"):
        if not path.is_file():
            continue
        if any(part.startswith("._") or part == "__MACOSX" for part in path.parts):
            continue
        if path.name.startswith("._"):
            continue
        video_paths.append(path)
    return sorted(video_paths)


def limit_paths(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None or limit < 0:
        return paths
    return paths[:limit]


def build_feature_index(feature_root: Path, clip_index: int = 0) -> dict[str, Path]:
    suffix = f"__{clip_index}.npy"
    index: dict[str, Path] = {}
    for feature_path in sorted(feature_root.rglob(f"*{suffix}")):
        stem = feature_path.name[: -len(suffix)]
        index.setdefault(stem, feature_path)
    return index


def match_feature_path(video_path: Path, feature_index: dict[str, Path]) -> Path | None:
    return feature_index.get(video_path.stem)


def video_output_dir(output_root: Path, video_name: str) -> Path:
    return output_root / video_name


def is_normal_video(video_path: Path) -> bool:
    if video_path.stem.startswith("Normal_"):
        return True
    return "Testing_Normal_Videos_Anomaly" in str(video_path)


def extract_video_label(video_name: str) -> str:
    stem = Path(video_name).stem
    if stem.startswith("Normal_") or stem.startswith("Normal_Videos_"):
        return "normal"

    match = re.match(r"([A-Za-z]+(?:_[A-Za-z]+)*)", stem)
    if not match:
        return stem.lower()

    raw = match.group(1).replace("_", " ")
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", raw)
    label = " ".join(words) if words else raw
    return label.strip().lower()


def build_label_guided_anomaly_prompt(video_name: str) -> str:
    label = extract_video_label(video_name)
    return (
        f"Could you specify the anomaly event in this video, knowing that the assigned anomaly label is "
        f"'{label}'? Focus on describing the visible abnormal actions and explain briefly how they match "
        f"the label."
    )


def choose_prompt_for_video(
    video_path: Path,
    rng: random.Random | None = None,
    override_prompt: str | None = None,
) -> str:
    if override_prompt:
        return override_prompt
    rng = rng or random
    prompt_pool = NORMAL_PROMPT_LIST if is_normal_video(video_path) else PROMPT_LIST
    return rng.choice(prompt_pool)


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "srm_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _build_vadclip_model(device):
    ensure_import_paths()

    import option
    from model import CLIPVAD

    args = option.parser.parse_args([])
    model = CLIPVAD(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
        use_tgm=False,
    )
    return model, args


def load_custom_scorer_model(checkpoint_path: Path, device_arg: str = "auto"):
    import torch

    device = resolve_device(device_arg)
    model, args = _build_vadclip_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, args, device, missing, unexpected


def _pad_features(features: np.ndarray, target_length: int) -> np.ndarray:
    if features.shape[0] >= target_length:
        return features
    pad_rows = target_length - features.shape[0]
    return np.pad(features, ((0, pad_rows), (0, 0)), mode="constant", constant_values=0.0)


def split_features_for_model(features: np.ndarray, max_length: int) -> tuple[np.ndarray, np.ndarray, int]:
    features = np.asarray(features, dtype=np.float32)
    original_length = int(features.shape[0])

    if original_length < max_length:
        splits = _pad_features(features, max_length)[None, ...]
    else:
        split_num = int(original_length / max_length) + 1
        chunks = []
        for split_idx in range(split_num):
            start = split_idx * max_length
            end = start + max_length
            chunk = features[start:end]
            chunk = _pad_features(chunk, max_length)
            chunks.append(chunk[None, ...])
        splits = np.concatenate(chunks, axis=0)

    lengths = np.zeros(int(original_length / max_length) + 1, dtype=np.int64)
    remaining = int(original_length)
    for idx in range(len(lengths)):
        if idx == 0 and remaining < max_length:
            lengths[idx] = remaining
        elif idx == 0 and remaining > max_length:
            lengths[idx] = max_length
            remaining -= max_length
        elif remaining > max_length:
            lengths[idx] = max_length
            remaining -= max_length
        else:
            lengths[idx] = remaining

    return splits, lengths, original_length


def infer_temporal_scores(model, features: np.ndarray, device, max_length: int) -> np.ndarray:
    import torch
    from utils.tools import get_batch_mask

    splits, lengths_np, original_length = split_features_for_model(features, max_length=max_length)
    visual = torch.from_numpy(splits).to(device)
    lengths = torch.from_numpy(lengths_np).to(device=device, dtype=torch.int64)
    padding_mask = get_batch_mask(lengths.cpu(), max_length).to(device)

    with torch.no_grad():
        visual_features = model.encode_video(visual, padding_mask, lengths)
        logits1 = model.classifier(visual_features + model.mlp2(visual_features))
        scores = torch.sigmoid(logits1).squeeze(-1).reshape(-1)[:original_length]

    return scores.detach().cpu().numpy().astype(np.float32)


def infer_scores_for_feature_file(
    model,
    feature_path: Path,
    device,
    max_length: int,
) -> np.ndarray:
    features = np.load(feature_path).astype(np.float32)
    return infer_temporal_scores(model, features, device=device, max_length=max_length)


def uniform_sample_indices(num_items: int, select_frames: int) -> list[int]:
    if num_items <= 0:
        return []
    return [int(idx) for idx in np.rint(np.linspace(0, num_items - 1, select_frames))]


def sample_snippet_indices_from_scores(
    scores: np.ndarray,
    select_frames: int = 12,
    tau: float = 0.1,
) -> tuple[np.ndarray, list[int]]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    num_items = int(scores.shape[0])

    if num_items == 0:
        return scores, []

    if num_items <= select_frames or float(scores.sum()) < 1.0:
        return scores, uniform_sample_indices(num_items, select_frames)

    density_scores = scores + tau
    if np.any(density_scores <= 0):
        density_scores = density_scores - density_scores.min() + tau

    if float(density_scores.sum()) <= 0:
        return scores, uniform_sample_indices(num_items, select_frames)

    score_cumsum = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(density_scores, dtype=np.float32)],
        axis=0,
    )

    if np.any(np.diff(score_cumsum) <= 0):
        return scores, uniform_sample_indices(num_items, select_frames)

    scale_x = np.linspace(1, float(score_cumsum[-1]), select_frames)
    sampled = np.interp(scale_x, score_cumsum, np.arange(num_items + 1))
    indices = [min(num_items - 1, max(0, int(idx))) for idx in sampled]
    return scores, indices


def snippet_indices_to_frame_indices(
    snippet_indices: Iterable[int],
    num_frames: int,
    snippet_stride: int = 16,
    frame_offset: int = 0,
) -> np.ndarray:
    frame_indices = []
    max_frame = max(0, num_frames - 1)
    for snippet_idx in snippet_indices:
        frame_idx = int(snippet_idx) * snippet_stride + frame_offset
        frame_indices.append(min(max_frame, max(0, frame_idx)))
    return np.asarray(frame_indices, dtype=np.int64)


def load_holmes_model(model_path: Path, device_arg: str = "auto"):
    ensure_import_paths()

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(device_arg)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    load_errors = []
    model = None
    for kwargs in (
        {"use_flash_attn": True},
        {"use_flash_attn": False},
        {},
    ):
        try:
            model = AutoModel.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **kwargs,
            ).eval()
            break
        except Exception as exc:  # pragma: no cover - depends on local runtime
            load_errors.append(f"{kwargs or {'use_flash_attn': 'unset'}} -> {exc}")

    if model is None:
        error_text = "\n".join(load_errors)
        raise RuntimeError(f"Failed to load HolmesVAU model from {model_path}:\n{error_text}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
    )
    model = model.to(device)
    generation_config = {"max_new_tokens": 256, "do_sample": False}
    return model, tokenizer, generation_config, device


def _get_pixel_values(video_reader, frame_indices, input_size: int = 448, max_num: int = 1):
    import torch
    from PIL import Image
    from holmesvau.internvl_utils import build_transform, dynamic_preprocess

    transform = build_transform(input_size=input_size)
    pixel_values_list = []
    num_patches_list = []

    for frame_index in frame_indices:
        image = Image.fromarray(video_reader[int(frame_index)].asnumpy()).convert("RGB")
        processed = dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )
        pixel_values = [transform(tile) for tile in processed]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list


def generate_description_from_frames(
    model,
    tokenizer,
    generation_config: dict,
    video_path: Path,
    frame_indices: Iterable[int],
    prompt: str = DEFAULT_PROMPT,
) -> str:
    from decord import VideoReader, cpu

    frame_indices = [int(idx) for idx in frame_indices]
    if not frame_indices:
        raise ValueError(f"No frame indices were provided for {video_path}.")

    video_reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    pixel_values, num_patches_list = _get_pixel_values(video_reader, frame_indices)
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    pixel_values = pixel_values.to(dtype=model_dtype, device=model_device)

    history = None
    video_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
    question = video_prefix + prompt
    response, _ = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=history,
        return_history=True,
    )
    return response


def save_sampled_frames_grid(
    video_path: Path,
    frame_indices: Iterable[int],
    output_path: Path,
    max_cols: int = 4,
) -> None:
    from decord import VideoReader, cpu
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = [int(idx) for idx in frame_indices]
    video_reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    frames = [video_reader[idx].asnumpy() for idx in frame_indices]
    if not frames:
        raise ValueError(f"No frames available for visualization: {video_path}")

    cols = min(max_cols, len(frames))
    rows = int(math.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_array = np.array(axes, dtype=object).reshape(rows, cols)

    for axis in axes_array.flat:
        axis.axis("off")

    for axis, frame, frame_index in zip(axes_array.flat, frames, frame_indices):
        axis.imshow(frame)
        axis.set_title(f"frame {frame_index}")
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_anomaly_scores_plot(
    scores: np.ndarray,
    selected_snippet_indices: Iterable[int],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    selected_snippet_indices = [int(idx) for idx in selected_snippet_indices]

    fig, axis = plt.subplots(figsize=(10, 3))
    axis.plot(np.arange(len(scores)), scores, color="#1f77b4", linewidth=1.8)
    for idx in selected_snippet_indices:
        axis.axvline(idx, color="#d62728", linestyle="--", linewidth=1.1)
    axis.set_xlabel("snippet index")
    axis.set_ylabel("anomaly score")
    axis.set_title("Temporal anomaly scores with ATS selections")
    axis.set_xlim(0, max(0, len(scores) - 1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
