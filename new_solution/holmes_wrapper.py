from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


HOLMES_PROJECT_ROOT = Path("/workspace/score_fuse/HolmesVAU")
DEFAULT_HOLMES_MODEL = "ppxin321/HolmesVAU-2B"


@dataclass
class HolmesBundle:
    model: object
    processor: object
    generation_config: dict
    device: str


def _ensure_holmes_import_path() -> None:
    project_root = str(HOLMES_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _resolve_device(device_arg: str) -> str:
    import torch

    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_holmes_model(
    model_path: str | Path = DEFAULT_HOLMES_MODEL,
    device: str = "auto",
    max_new_tokens: int = 256,
) -> HolmesBundle:
    _ensure_holmes_import_path()

    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = _resolve_device(device)
    torch_dtype = torch.bfloat16 if str(resolved_device).startswith("cuda") else torch.float32

    load_errors: list[str] = []
    model = None
    for kwargs in ({"use_flash_attn": True}, {"use_flash_attn": False}, {}):
        try:
            model = AutoModel.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **kwargs,
            ).eval()
            break
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            load_errors.append(f"{kwargs or {'use_flash_attn': 'unset'}} -> {exc}")

    if model is None:
        error_details = "\n".join(load_errors)
        if "flash_attn_2_cuda" in error_details or "undefined symbol" in error_details:
            error_details += (
                "\nDetected a broken flash-attn installation. HolmesVAU can run without flash-attn, "
                "but Transformers may still fail to import while the incompatible extension is present. "
                "Uninstall `flash-attn` from this environment, or reinstall a version built for the exact "
                "PyTorch/CUDA combination you are using."
            )
        raise RuntimeError(f"Failed to load HolmesVAU-2B from {model_path}:\n{error_details}")

    processor = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
    )
    model = model.to(resolved_device)
    generation_config = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
    return HolmesBundle(
        model=model,
        processor=processor,
        generation_config=generation_config,
        device=str(resolved_device),
    )


def _frames_to_pixel_values(
    frames: Sequence[np.ndarray],
    input_size: int = 448,
    max_num: int = 1,
):
    import torch
    from PIL import Image
    from holmesvau.internvl_utils import build_transform, dynamic_preprocess

    if not frames:
        raise ValueError("At least one RGB frame is required for HolmesVAU generation.")

    transform = build_transform(input_size=input_size)
    pixel_values_list = []
    num_patches_list: list[int] = []

    for frame in frames:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("Each frame must be an HxWx3 RGB array.")

        image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        processed_tiles = dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )
        pixel_values = [transform(tile) for tile in processed_tiles]
        stacked = torch.stack(pixel_values)
        pixel_values_list.append(stacked)
        num_patches_list.append(int(stacked.shape[0]))

    return torch.cat(pixel_values_list, dim=0), num_patches_list


def generate_segment_description(
    model,
    processor,
    frames: Sequence[np.ndarray],
    prompt: str,
    generation_config: dict | None = None,
) -> str:
    pixel_values, num_patches_list = _frames_to_pixel_values(frames)

    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    pixel_values = pixel_values.to(dtype=model_dtype, device=model_device)

    prompt_prefix = "".join(f"Frame{index + 1}: <image>\n" for index in range(len(num_patches_list)))
    question = prompt_prefix + prompt

    response, _ = model.chat(
        processor,
        pixel_values,
        question,
        generation_config or {"max_new_tokens": 256, "do_sample": False},
        num_patches_list=num_patches_list,
        history=None,
        return_history=True,
    )
    return str(response).strip()
