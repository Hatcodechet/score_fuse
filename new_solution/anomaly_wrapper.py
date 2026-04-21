from __future__ import annotations

import json
import math
import sys
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CHECKPOINT_PATH = Path("/workspace/model_ucf.pth")
DEFAULT_FEATURE_ROOT = Path("/workspace/UCFClipFeatures")
DEFAULT_SCORER_SOURCE_ROOT = Path("/workspace/VadCLIP/src")


@dataclass
class ScoreResult:
    scores: np.ndarray
    source: str
    metadata: dict[str, Any]


def _resolve_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _extract_state_dict(checkpoint: object) -> object:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "srm_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _pad_features(features: np.ndarray, target_length: int) -> np.ndarray:
    if features.shape[0] >= target_length:
        return features
    padding_rows = target_length - features.shape[0]
    return np.pad(features, ((0, padding_rows), (0, 0)), mode="constant", constant_values=0.0)


def _split_features_for_model(features: np.ndarray, max_length: int) -> tuple[np.ndarray, np.ndarray, int]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {features.shape}")

    original_length = int(features.shape[0])
    if original_length <= 0:
        feature_dim = int(features.shape[1]) if features.ndim == 2 else 1
        return np.zeros((1, max_length, feature_dim), dtype=np.float32), np.zeros((1,), dtype=np.int64), 0

    if original_length < max_length:
        splits = _pad_features(features, max_length)[None, ...]
    else:
        split_num = int(original_length / max_length) + 1
        chunks = []
        for split_idx in range(split_num):
            start = split_idx * max_length
            end = start + max_length
            chunk = _pad_features(features[start:end], max_length)
            chunks.append(chunk[None, ...])
        splits = np.concatenate(chunks, axis=0)

    lengths = np.zeros((int(original_length / max_length) + 1,), dtype=np.int64)
    remaining = int(original_length)
    for index in range(lengths.shape[0]):
        if index == 0 and remaining < max_length:
            lengths[index] = remaining
        elif remaining > max_length:
            lengths[index] = max_length
            remaining -= max_length
        else:
            lengths[index] = remaining
    return splits, lengths, original_length


class AnomalyScoreService:
    """Loads per-video anomaly scores from files or a project-specific scorer checkpoint.

    The checkpoint at `/workspace/model_ucf.pth` appears to belong to a feature-based
    anomaly scoring project. This wrapper reuses that style when the original source
    tree and precomputed feature `.npy` files are available. If they are not available,
    callers can still provide external score files and run the rest of the pipeline.
    """

    def __init__(
        self,
        video_root: Path,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        score_root: Path | None = None,
        feature_root: Path | None = None,
        scorer_source_root: Path | None = None,
        scorer_device: str = "auto",
        allow_uniform_fallback: bool = False,
    ) -> None:
        self.video_root = Path(video_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.score_root = Path(score_root) if score_root else None
        self.feature_root = Path(feature_root) if feature_root else None
        self.scorer_source_root = Path(scorer_source_root) if scorer_source_root else None
        self.scorer_device = scorer_device
        self.allow_uniform_fallback = allow_uniform_fallback

        self._score_index = self._build_score_index(self.score_root) if self.score_root else {}
        self._feature_index = self._build_feature_index(self.feature_root) if self.feature_root else {}
        self._scorer_bundle: tuple[object, Any, Any] | None = None

    def get_anomaly_scores_for_video(
        self,
        video_path: str | Path,
        num_frames: int | None = None,
    ) -> ScoreResult:
        video_path = Path(video_path)

        score_result = self._load_scores_from_files(video_path)
        if score_result is not None:
            return score_result

        checkpoint_error: str | None = None
        if self.checkpoint_path.exists():
            try:
                return self._infer_scores_with_checkpoint(video_path)
            except Exception as exc:
                checkpoint_error = str(exc)

        if self.allow_uniform_fallback:
            fallback_length = max(1, int(math.ceil(max(int(num_frames or 1), 1) / 16.0)))
            return ScoreResult(
                scores=np.zeros((fallback_length,), dtype=np.float32),
                source="uniform_fallback",
                metadata={
                    "warning": (
                        "Falling back to flat zero anomaly scores because neither an external "
                        "score file nor a usable checkpoint interface was available."
                    )
                },
            )

        raise RuntimeError(
            "Unable to obtain anomaly scores for "
            f"{video_path}. Provide --score-root with existing score files, or provide the "
            "original scorer source tree plus precomputed features for /workspace/model_ucf.pth. "
            f"Checkpoint error: {checkpoint_error or 'not attempted'}"
        )

    def _build_score_index(self, score_root: Path) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in sorted(score_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".npy", ".json"}:
                continue
            index.setdefault(path.stem.replace("_scores", ""), path)
        return index

    def _load_scores_from_files(self, video_path: Path) -> ScoreResult | None:
        if not self.score_root:
            return None

        relative_candidates: list[Path] = []
        try:
            relative_video = video_path.relative_to(self.video_root)
            relative_candidates.extend(
                [
                    self.score_root / relative_video.with_suffix(".npy"),
                    self.score_root / relative_video.with_name(f"{relative_video.stem}_scores.npy"),
                    self.score_root / relative_video.with_suffix(".json"),
                    self.score_root / relative_video.with_name(f"{relative_video.stem}_scores.json"),
                ]
            )
        except ValueError:
            pass

        relative_candidates.extend(
            [
                self.score_root / f"{video_path.stem}.npy",
                self.score_root / f"{video_path.stem}_scores.npy",
                self.score_root / f"{video_path.stem}.json",
                self.score_root / f"{video_path.stem}_scores.json",
            ]
        )

        for candidate in relative_candidates:
            if candidate.exists():
                scores = self._read_score_file(candidate)
                return ScoreResult(
                    scores=scores,
                    source="score_file",
                    metadata={"score_path": str(candidate)},
                )

        indexed = self._score_index.get(video_path.stem)
        if indexed and indexed.exists():
            scores = self._read_score_file(indexed)
            return ScoreResult(
                scores=scores,
                source="score_file_index",
                metadata={"score_path": str(indexed)},
            )
        return None

    def _read_score_file(self, score_path: Path) -> np.ndarray:
        if score_path.suffix.lower() == ".npy":
            return np.load(score_path).astype(np.float32).reshape(-1)

        payload = json.loads(score_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return np.asarray(payload, dtype=np.float32).reshape(-1)
        if isinstance(payload, dict):
            for key in ("scores", "anomaly_scores"):
                if key in payload:
                    return np.asarray(payload[key], dtype=np.float32).reshape(-1)
        raise ValueError(f"Unsupported score JSON format: {score_path}")

    def _build_feature_index(self, feature_root: Path) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for feature_path in sorted(feature_root.rglob("*.npy")):
            stem = feature_path.stem
            if "__" in stem:
                stem = stem.split("__", 1)[0]
            index.setdefault(stem, feature_path)
        return index

    def _discover_feature_root(self) -> Path:
        if self.feature_root and self.feature_root.exists():
            return self.feature_root

        candidate_roots = [
            DEFAULT_FEATURE_ROOT,
            Path("/workspace/VHung/data/UCFClipFeatures"),
        ]
        for candidate in candidate_roots:
            if candidate.exists():
                self.feature_root = candidate
                self._feature_index = self._build_feature_index(candidate)
                return candidate

        workspace_root = Path("/workspace")
        for candidate in sorted(workspace_root.rglob("UCFClipFeatures")):
            if candidate.is_dir():
                self.feature_root = candidate
                self._feature_index = self._build_feature_index(candidate)
                return candidate

        raise FileNotFoundError(
            "Could not discover the precomputed UCF feature directory automatically. "
            "Pass --feature-root pointing to the folder that contains files like `Abuse020_x264__0.npy`."
        )

    def _discover_scorer_source_root(self) -> Path:
        if (
            self.scorer_source_root
            and self.scorer_source_root.exists()
            and (self.scorer_source_root / "model.py").exists()
            and (self.scorer_source_root / "utils" / "tools.py").exists()
            and (
                (self.scorer_source_root / "option.py").exists()
                or (self.scorer_source_root / "ucf_option.py").exists()
            )
        ):
            return self.scorer_source_root

        candidate_roots = [
            DEFAULT_SCORER_SOURCE_ROOT,
            Path("/workspace/VHung/src"),
        ]
        for candidate in candidate_roots:
            if (
                candidate.exists()
                and (candidate / "model.py").exists()
                and (candidate / "utils" / "tools.py").exists()
                and ((candidate / "option.py").exists() or (candidate / "ucf_option.py").exists())
            ):
                self.scorer_source_root = candidate
                return candidate

        workspace_root = Path("/workspace")
        for option_name in ("option.py", "ucf_option.py"):
            for option_path in workspace_root.rglob(option_name):
                candidate_root = option_path.parent
                if (candidate_root / "model.py").exists() and (candidate_root / "utils" / "tools.py").exists():
                    self.scorer_source_root = candidate_root
                    return candidate_root
        raise FileNotFoundError(
            "Could not discover scorer source root automatically. "
            "Pass --scorer-source-root pointing to the project that defines model.py and utils/tools.py "
            "plus either option.py or ucf_option.py."
        )

    def _load_option_module(self, scorer_source_root: Path):
        import importlib

        for module_name in ("option", "ucf_option"):
            try:
                return importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            f"Neither option.py nor ucf_option.py could be imported from {scorer_source_root}"
        )

    def _instantiate_clipvad(self, clipvad_cls, args, device):
        init_kwargs = {
            "num_class": args.classes_num,
            "embed_dim": args.embed_dim,
            "visual_length": args.visual_length,
            "visual_width": args.visual_width,
            "visual_head": args.visual_head,
            "visual_layers": args.visual_layers,
            "attn_window": args.attn_window,
            "prompt_prefix": args.prompt_prefix,
            "prompt_postfix": args.prompt_postfix,
            "device": device,
        }
        signature = inspect.signature(clipvad_cls.__init__)
        if "use_tgm" in signature.parameters:
            init_kwargs["use_tgm"] = False
        return clipvad_cls(**init_kwargs)

    def _load_checkpoint_scorer(self) -> tuple[object, Any, Any]:
        if self._scorer_bundle is not None:
            return self._scorer_bundle

        scorer_source_root = self._discover_scorer_source_root()
        source_root_str = str(scorer_source_root)
        if source_root_str not in sys.path:
            sys.path.insert(0, source_root_str)

        import torch
        from model import CLIPVAD

        option = self._load_option_module(scorer_source_root)

        device = _resolve_device(self.scorer_device)
        args = option.parser.parse_args([])

        model = self._instantiate_clipvad(CLIPVAD, args, device)
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(_extract_state_dict(checkpoint), strict=False)
        model = model.to(device)
        model.eval()
        self._scorer_bundle = (model, args, device)
        return self._scorer_bundle

    def _infer_scores_with_checkpoint(self, video_path: Path) -> ScoreResult:
        feature_root = self._discover_feature_root()

        feature_path = self._feature_index.get(video_path.stem)
        if feature_path is None or not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found for {video_path.stem} under {feature_root}. "
                "TODO: plug in the original project's raw-video feature extractor if you want "
                "to score videos directly from mp4 files."
            )

        model, args, device = self._load_checkpoint_scorer()
        features = np.load(feature_path).astype(np.float32)
        scores = self._infer_temporal_scores(model, features, device=device, max_length=args.visual_length)
        return ScoreResult(
            scores=scores,
            source="checkpoint_feature_scorer",
            metadata={
                "checkpoint_path": str(self.checkpoint_path),
                "feature_path": str(feature_path),
                "feature_root": str(feature_root),
            },
        )

    def _infer_temporal_scores(
        self,
        model,
        features: np.ndarray,
        device,
        max_length: int,
    ) -> np.ndarray:
        import torch
        from utils.tools import get_batch_mask

        splits, lengths_np, original_length = _split_features_for_model(features, max_length=max_length)
        visual = torch.from_numpy(splits).to(device)
        lengths = torch.from_numpy(lengths_np).to(device=device, dtype=torch.int64)
        padding_mask = get_batch_mask(lengths.cpu(), max_length).to(device)

        with torch.no_grad():
            visual_features = model.encode_video(visual, padding_mask, lengths)
            logits = model.classifier(visual_features + model.mlp2(visual_features))
            scores = torch.sigmoid(logits).squeeze(-1).reshape(-1)[:original_length]
        return scores.detach().cpu().numpy().astype(np.float32)


def get_anomaly_scores_for_video(
    video_path: str,
    service: AnomalyScoreService,
) -> np.ndarray:
    return service.get_anomaly_scores_for_video(video_path).scores
