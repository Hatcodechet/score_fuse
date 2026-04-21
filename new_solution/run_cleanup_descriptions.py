"""Clean segment description JSON files and write sibling .cleanup.json outputs."""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from pathlib import Path
from typing import Any

from cleanup_wrapper import (
    DEFAULT_CLEANUP_MODEL,
    DEFAULT_CLEANUP_PROMPT,
    DescriptionCleanupModel,
    build_cleanup_model,
)


DEFAULT_INPUT_ROOT = Path("/workspace/score_fuse/new_solution/outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean segment-level description JSON files and save sibling .cleanup.json files."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--cleanup-backend",
        type=str,
        default="openrouter",
        help="Cleanup backend: openrouter, gemini, or passthrough.",
    )
    parser.add_argument(
        "--cleanup-model",
        type=str,
        default=DEFAULT_CLEANUP_MODEL,
        help="Model name for the selected backend. For OpenRouter, use provider/model format.",
    )
    parser.add_argument("--cleanup-prompt", type=str, default=DEFAULT_CLEANUP_PROMPT)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retries after the initial request when a 429 response is returned.",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=20.0,
        help="Base backoff in seconds for retryable 429 responses.",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=60.0,
        help="Maximum backoff in seconds for retryable 429 responses.",
    )
    return parser.parse_args()


def find_segment_description_files(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    files: list[Path] = []
    for path in sorted(input_root.rglob("*.segment_descriptions.json")):
        if not path.is_file():
            continue
        if path.name.endswith(".cleanup.json"):
            continue
        files.append(path)
    return files


def get_cleanup_output_path(input_path: Path) -> Path:
    suffix = ".segment_descriptions.json"
    if not input_path.name.endswith(suffix):
        raise ValueError(f"Unexpected input filename: {input_path}")
    output_name = f"{input_path.name[: -len(suffix)]}.cleanup.json"
    return input_path.with_name(output_name)


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, got {type(payload).__name__}")
    return payload


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(f"{serialized}\n", encoding="utf-8")


def clean_segment_description(description: Any, cleanup_model: DescriptionCleanupModel) -> str:
    raw_text = str(description or "").strip()
    if not raw_text:
        return ""
    cleaned = cleanup_model.cleanup_description(raw_text)
    return str(cleaned).strip()


def clean_payload(payload: dict[str, Any], cleanup_model: DescriptionCleanupModel) -> dict[str, Any]:
    output_payload = copy.deepcopy(payload)
    segments = output_payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Expected `segments` to be a list.")

    cleaned_descriptions: list[str] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Expected segment {index} to be an object.")

        cleaned_description = clean_segment_description(segment.get("description", ""), cleanup_model)
        segment["cleaned_description"] = cleaned_description
        cleaned_descriptions.append(cleaned_description)

    output_payload["cleaned_descriptions"] = cleaned_descriptions
    return output_payload


def process_file(
    input_path: Path,
    output_path: Path,
    cleanup_model: DescriptionCleanupModel,
    overwrite: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] {input_path} -> {output_path} already exists")
        return

    payload = read_json_file(input_path)
    cleaned_payload = clean_payload(payload, cleanup_model)
    write_json_file(output_path, cleaned_payload)
    print(f"[ok] {input_path} -> {output_path}")


def main() -> None:
    args = parse_args()
    input_files = find_segment_description_files(args.input_root)
    if args.limit is not None and args.limit >= 0:
        input_files = input_files[: args.limit]

    cleanup_model = build_cleanup_model(
        backend=args.cleanup_backend,
        model_name=args.cleanup_model,
        prompt=args.cleanup_prompt,
        timeout_seconds=args.request_timeout,
        max_retries=args.max_retries,
        backoff_base_seconds=args.backoff_base,
        max_backoff_seconds=args.max_backoff,
    )

    num_success = 0
    errors: list[dict[str, str]] = []

    for input_path in input_files:
        output_path = get_cleanup_output_path(input_path)
        try:
            process_file(
                input_path=input_path,
                output_path=output_path,
                cleanup_model=cleanup_model,
                overwrite=args.overwrite,
            )
            if output_path.exists():
                num_success += 1
        except Exception as exc:
            errors.append(
                {
                    "input_path": str(input_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )
            print(f"[error] {input_path}: {exc}")

    print(
        json.dumps(
            {
                "input_root": str(args.input_root),
                "num_discovered": len(input_files),
                "num_success": num_success,
                "num_error": len(errors),
                "errors": errors,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
