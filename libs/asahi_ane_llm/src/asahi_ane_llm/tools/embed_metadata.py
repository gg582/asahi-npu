"""CLI helpers for embedding ANE metadata into ONNX models."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..metadata import MissingAneMetadataError, parse_ane_metadata, with_ane_metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the source ONNX model")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to overwriting the input model.",
    )
    parser.add_argument(
        "--microcode",
        required=True,
        type=Path,
        help="Path to the compiled ANE microcode blob (typically a .tsk file).",
    )
    parser.add_argument(
        "--td-size",
        required=True,
        type=int,
        help="Tile descriptor size in bytes produced by the ANE compiler.",
    )
    parser.add_argument(
        "--td-count",
        required=True,
        type=int,
        help="Number of tile descriptor entries produced by the ANE compiler.",
    )
    parser.add_argument(
        "--tile-descriptors",
        type=Path,
        help="Optional tile descriptor binary to embed alongside the metadata.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Optional ANE weights blob to embed alongside the microcode.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Verify the resulting model contains the expected metadata entries.",
    )
    return parser


def _validate_output(model_bytes: bytes) -> None:
    try:
        parse_ane_metadata(model_bytes)
    except MissingAneMetadataError as exc:  # pragma: no cover - defensive path
        raise RuntimeError(
            "The generated model is still missing ANE metadata entries"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.td_size <= 0:
        parser.error("--td-size must be a positive integer")
    if args.td_count <= 0:
        parser.error("--td-count must be a positive integer")

    model_bytes = args.input.read_bytes()
    microcode = args.microcode.read_bytes()
    tile_descriptors = (
        args.tile_descriptors.read_bytes() if args.tile_descriptors else None
    )
    weights = args.weights.read_bytes() if args.weights else None

    updated_model = with_ane_metadata(
        model_bytes,
        microcode=microcode,
        td_size=args.td_size,
        td_count=args.td_count,
        tile_descriptors=tile_descriptors,
        weights=weights,
    )

    output_path = args.output or args.input
    output_path.write_bytes(updated_model)

    if args.validate:
        _validate_output(updated_model)

    print(
        "ANE metadata embedding complete:\n"
        f"  output model : {output_path}\n"
        f"  microcode    : {args.microcode}\n"
        f"  td size      : {args.td_size}\n"
        f"  td count     : {args.td_count}\n"
        f"  tile desc    : {args.tile_descriptors or 'not provided'}\n"
        f"  weights      : {args.weights or 'not provided'}"
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry point
    raise SystemExit(main())
