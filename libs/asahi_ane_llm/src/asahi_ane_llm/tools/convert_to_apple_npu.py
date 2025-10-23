"""Assemble ANE artefacts and embed them into an ONNX model in one step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - script execution guard
    PACKAGE_SRC = Path(__file__).resolve().parents[3]
    if str(PACKAGE_SRC) not in sys.path:
        sys.path.insert(0, str(PACKAGE_SRC))

from .convert_to_ane import ConversionError, convert_model


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the source ONNX model")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Path for the converted model. Defaults to '<input>_ane.onnx' next to "
            "the source model."
        ),
    )
    parser.add_argument(
        "--ane-schema",
        required=True,
        type=Path,
        help="JSON schema describing the ANE opcode layout",
    )
    parser.add_argument(
        "--ane-program",
        required=True,
        type=Path,
        help="JSON program listing the instructions to assemble",
    )
    parser.add_argument(
        "--ane-tile-spec",
        type=Path,
        help=(
            "Optional JSON description of the tile descriptor table. Provide this "
            "when you want the helper to emit tile descriptors automatically."
        ),
    )
    parser.add_argument(
        "--ane-weights-spec",
        type=Path,
        help="Optional JSON description of weights to embed alongside the microcode.",
    )
    parser.add_argument(
        "--ane-output-dir",
        type=Path,
        help="Directory where the assembled artefacts should be written",
    )
    parser.add_argument(
        "--td-size",
        type=int,
        help="Tile descriptor entry size in bytes when no tile spec is provided.",
    )
    parser.add_argument(
        "--td-count",
        type=int,
        help="Tile descriptor entry count when no tile spec is provided.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip parsing the resulting model to double check the metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    convert_args = argparse.Namespace(
        input=args.input,
        output=args.output,
        bundle=None,
        microcode=None,
        weights=None,
        tile_descriptors=None,
        td_size=args.td_size,
        td_count=args.td_count,
        artifact_root=None,
        ane_schema=args.ane_schema,
        ane_program=args.ane_program,
        ane_tile_spec=args.ane_tile_spec,
        ane_weights_spec=args.ane_weights_spec,
        ane_output_dir=args.ane_output_dir,
        skip_validation=args.skip_validation,
    )

    try:
        convert_model(convert_args)
    except ConversionError as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry point
    raise SystemExit(main())
