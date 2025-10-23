"""Compile ANE microcode, tile descriptors, and weights from JSON specs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..microcode.builder import BuildArtifacts, MicrocodeBuildError, compile_from_spec


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schema", type=Path, help="JSON schema describing the opcode layout")
    parser.add_argument(
        "program",
        type=Path,
        help="JSON program listing the instructions to assemble",
    )
    parser.add_argument(
        "--tile-spec",
        type=Path,
        help="Optional JSON description of the tile descriptor table",
    )
    parser.add_argument(
        "--weights-spec",
        type=Path,
        help="Optional JSON description of weights to embed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory where the compiled artefacts should be written. The helper "
            "always emits the microcode and conditionally writes tile descriptors "
            "and weights when the matching specs are provided."
        ),
    )
    return parser


def run_build(args: argparse.Namespace) -> BuildArtifacts:
    if not args.schema.is_file():
        raise MicrocodeBuildError(f"Schema file '{args.schema}' does not exist")
    if not args.program.is_file():
        raise MicrocodeBuildError(f"Program file '{args.program}' does not exist")

    tile_spec = args.tile_spec
    weights_spec = args.weights_spec

    if tile_spec is not None and not tile_spec.is_file():
        raise MicrocodeBuildError(f"Tile descriptor spec '{tile_spec}' does not exist")
    if weights_spec is not None and not weights_spec.is_file():
        raise MicrocodeBuildError(f"Weights spec '{weights_spec}' does not exist")

    return compile_from_spec(
        schema_path=args.schema,
        program_path=args.program,
        tile_spec_path=tile_spec,
        weights_spec_path=weights_spec,
        output_dir=args.output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        artifacts = run_build(args)
    except MicrocodeBuildError as exc:
        parser.error(str(exc))

    print("ANE microcode compilation complete:")
    for name, path in artifacts.outputs.items():
        print(f"  {name:>16} -> {path}")

    if not artifacts.outputs:
        print("  (no artefacts were written to disk)")

    td_status = (
        f"size={artifacts.td_size}, count={artifacts.td_count}"
        if artifacts.td_size and artifacts.td_count
        else "not generated"
    )
    weights_status = "present" if artifacts.weights else "not generated"

    print("  microcode bytes :", len(artifacts.microcode))
    print("  tile descriptors:", td_status)
    print("  weights        :", weights_status)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
