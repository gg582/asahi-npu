"""Assemble ANE artefacts and embed them into an ONNX model in one step."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - script execution guard
    PACKAGE_SRC = Path(__file__).resolve().parents[3]
    if str(PACKAGE_SRC) not in sys.path:
        sys.path.insert(0, str(PACKAGE_SRC))

from asahi_ane_llm.metadata import extract_ane_payloads
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
    parser.add_argument(
        "--emit-json-specs",
        type=Path,
        help=(
            "Directory where JSON bundle, tile descriptor, and weights specs "
            "extracted from the converted model should be written."
        ),
    )
    return parser


def _encode_b64(data: bytes) -> str:
    return base64.b64encode(data, altchars=b"-_").decode("ascii")


def _write_json_specs(model_bytes: bytes, output_dir: Path) -> list[tuple[str, Path]]:
    payloads = extract_ane_payloads(model_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[tuple[str, Path]] = []

    def _write(label: str, filename: str, payload: dict[str, object]) -> None:
        path = output_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append((label, path))

    microcode_b64 = _encode_b64(payloads.microcode)
    bundle_payload: dict[str, object] = {
        "microcode_b64": microcode_b64,
        "td_size": payloads.td_size,
        "td_count": payloads.td_count,
    }

    if payloads.tile_descriptors is not None:
        tile_desc_b64 = _encode_b64(payloads.tile_descriptors)
        bundle_payload["tile_descriptors"] = {
            "b64": tile_desc_b64,
            "entry_size": payloads.td_size,
            "count": payloads.td_count,
        }
    if payloads.weights is not None:
        bundle_payload["weights_b64"] = _encode_b64(payloads.weights)

    _write("bundle", "bundle.json", bundle_payload)
    _write("microcode_json", "microcode.json", {"data_b64": microcode_b64})

    if payloads.tile_descriptors is not None:
        _write(
            "tile_desc_json",
            "tile_desc.json",
            {
                "data_b64": _encode_b64(payloads.tile_descriptors),
                "entry_size": payloads.td_size,
                "entry_count": payloads.td_count,
            },
        )

    if payloads.weights is not None:
        _write("weights_json", "weights.json", {"data_b64": _encode_b64(payloads.weights)})

    return outputs


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
        output_path = convert_model(convert_args)
    except ConversionError as exc:
        parser.error(str(exc))

    if args.emit_json_specs:
        exports = _write_json_specs(output_path.read_bytes(), args.emit_json_specs)
        if exports:
            print(
                "Exported ANE metadata JSON specs:\n"
                + "\n".join(f"  {name:>16} -> {path}" for name, path in exports)
            )

    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry point
    raise SystemExit(main())
