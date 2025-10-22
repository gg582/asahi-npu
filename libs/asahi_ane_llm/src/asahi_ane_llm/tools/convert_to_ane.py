"""Convert a generic ONNX model into an Asahi ANE ready payload."""

from __future__ import annotations

import base64
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..metadata import MissingAneMetadataError, parse_ane_metadata, with_ane_metadata


class ConversionError(RuntimeError):
    """Raised when the conversion inputs are insufficient or inconsistent."""


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the source ONNX model")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Path for the converted model. Defaults to '<input>_ane.onnx' "
            "next to the source model."
        ),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help=(
            "Path to a JSON bundle that already contains the compiled ANE microcode "
            "and tile descriptor metadata. The helper extracts the metadata from the "
            "bundle and ignores --microcode/--weights/--tile-descriptors when the "
            "option is supplied."
        ),
    )
    parser.add_argument(
        "--microcode",
        type=Path,
        help="Path to the compiled ANE microcode blob (typically a .tsk file).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Optional ANE weights blob to embed alongside the microcode.",
    )
    parser.add_argument(
        "--tile-descriptors",
        type=Path,
        help=(
            "Optional tile descriptor binary emitted by the ANE compiler. "
            "Its size is used to validate --td-size/--td-count or infer the "
            "missing value when only one of them is provided."
        ),
    )
    parser.add_argument(
        "--td-size",
        type=int,
        help="Tile descriptor entry size in bytes.",
    )
    parser.add_argument(
        "--td-count",
        type=int,
        help="Number of tile descriptor entries emitted by the compiler.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "Directory containing outputs from the ANE compiler. The helper scans "
            "the directory recursively and auto-selects microcode/weights/tile "
            "descriptor payloads when the explicit --microcode/--weights/"
            "--tile-descriptors options are omitted."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip parsing the resulting model to double check the metadata.",
    )
    return parser


def _ensure_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ConversionError(f"{description} '{path}' does not exist")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ConversionError(f"{name} must be a positive integer")


def _infer_td_parameters(
    *,
    td_bytes: bytes | None,
    td_size: int | None,
    td_count: int | None,
) -> tuple[int, int]:
    """Derive the tile descriptor parameters from the provided data."""

    if td_bytes is None:
        if td_size is None or td_count is None:
            raise ConversionError(
                "Tile descriptor size and count must be provided when "
                "--tile-descriptors is omitted."
            )
        _validate_positive("--td-size", td_size)
        _validate_positive("--td-count", td_count)
        return td_size, td_count

    td_len = len(td_bytes)
    if td_len == 0:
        raise ConversionError("Tile descriptor blob is empty")

    if td_size is not None:
        _validate_positive("--td-size", td_size)
    if td_count is not None:
        _validate_positive("--td-count", td_count)

    if td_size is not None and td_count is not None:
        if td_size * td_count != td_len:
            raise ConversionError(
                "--td-size x --td-count does not match the tile descriptor payload length"
            )
        return td_size, td_count

    if td_size is not None:
        if td_len % td_size:
            raise ConversionError(
                "Tile descriptor payload length is not a multiple of --td-size; "
                "provide --td-count explicitly"
            )
        return td_size, td_len // td_size

    if td_count is not None:
        if td_len % td_count:
            raise ConversionError(
                "Tile descriptor payload length is not a multiple of --td-count; "
                "provide --td-size explicitly"
            )
        return td_len // td_count, td_count

    raise ConversionError(
        "Unable to infer tile descriptor size/count; supply --td-size or --td-count"
    )


def _decode_base64(value: str, *, field: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ConversionError(f"Failed to decode base64 payload for {field}: {exc}") from exc


def _collect_matches(root: Path, predicate: Callable[[Path], bool]) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and predicate(path):
            matches.append(path)
    matches.sort(key=lambda p: (len(p.parts), str(p)))
    return matches


def _match_keywords(name: str, keyword_sets: Sequence[Sequence[str]]) -> bool:
    lowered = name.lower()
    for keywords in keyword_sets:
        if all(keyword in lowered for keyword in keywords):
            return True
    return False


def _select_artifact(
    *,
    root: Path,
    description: str,
    flag: str,
    suffixes: Sequence[str],
    keyword_sets: Sequence[Sequence[str]] = (),
) -> Path | None:
    suffixes_lc = tuple(suffix.lower() for suffix in suffixes)
    matches = _collect_matches(
        root,
        lambda candidate: (
            bool(suffixes_lc) and candidate.name.lower().endswith(suffixes_lc)
        )
        or _match_keywords(candidate.name, keyword_sets),
    )

    if not matches:
        return None

    if len(matches) > 1:
        display = ", ".join(str(path) for path in matches[:5])
        if len(matches) > 5:
            display += f", ... (+{len(matches) - 5} more)"
        raise ConversionError(
            f"Multiple {description} candidates discovered under '{root}'. "
            f"Disambiguate the selection by specifying {flag} explicitly. Candidates: {display}"
        )

    return matches[0]


@dataclass
class AneBundle:
    """Container describing the metadata emitted by the ANE compiler."""

    microcode: bytes
    td_size: int
    td_count: int
    tile_descriptors: bytes | None = None
    weights: bytes | None = None

    @classmethod
    def from_path(cls, path: Path) -> "AneBundle":
        """Parse a bundle description and return the decoded artefacts."""

        if path.is_dir():
            bundle_path = path / "bundle.json"
            if not bundle_path.is_file():
                raise ConversionError(
                    "Bundle directory does not contain 'bundle.json'. Provide the JSON "
                    "description emitted by the ANE tooling."
                )
            raw = bundle_path.read_text(encoding="utf-8")
        else:
            raw = path.read_text(encoding="utf-8")

        try:
            payload: Mapping[str, object] = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
            raise ConversionError(f"Failed to parse bundle JSON: {exc}") from exc

        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "AneBundle":
        def _get_text(key: str) -> str | None:
            value = payload.get(key)
            return value if isinstance(value, str) else None

        microcode_b64 = _get_text("microcode_b64")
        if not microcode_b64:
            raise ConversionError(
                "Bundle payload is missing the 'microcode_b64' entry produced by the "
                "ANE compiler"
            )
        microcode = _decode_base64(microcode_b64, field="microcode")

        weights_b64 = _get_text("weights_b64")
        weights = _decode_base64(weights_b64, field="weights") if weights_b64 else None

        td_info = payload.get("tile_descriptors")
        td_payload: Mapping[str, object] | None
        if isinstance(td_info, Mapping):
            td_payload = td_info
        else:
            td_payload = None

        if td_payload is not None:
            td_b64 = td_payload.get("b64")
            if isinstance(td_b64, str):
                td_bytes = _decode_base64(td_b64, field="tile_descriptors")
            else:
                td_bytes = None
            raw_td_size = td_payload.get("entry_size")
            raw_td_count = td_payload.get("count")
        else:
            td_bytes = None
            raw_td_size = payload.get("td_size")
            raw_td_count = payload.get("td_count")

        def _maybe_int(value: object) -> int | None:
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value, 10)
                except ValueError as exc:
                    raise ConversionError(
                        f"Invalid integer value in bundle metadata: {value}"
                    ) from exc
            return None

        td_size = _maybe_int(raw_td_size)
        td_count = _maybe_int(raw_td_count)

        if td_bytes is not None:
            td_size, td_count = _infer_td_parameters(
                td_bytes=td_bytes, td_size=td_size, td_count=td_count
            )
        elif td_size is None or td_count is None:
            raise ConversionError(
                "Bundle is missing tile descriptor payload information. Provide "
                "'tile_descriptors' with base64 data or the explicit 'td_size' and "
                "'td_count' fields."
            )

        return cls(
            microcode=microcode,
            td_size=td_size,
            td_count=td_count,
            tile_descriptors=td_bytes,
            weights=weights,
        )


def _default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".onnx"
    return input_path.with_name(f"{input_path.stem}_ane{suffix}")


def convert_model(args: argparse.Namespace) -> Path:
    """Perform the metadata embedding and return the output path."""

    _ensure_file(args.input, "Input model")
    model_bytes = args.input.read_bytes()

    bundle: AneBundle | None = None
    microcode_path: Path | None = args.microcode
    weights_path: Path | None = args.weights
    tile_descriptor_path: Path | None = args.tile_descriptors

    if args.bundle:
        bundle = AneBundle.from_path(args.bundle)

        conflicts = {
            "--microcode": args.microcode,
            "--weights": args.weights,
            "--tile-descriptors": args.tile_descriptors,
            "--td-size": args.td_size,
            "--td-count": args.td_count,
            "--artifact-root": args.artifact_root,
        }
        conflicting = [flag for flag, value in conflicts.items() if value is not None]
        if conflicting:
            flags = ", ".join(conflicting)
            raise ConversionError(
                "--bundle already supplies the ANE artefacts; the following options "
                f"cannot be combined with it: {flags}"
            )

        microcode_bytes = bundle.microcode
        weights_bytes = bundle.weights
        td_bytes = bundle.tile_descriptors
        td_size = bundle.td_size
        td_count = bundle.td_count
    else:
        auto_report: list[str] = []

        if args.artifact_root:
            if not args.artifact_root.is_dir():
                raise ConversionError(
                    f"Artifact root '{args.artifact_root}' is not a directory"
                )

            auto_microcode = _select_artifact(
                root=args.artifact_root,
                description="microcode",
                flag="--microcode",
                suffixes=(".tsk", ".tsk.bin", ".microcode", ".microcode.bin"),
                keyword_sets=(("microcode",), ("tsk",)),
            )
            if auto_microcode and not microcode_path:
                microcode_path = auto_microcode
                auto_report.append(f"microcode -> {auto_microcode}")

            auto_weights = _select_artifact(
                root=args.artifact_root,
                description="weights",
                flag="--weights",
                suffixes=(".weights", ".weights.bin", ".aneweights", ".aneweights.bin"),
                keyword_sets=(("weight",),),
            )
            if auto_weights and not weights_path:
                weights_path = auto_weights
                auto_report.append(f"weights -> {auto_weights}")

            auto_tile = _select_artifact(
                root=args.artifact_root,
                description="tile descriptor",
                flag="--tile-descriptors",
                suffixes=(
                    ".td.bin",
                    ".tile_desc.bin",
                    ".tile_descriptors.bin",
                    ".tiledesc.bin",
                    ".tds",
                    ".ane.td",
                ),
                keyword_sets=(("tile", "desc"),),
            )
            if auto_tile and not tile_descriptor_path:
                tile_descriptor_path = auto_tile
                auto_report.append(f"tile descriptors -> {auto_tile}")

            if auto_report:
                print(
                    "Auto-detected ANE artefacts:\n" + "\n".join(f"  {line}" for line in auto_report)
                )

        if not microcode_path:
            raise ConversionError(
                "Specify --microcode or provide a --bundle exported by the ANE toolchain"
            )

        _ensure_file(microcode_path, "Microcode file")

        if weights_path:
            _ensure_file(weights_path, "Weights file")
        if tile_descriptor_path:
            _ensure_file(tile_descriptor_path, "Tile descriptor file")

        microcode_bytes = microcode_path.read_bytes()
        weights_bytes = weights_path.read_bytes() if weights_path else None
        td_bytes = (
            tile_descriptor_path.read_bytes() if tile_descriptor_path else None
        )

        td_size, td_count = _infer_td_parameters(
            td_bytes=td_bytes, td_size=args.td_size, td_count=args.td_count
        )

    if args.bundle and bundle is not None:
        td_size = bundle.td_size
        td_count = bundle.td_count

    try:
        parse_ane_metadata(model_bytes)
    except MissingAneMetadataError:
        metadata_present = False
    else:
        metadata_present = True

    updated_bytes = with_ane_metadata(
        model_bytes,
        microcode=microcode_bytes,
        td_size=td_size,
        td_count=td_count,
        weights=weights_bytes,
    )

    if not args.skip_validation:
        parse_ane_metadata(updated_bytes)

    output_path = args.output or _default_output_path(args.input)
    output_path.write_bytes(updated_bytes)

    if metadata_present:
        print(
            "Input model already contained ANE metadata; entries were replaced "
            "with the supplied microcode and parameters."
        )

    microcode_source: str
    weights_source: str
    td_source: str

    if args.bundle:
        microcode_source = f"from bundle {args.bundle}"
        weights_source = (
            f"from bundle {args.bundle}" if weights_bytes else "not provided"
        )
        td_source = (
            f"from bundle {args.bundle}"
            if td_bytes is not None
            else "not provided"
        )
    else:
        microcode_source = str(microcode_path)
        weights_source = str(weights_path) if weights_path else "not provided"
        td_source = (
            str(tile_descriptor_path) if tile_descriptor_path else "not provided"
        )

    print(
        "ANE conversion complete:\n"
        f"  output model : {output_path}\n"
        f"  microcode    : {microcode_source}\n"
        f"  td size      : {td_size}\n"
        f"  td count     : {td_count}\n"
        f"  weights      : {weights_source}\n"
        f"  tile desc    : {td_source}"
    )

    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        convert_model(args)
    except ConversionError as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry point
    raise SystemExit(main())
