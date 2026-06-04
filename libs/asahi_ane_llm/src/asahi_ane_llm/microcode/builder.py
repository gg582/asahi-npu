"""Utilities for compiling ANE microprogram descriptions into artefacts."""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .schema import MicrocodeSchema, SchemaError


class MicrocodeBuildError(RuntimeError):
    """Raised when any of the build inputs are invalid."""


@dataclass(frozen=True)
class BuildArtifacts:
    """Container for the assembled ANE artefacts."""

    microcode: bytes
    tile_descriptors: bytes | None
    td_size: int | None
    td_count: int | None
    weights: bytes | None
    outputs: Mapping[str, Path]


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MicrocodeBuildError(f"File '{path}' does not exist") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise MicrocodeBuildError(f"Failed to parse JSON from '{path}': {exc}") from exc


def _decode_base64_field(payload: Mapping[str, Any], *, field: str) -> bytes:
    raw_value = payload.get(field)
    if raw_value is None:
        raise MicrocodeBuildError(f"'{field}' entry is required in the JSON description")
    if not isinstance(raw_value, str):
        raise MicrocodeBuildError(f"'{field}' must be a base64 encoded string")

    try:
        return base64.b64decode(raw_value, altchars=b"-_", validate=True)
    except Exception as exc:  # pragma: no cover - defensive path
        raise MicrocodeBuildError(
            f"Failed to decode base64 payload from '{field}': {exc}"
        ) from exc


def _compile_microcode(schema_path: Path, program_path: Path) -> bytes:
    schema_payload = _load_json(schema_path)
    if not isinstance(schema_payload, Mapping):
        raise MicrocodeBuildError("Schema JSON must describe an object")

    program_payload = _load_json(program_path)
    if not isinstance(program_payload, list):
        raise MicrocodeBuildError("Program JSON must be an array of instruction objects")

    try:
        schema = MicrocodeSchema.from_mapping(schema_payload)
    except SchemaError as exc:
        raise MicrocodeBuildError(str(exc)) from exc

    try:
        return schema.assemble_program(program_payload)
    except SchemaError as exc:
        raise MicrocodeBuildError(str(exc)) from exc


def _compile_tile_descriptors(payload: Mapping[str, Any]) -> tuple[bytes, int, int]:
    from .. import hw

    if "data_b64" in payload:
        td_bytes = _decode_base64_field(payload, field="data_b64")
        td_size = payload.get("entry_size")
        td_count = payload.get("entry_count")
    else:
        entries_raw = payload.get("entries")
        if not isinstance(entries_raw, list):
            # Try if the payload itself is a list (for backward compatibility or direct list usage)
            if isinstance(payload, list):
                entries_raw = payload
            else:
                raise MicrocodeBuildError(
                    "Tile descriptor spec must provide either 'data_b64' or an 'entries' list"
                )

        td_size = payload.get("entry_size", 0x274) if isinstance(payload, Mapping) else 0x274
        try:
            entry_size = int(td_size)
        except (TypeError, ValueError) as exc:
            raise MicrocodeBuildError("'entry_size' must be an integer") from exc

        td_bytes = bytearray()
        entry_count = 0
        for index, entry in enumerate(entries_raw):
            repeat = 1
            if isinstance(entry, Mapping):
                repeat = int(entry.get("repeat", 1))
                
                if "bytes_b64" in entry or "data_b64" in entry:
                    blob = (
                        _decode_base64_field(entry, field="bytes_b64")
                        if "bytes_b64" in entry
                        else _decode_base64_field(entry, field="data_b64")
                    )
                elif "registers" in entry or "binary_patches" in entry:
                    # New structured format
                    tmp = bytearray(entry_size)
                    
                    # 1. Apply binary patches first (e.g. DMA context)
                    patches = entry.get("binary_patches", {})
                    for offset_str, hex_data in patches.items():
                        try:
                            offset = int(offset_str, 0)
                            data = bytes.fromhex(hex_data.replace(' ', '').replace('\n', ''))
                            if offset + len(data) > entry_size:
                                raise MicrocodeBuildError(f"Binary patch at {offset} exceeds entry_size")
                            tmp[offset:offset+len(data)] = data
                        except Exception as exc:
                            raise MicrocodeBuildError(f"Invalid binary patch at '{offset_str}': {exc}") from exc

                    # 2. Apply register assignments (can overwrite patches)
                    regs = entry.get("registers", {})
                    for reg_path, val in regs.items():
                        # Support paths like "TD.W0" or "Common.InDim"
                        try:
                            parts = reg_path.split('.')
                            if len(parts) == 2:
                                cls_name, attr_name = parts
                                target_cls = getattr(hw, cls_name)
                                offset = getattr(target_cls, attr_name)
                            else:
                                offset = int(reg_path, 0)
                            
                            struct.pack_into('<I', tmp, offset, int(val, 0) if isinstance(val, str) else int(val))
                        except Exception as exc:
                            raise MicrocodeBuildError(f"Invalid register assignment '{reg_path}': {exc}") from exc
                    blob = bytes(tmp)
                elif "value" in entry:
                    # ... (existing value logic)
                    try:
                        value = int(entry["value"])
                    except (TypeError, ValueError) as exc:
                        raise MicrocodeBuildError(
                            f"Tile descriptor entry {index} value must be an integer"
                        ) from exc
                    signed = bool(entry.get("signed", False))
                    if signed:
                        limit = 1 << (entry_size * 8 - 1)
                        if not (-limit <= value < limit):
                            raise MicrocodeBuildError(
                                f"Tile descriptor entry {index} signed value does not fit"
                            )
                        value &= (1 << (entry_size * 8)) - 1
                    else:
                        if not (0 <= value < (1 << (entry_size * 8))):
                            raise MicrocodeBuildError(
                                f"Tile descriptor entry {index} unsigned value does not fit"
                            )
                    entry_endianness = str(
                        entry.get("endianness", payload.get("endianness", "little") if isinstance(payload, Mapping) else "little")
                    )
                    blob = value.to_bytes(entry_size, entry_endianness)
                else:
                    # Support the "op": "dispatch" style if needed, 
                    # but for now let's focus on structured registers.
                    # If it's an old style with "payload" list:
                    if "payload" in entry:
                        payload_data = entry["payload"]
                        blob = bytes(payload_data)
                        if len(blob) < entry_size:
                            blob = blob + b'\x00' * (entry_size - len(blob))
                    else:
                        raise MicrocodeBuildError(
                            f"Tile descriptor entry {index} must provide 'bytes_b64', 'registers', 'value' or 'payload'"
                        )
            else:
                # Support raw bytes if entry is a base64 string
                try:
                    blob = base64.b64decode(entry, altchars=b"-_", validate=True)
                except Exception:
                    raise MicrocodeBuildError(
                        f"Tile descriptor entry {index} must be an object or base64 string"
                    )

            if len(blob) > entry_size:
                 raise MicrocodeBuildError(
                    f"Tile descriptor entry {index} size {len(blob)} exceeds entry_size {entry_size}"
                )
            if len(blob) < entry_size:
                blob = blob + b'\x00' * (entry_size - len(blob))

            td_bytes.extend(blob * repeat)
            entry_count += repeat

        td_count = entry_count

    if td_size is None:
        raise MicrocodeBuildError(
            "Tile descriptor spec must provide 'entry_size' when using 'data_b64'"
        )
    if td_count is None:
        if len(td_bytes) % int(td_size):
            raise MicrocodeBuildError(
                "Unable to infer tile descriptor count from payload length"
            )
        td_count = len(td_bytes) // int(td_size)

    try:
        td_size_int = int(td_size)
        td_count_int = int(td_count)
    except (TypeError, ValueError) as exc:
        raise MicrocodeBuildError("Tile descriptor size/count must be integers") from exc

    if td_size_int <= 0 or td_count_int <= 0:
        raise MicrocodeBuildError("Tile descriptor size/count must be positive")

    return bytes(td_bytes), td_size_int, td_count_int


def _compile_weights(payload: Mapping[str, Any]) -> bytes:
    if "data_b64" in payload:
        return _decode_base64_field(payload, field="data_b64")

    values = payload.get("values")
    if not isinstance(values, list):
        raise MicrocodeBuildError(
            "Weights spec must provide 'data_b64' or a numeric 'values' array"
        )

    word_size = payload.get("word_size", 32)
    try:
        word_size_bits = int(word_size)
    except (TypeError, ValueError) as exc:
        raise MicrocodeBuildError("Weights word_size must be an integer") from exc
    if word_size_bits % 8:
        raise MicrocodeBuildError("Weights word_size must be a multiple of 8")
    if word_size_bits <= 0:
        raise MicrocodeBuildError("Weights word_size must be positive")

    byte_width = word_size_bits // 8
    endianness = payload.get("endianness", "little")
    if endianness not in {"little", "big"}:
        raise MicrocodeBuildError("Weights endianness must be 'little' or 'big'")

    output = bytearray()
    for index, value_raw in enumerate(values):
        try:
            value = int(value_raw)
        except (TypeError, ValueError) as exc:
            raise MicrocodeBuildError(
                f"Weights entry {index} must be an integer"
            ) from exc
        if not (0 <= value < (1 << word_size_bits)):
            raise MicrocodeBuildError(
                f"Weights entry {index} does not fit in {word_size_bits} bits"
            )
        output.extend(value.to_bytes(byte_width, endianness))

    return bytes(output)


def compile_from_spec(
    *,
    schema_path: Path,
    program_path: Path,
    tile_spec_path: Path | None = None,
    weights_spec_path: Path | None = None,
    output_dir: Path | None = None,
) -> BuildArtifacts:
    """Compile ANE artefacts from reverse engineered specifications."""

    microcode = _compile_microcode(schema_path, program_path)

    tile_descriptors: bytes | None = None
    td_size: int | None = None
    td_count: int | None = None
    if tile_spec_path:
        tile_payload_raw = _load_json(tile_spec_path)
        if not isinstance(tile_payload_raw, Mapping):
            raise MicrocodeBuildError("Tile descriptor JSON must describe an object")
        tile_descriptors, td_size, td_count = _compile_tile_descriptors(tile_payload_raw)

    weights: bytes | None = None
    if weights_spec_path:
        weights_payload = _load_json(weights_spec_path)
        if not isinstance(weights_payload, Mapping):
            raise MicrocodeBuildError("Weights JSON must describe an object")
        weights = _compile_weights(weights_payload)

    outputs: dict[str, Path] = {}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        microcode_path = output_dir / "microcode.tsk"
        microcode_path.write_bytes(microcode)
        outputs["microcode"] = microcode_path

        if tile_descriptors is not None:
            tile_path = output_dir / "tile_descriptors.bin"
            tile_path.write_bytes(tile_descriptors)
            outputs["tile_descriptors"] = tile_path

        if weights is not None:
            weights_path = output_dir / "weights.bin"
            weights_path.write_bytes(weights)
            outputs["weights"] = weights_path

    return BuildArtifacts(
        microcode=microcode,
        tile_descriptors=tile_descriptors,
        td_size=td_size,
        td_count=td_count,
        weights=weights,
        outputs=outputs,
    )


__all__ = [
    "BuildArtifacts",
    "MicrocodeBuildError",
    "compile_from_spec",
]
