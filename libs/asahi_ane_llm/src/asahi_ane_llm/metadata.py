"""Helpers for reading and writing ANE-specific ONNX metadata."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import onnx


@dataclass(frozen=True)
class AneModelMetadata:
    """Metadata entries extracted from an ONNX model."""

    microcode_len: int
    weights_len: int
    td_size: int
    td_count: int

    @property
    def command_buffer_size(self) -> int:
        """Return the minimum command buffer size required for submission."""
        return microcode_aligned_size(self.microcode_len) + self.weights_len

    @property
    def btsp_size(self) -> int:
        """Return the BTSP size derived from the tile descriptors."""
        return self.td_size * self.td_count


@dataclass(frozen=True)
class AneMetadataPayloads:
    """Container for decoded ANE metadata payloads embedded in an ONNX model."""

    microcode: bytes
    tile_descriptors: bytes | None
    weights: bytes | None
    td_size: int
    td_count: int


def microcode_aligned_size(microcode_len: int, alignment: int = 16) -> int:
    """Round the microcode length to the next alignment boundary."""
    return (microcode_len + alignment - 1) // alignment * alignment


class MissingAneMetadataError(RuntimeError):
    """Raised when an ONNX model omits the required ANE metadata entries."""


def _load_metadata_map(model_bytes: bytes) -> dict[str, str]:
    model = onnx.load_from_string(model_bytes)
    return {prop.key: prop.value for prop in model.metadata_props}


def _require_metadata_entries(metadata: dict[str, str]) -> dict[str, str]:
    required_keys = ("ane.microcode.b64", "ane.td_size", "ane.td_count")
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:  # pragma: no cover - metadata errors
        missing_list = ", ".join(sorted(missing_keys))
        raise MissingAneMetadataError(
            "The supplied ONNX model is missing the Asahi-specific metadata "
            f"entries: {missing_list}. Use the Asahi conversion tooling to "
            "embed the ANE microcode (see 'python -m asahi_ane_llm.tools."
            "embed_metadata')."
        )
    return metadata


def parse_ane_metadata(model_bytes: bytes) -> AneModelMetadata:
    """Parse the metadata stored alongside the ONNX payload."""
    metadata = _require_metadata_entries(_load_metadata_map(model_bytes))

    try:
        microcode_b64 = metadata["ane.microcode.b64"].encode("ascii")
        td_size = int(metadata["ane.td_size"], 10)
        td_count = int(metadata["ane.td_count"], 10)
    except ValueError as exc:  # pragma: no cover - metadata errors
        raise RuntimeError(f"Invalid metadata entry: {exc}") from exc

    weights_b64 = metadata.get("ane.weights.b64", "").encode("ascii")

    microcode_len = len(base64.b64decode(microcode_b64, altchars=b"-_"))
    weights_len = (
        len(base64.b64decode(weights_b64, altchars=b"-_")) if weights_b64 else 0
    )

    return AneModelMetadata(
        microcode_len=microcode_len,
        weights_len=weights_len,
        td_size=td_size,
        td_count=td_count,
    )


def extract_ane_payloads(model_bytes: bytes) -> AneMetadataPayloads:
    """Return the raw ANE metadata payloads stored inside an ONNX model."""

    metadata = _require_metadata_entries(_load_metadata_map(model_bytes))

    try:
        td_size = int(metadata["ane.td_size"], 10)
        td_count = int(metadata["ane.td_count"], 10)
    except ValueError as exc:  # pragma: no cover - metadata errors
        raise RuntimeError(f"Invalid metadata entry: {exc}") from exc

    def _decode_b64(key: str) -> bytes | None:
        value = metadata.get(key)
        if not value:
            return None
        try:
            return base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
        except Exception as exc:  # pragma: no cover - defensive path
            raise RuntimeError(f"Failed to decode metadata entry {key}: {exc}") from exc

    microcode = _decode_b64("ane.microcode.b64")
    if microcode is None:  # pragma: no cover - sanity check
        raise MissingAneMetadataError("ane.microcode.b64 entry missing from ONNX model")

    tile_descriptors = _decode_b64("ane.tile_descriptors.b64")
    weights = _decode_b64("ane.weights.b64")

    return AneMetadataPayloads(
        microcode=microcode,
        tile_descriptors=tile_descriptors,
        weights=weights,
        td_size=td_size,
        td_count=td_count,
    )


def with_ane_metadata(
    model_bytes: bytes,
    *,
    microcode: bytes,
    td_size: int,
    td_count: int,
    tile_descriptors: bytes | None = None,
    weights: bytes | None = None,
) -> bytes:
    """Return a new ONNX payload that includes the ANE metadata entries.

    Parameters
    ----------
    model_bytes:
        Raw ONNX model bytes to augment.
    microcode:
        Compiled ANE microcode blob that should be embedded as base64.
    td_size:
        Size in bytes of a single tile descriptor entry.
    td_count:
        Number of tile descriptor entries produced by the compiler.
    tile_descriptors:
        Optional tile descriptor payload to store alongside the metadata.
    weights:
        Optional ANE weights blob to embed alongside the microcode.
    """

    if td_size <= 0:
        raise ValueError("Tile descriptor size must be a positive integer")
    if td_count <= 0:
        raise ValueError("Tile descriptor count must be a positive integer")

    model = onnx.load_from_string(model_bytes)

    def _set_metadata(key: str, value: str) -> None:
        for prop in model.metadata_props:
            if prop.key == key:
                prop.value = value
                break
        else:
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value

    def _remove_metadata(key: str) -> None:
        model.metadata_props[:] = [prop for prop in model.metadata_props if prop.key != key]

    microcode_b64 = base64.b64encode(microcode, altchars=b"-_").decode("ascii")
    _set_metadata("ane.microcode.b64", microcode_b64)

    if tile_descriptors:
        tile_desc_b64 = base64.b64encode(tile_descriptors, altchars=b"-_").decode("ascii")
        _set_metadata("ane.tile_descriptors.b64", tile_desc_b64)
    else:
        _remove_metadata("ane.tile_descriptors.b64")

    if weights:
        weights_b64 = base64.b64encode(weights, altchars=b"-_").decode("ascii")
        _set_metadata("ane.weights.b64", weights_b64)
    else:
        _remove_metadata("ane.weights.b64")

    _set_metadata("ane.td_size", str(td_size))
    _set_metadata("ane.td_count", str(td_count))

    buffer = io.BytesIO()
    onnx.save_model(model, buffer)
    return buffer.getvalue()


__all__ = [
    "AneMetadataPayloads",
    "AneModelMetadata",
    "MissingAneMetadataError",
    "parse_ane_metadata",
    "extract_ane_payloads",
    "microcode_aligned_size",
    "with_ane_metadata",
]
