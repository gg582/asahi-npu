"""Parsing utilities for ANE specific ONNX metadata."""

from __future__ import annotations

import base64
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


def microcode_aligned_size(microcode_len: int, alignment: int = 16) -> int:
    """Round the microcode length to the next alignment boundary."""
    return (microcode_len + alignment - 1) // alignment * alignment


def parse_ane_metadata(model_bytes: bytes) -> AneModelMetadata:
    """Parse the metadata stored alongside the ONNX payload."""
    model = onnx.load_from_string(model_bytes)
    metadata = {prop.key: prop.value for prop in model.metadata_props}

    required_keys = ("ane.microcode.b64", "ane.td_size", "ane.td_count")
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:  # pragma: no cover - metadata errors
        missing_list = ", ".join(sorted(missing_keys))
        raise RuntimeError(
            "The supplied ONNX model is missing the Asahi-specific metadata "
            f"entries: {missing_list}.\n"
            "Ensure you converted the model with the ANE tooling so that the "
            "microcode and tile descriptor metadata are embedded."
        )

    try:
        microcode_b64 = metadata["ane.microcode.b64"].encode("ascii")
        td_size = int(metadata["ane.td_size"], 10)
        td_count = int(metadata["ane.td_count"], 10)
    except ValueError as exc:  # pragma: no cover - metadata errors
        raise RuntimeError(f"Invalid metadata entry: {exc}") from exc

    weights_b64 = metadata.get("ane.weights.b64", "").encode("ascii")

    microcode_len = len(base64.b64decode(microcode_b64, altchars=b"-_"))
    weights_len = len(base64.b64decode(weights_b64, altchars=b"-_")) if weights_b64 else 0

    return AneModelMetadata(
        microcode_len=microcode_len,
        weights_len=weights_len,
        td_size=td_size,
        td_count=td_count,
    )


__all__ = ["AneModelMetadata", "parse_ane_metadata", "microcode_aligned_size"]
