"""High level helper to submit ONNX models to the ANE driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .device import (
    ANE_CMD_GRAN,
    ANE_SUBMIT_FLAG_ONNX,
    CMD_BUF_BDX,
    KRN_BUF_BDX,
    ANEDevice,
    AneBuffer,
    DrmAneSubmit,
    IOCTL_ANE_SUBMIT,
    drm_ioctl,
)
from .metadata import AneModelMetadata, microcode_aligned_size

__all__ = ["AneOnnxSubmission", "submit_onnx_model"]


@dataclass
class AneOnnxSubmission:
    """Summary information returned by the kernel submission."""

    handles: Tuple[int, ...]
    btsp_handle: int
    tsk_size: int
    td_count: int
    td_size: int


def submit_onnx_model(
    device: ANEDevice,
    model_bytes: bytes,
    metadata: AneModelMetadata,
) -> AneOnnxSubmission:
    """Upload the ONNX payload and issue the `DRM_IOCTL_ANE_SUBMIT` ioctl."""
    if device.fd is None:
        raise RuntimeError("Device must be opened before submission")

    onnx_size = len(model_bytes)
    cmd_payload = microcode_aligned_size(metadata.microcode_len, ANE_CMD_GRAN) + metadata.weights_len
    cmd_size = max(cmd_payload, onnx_size)

    btsp_size = metadata.btsp_size
    if btsp_size <= 0:
        raise RuntimeError("Tile descriptor size metadata resulted in zero BTSP size")

    with device.allocate_buffer(cmd_size) as cmd_bo, device.allocate_buffer(btsp_size) as btsp_bo:
        _populate_command_buffer(cmd_bo, model_bytes, cmd_size)
        btsp_bo.zero()

        submit = DrmAneSubmit()
        for idx in range(len(submit.handles)):
            submit.handles[idx] = 0
        submit.handles[CMD_BUF_BDX] = cmd_bo.handle
        submit.handles[KRN_BUF_BDX] = 0
        submit.btsp_handle = btsp_bo.handle
        submit.tsk_size = 0
        submit.td_count = 0
        submit.td_size = 0
        submit.pad = ANE_SUBMIT_FLAG_ONNX

        drm_ioctl(device.fd, IOCTL_ANE_SUBMIT, submit)

    return AneOnnxSubmission(
        handles=tuple(submit.handles),
        btsp_handle=submit.btsp_handle,
        tsk_size=submit.tsk_size,
        td_count=submit.td_count,
        td_size=submit.td_size,
    )


def _populate_command_buffer(buffer: AneBuffer, model_bytes: bytes, cmd_size: int) -> None:
    """Write the ONNX payload to the command buffer, padding with zeroes."""
    with buffer.mmap() as cmd_map:
        cmd_map.seek(0)
        cmd_map.write(model_bytes)
        remainder = cmd_size - len(model_bytes)
        if remainder > 0:
            cmd_map.write(b"\x00" * remainder)
        cmd_map.flush()
