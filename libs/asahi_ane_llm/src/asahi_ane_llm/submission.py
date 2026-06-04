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
    handles: dict[int, int] | None = None,
) -> AneOnnxSubmission:
    """Upload the ANE payloads and issue the `DRM_IOCTL_ANE_SUBMIT` ioctl."""
    if device.fd is None:
        raise RuntimeError("Device must be opened before submission")

    from .metadata import extract_ane_payloads
    payloads = extract_ane_payloads(model_bytes)

    microcode_size = microcode_aligned_size(metadata.microcode_len, ANE_CMD_GRAN)
    cmd_size = microcode_size + metadata.weights_len
    btsp_size = metadata.btsp_size

    if btsp_size <= 0:
        raise RuntimeError("Tile descriptor size metadata resulted in zero BTSP size")
    if payloads.tile_descriptors is None:
        raise RuntimeError("ONNX model is missing tile descriptor payloads")

    with device.allocate_buffer(cmd_size) as cmd_bo, device.allocate_buffer(btsp_size) as btsp_bo:
        # 1. Fill Command Buffer (Microcode + Weights)
        with cmd_bo.mmap() as cmd_map:
            cmd_map.seek(0)
            cmd_map.write(payloads.microcode)
            if payloads.weights:
                cmd_map.seek(microcode_size)
                cmd_map.write(payloads.weights)
            cmd_map.flush()

        # 2. Fill BTSP Buffer (Tile Descriptors)
        with btsp_bo.mmap() as btsp_map:
            btsp_map.seek(0)
            btsp_map.write(payloads.tile_descriptors)
            btsp_map.flush()

        # 3. Submit to Hardware
        submit = DrmAneSubmit()
        for idx in range(len(submit.handles)):
            submit.handles[idx] = handles.get(idx, 0) if handles else 0
        
        # Mapping handles:
        # handles[0] = BTSP (Task Descriptor)
        # handles[1] = Kernel/Weights (implicitly handled by driver if we set tsk_size?)
        # Wait, the driver calculates req.bar[KRN_BUF_BDX] = req.bar[CMD_BUF_BDX] + round_up(args.tsk_size, ANE_CMD_GRAN)
        # So we put CMD_BUF in handles[CMD_BUF_BDX] (0) and set tsk_size to the microcode size.
        
        submit.handles[CMD_BUF_BDX] = cmd_bo.handle
        submit.btsp_handle = btsp_bo.handle
        submit.tsk_size = metadata.microcode_len
        submit.td_count = metadata.td_count
        submit.td_size = metadata.td_size
        submit.pad = 0 # Direct submission, no ONNX flag

        drm_ioctl(device.fd, IOCTL_ANE_SUBMIT, submit)

    return AneOnnxSubmission(
        handles=tuple(submit.handles),
        btsp_handle=submit.btsp_handle,
        tsk_size=int(submit.tsk_size),
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
