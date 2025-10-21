#!/usr/bin/env python3
"""
Example script demonstrating how to submit an ONNX model to the Apple Neural
Engine DRM driver using the in-kernel ONNX ingestion path.

The driver expects the ONNX model to be uploaded into a GEM command buffer
before ``DRM_IOCTL_ANE_SUBMIT`` is issued with ``ANE_SUBMIT_FLAG_ONNX`` set. The
kernel parses the metadata, repacks the payload into ANE microcode + weights,
and populates the tile descriptor sizing fields automatically.

Requirements
------------
* Python 3.8+
* ``onnx`` python package (``pip install onnx``)
* Access to the ANE DRM render node (typically ``/dev/dri/renderD129``)

The DRM ioctl layout mirrors ``include/uapi/drm/ane_accel.h`` from the kernel.
Adjust ``ANE_TILE_COUNT`` if your kernel exposes a different value.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import mmap
import os
import sys
from pathlib import Path
from typing import Dict

try:
    import onnx
except ImportError as exc:  # pragma: no cover - optional dependency error path
    print(
        "The onnx package is required for this example. Install it with 'pip install onnx'",
        file=sys.stderr,
    )
    raise


ANE_TILE_COUNT = 8
ANE_CMD_GRAN = 16
ANE_SUBMIT_FLAG_ONNX = 0x1

CMD_BUF_BDX = 0
KRN_BUF_BDX = 1

DRM_IOCTL_BASE = ord("d")
DRM_COMMAND_BASE = 0x40


class DrmAneBoInit(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]


class DrmAneBoFree(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class DrmAneSubmit(ctypes.Structure):
    _fields_ = [
        ("handles", ctypes.c_uint32 * ANE_TILE_COUNT),
        ("btsp_handle", ctypes.c_uint32),
        ("tsk_size", ctypes.c_uint32),
        ("td_count", ctypes.c_uint32),
        ("td_size", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(direction: int, type_char: int, nr: int, size: int) -> int:
    return (
        (direction << _IOC_DIRSHIFT)
        | (type_char << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def DRM_IOWR(nr: int, struct_type: ctypes.Structure) -> int:
    return _IOC(_IOC_READ | _IOC_WRITE, DRM_IOCTL_BASE, nr, ctypes.sizeof(struct_type))


IOCTL_ANE_BO_INIT = DRM_IOWR(DRM_COMMAND_BASE + 0x0, DrmAneBoInit)
IOCTL_ANE_BO_FREE = DRM_IOWR(DRM_COMMAND_BASE + 0x1, DrmAneBoFree)
IOCTL_ANE_SUBMIT = DRM_IOWR(DRM_COMMAND_BASE + 0x2, DrmAneSubmit)


libc = ctypes.CDLL(None, use_errno=True)
libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
libc.ioctl.restype = ctypes.c_int


def drm_ioctl(fd: int, request: int, obj: ctypes.Structure) -> None:
    ret = libc.ioctl(fd, request, ctypes.byref(obj))
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


class BoHandle:
    def __init__(self, fd: int, size: int):
        self.fd = fd
        self.init = DrmAneBoInit()
        self.init.size = size
        drm_ioctl(fd, IOCTL_ANE_BO_INIT, self.init)

    def mmap(self) -> mmap.mmap:
        return mmap.mmap(
            self.fd,
            int(self.init.size),
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_SHARED,
            offset=int(self.init.offset),
        )

    def free(self) -> None:
        req = DrmAneBoFree()
        req.handle = self.init.handle
        drm_ioctl(self.fd, IOCTL_ANE_BO_FREE, req)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_metadata(model_bytes: bytes) -> Dict[str, int]:
    model = onnx.load_from_string(model_bytes)
    metadata: Dict[str, str] = {prop.key: prop.value for prop in model.metadata_props}

    try:
        microcode_b64 = metadata["ane.microcode.b64"].encode("ascii")
        td_size = int(metadata["ane.td_size"], 10)
        td_count = int(metadata["ane.td_count"], 10)
    except KeyError as exc:  # pragma: no cover - metadata errors
        raise RuntimeError(f"Missing required metadata entry: {exc}") from exc

    weights_b64 = metadata.get("ane.weights.b64", "").encode("ascii")

    microcode_len = len(base64.b64decode(microcode_b64, altchars=b"-_"))
    weights_len = len(base64.b64decode(weights_b64, altchars=b"-_")) if weights_b64 else 0

    return {
        "microcode_len": microcode_len,
        "weights_len": weights_len,
        "td_size": td_size,
        "td_count": td_count,
    }


def zero_map(mm: mmap.mmap) -> None:
    mm.seek(0)
    length = mm.size()
    chunk = 1 << 20  # zero in 1 MiB chunks to avoid allocating huge buffers
    offset = 0
    zero_block = b"\x00" * chunk
    while offset < length:
        step = min(chunk, length - offset)
        mm.write(zero_block[:step])
        offset += step
    mm.flush()


def submit_model(fd: int, model_bytes: bytes, metadata: Dict[str, int]) -> DrmAneSubmit:
    onnx_size = len(model_bytes)
    cmd_payload = align_up(metadata["microcode_len"], ANE_CMD_GRAN) + metadata["weights_len"]
    cmd_size = max(cmd_payload, onnx_size)

    btsp_size = metadata["td_size"] * metadata["td_count"]
    if btsp_size == 0:
        raise RuntimeError("Tile descriptor size metadata resulted in zero BTSP size")

    cmd_bo = BoHandle(fd, cmd_size)
    btsp_bo = BoHandle(fd, btsp_size)

    try:
        cmd_map = cmd_bo.mmap()
        try:
            cmd_map.seek(0)
            cmd_map.write(model_bytes)
            if cmd_size > onnx_size:
                cmd_map.write(b"\x00" * (cmd_size - onnx_size))
            cmd_map.flush()
        finally:
            cmd_map.close()

        btsp_map = btsp_bo.mmap()
        try:
            zero_map(btsp_map)
        finally:
            btsp_map.close()

        submit = DrmAneSubmit()
        for idx in range(ANE_TILE_COUNT):
            submit.handles[idx] = 0
        submit.handles[CMD_BUF_BDX] = cmd_bo.init.handle
        submit.handles[KRN_BUF_BDX] = 0
        submit.btsp_handle = btsp_bo.init.handle
        submit.tsk_size = 0
        submit.td_count = 0
        submit.td_size = 0
        submit.pad = ANE_SUBMIT_FLAG_ONNX

        drm_ioctl(fd, IOCTL_ANE_SUBMIT, submit)
        return submit
    finally:
        btsp_bo.free()
        cmd_bo.free()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to the ONNX model to submit")
    parser.add_argument(
        "--device",
        default="/dev/dri/renderD129",
        help="Path to the ANE DRM render node (default: %(default)s)",
    )
    args = parser.parse_args()

    model_path = args.model
    if not model_path.is_file():
        parser.error(f"Model file '{model_path}' does not exist")

    model_bytes = model_path.read_bytes()
    metadata = parse_metadata(model_bytes)

    fd = os.open(args.device, os.O_RDWR)
    try:
        submit = submit_model(fd, model_bytes, metadata)
    finally:
        os.close(fd)

    print(
        "Submission complete:\n"
        f"  microcode bytes : {submit.tsk_size}\n"
        f"  td entries      : {submit.td_count} x {submit.td_size} bytes",
    )


if __name__ == "__main__":
    main()
