"""Low level helpers for dealing with the ANE DRM device."""

from __future__ import annotations

import ctypes
import mmap
import os
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "ANE_TILE_COUNT",
    "ANE_CMD_GRAN",
    "ANE_SUBMIT_FLAG_ONNX",
    "CMD_BUF_BDX",
    "KRN_BUF_BDX",
    "ANEDevice",
    "AneBuffer",
    "AneIoctlError",
    "drm_ioctl",
    "zero_map",
]


ANE_TILE_COUNT = 8
ANE_CMD_GRAN = 16
ANE_SUBMIT_FLAG_ONNX = 0x1

CMD_BUF_BDX = 0
KRN_BUF_BDX = 1

DRM_IOCTL_BASE = ord("d")
DRM_COMMAND_BASE = 0x40

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


def DRM_IOWR(nr: int, struct_type: type[ctypes.Structure]) -> int:
    return _IOC(_IOC_READ | _IOC_WRITE, DRM_IOCTL_BASE, nr, ctypes.sizeof(struct_type))


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


IOCTL_ANE_BO_INIT = DRM_IOWR(DRM_COMMAND_BASE + 0x0, DrmAneBoInit)
IOCTL_ANE_BO_FREE = DRM_IOWR(DRM_COMMAND_BASE + 0x1, DrmAneBoFree)
IOCTL_ANE_SUBMIT = DRM_IOWR(DRM_COMMAND_BASE + 0x2, DrmAneSubmit)


libc = ctypes.CDLL(None, use_errno=True)
libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
libc.ioctl.restype = ctypes.c_int


class AneIoctlError(OSError):
    """Raised when an ioctl call fails."""


@dataclass
class AneBuffer:
    """Wrapper around an ANE buffer object."""

    device: "ANEDevice"
    size: int
    handle: int
    offset: int
    _freed: bool = False

    def mmap(self) -> mmap.mmap:
        """Return an mmap view over the buffer."""
        if self.device.fd is None:
            raise RuntimeError("Device is not open")
        return mmap.mmap(
            self.device.fd,
            int(self.size),
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_SHARED,
            offset=int(self.offset),
        )

    def zero(self) -> None:
        """Fill the buffer with zeroes."""
        with self.mmap() as mm:
            zero_map(mm)

    def close(self) -> None:
        """Release the buffer."""
        if self._freed:
            return
        if self.device.fd is None:
            raise RuntimeError("Device is not open")
        req = DrmAneBoFree()
        req.handle = self.handle
        drm_ioctl(self.device.fd, IOCTL_ANE_BO_FREE, req)
        self._freed = True

    def __enter__(self) -> "AneBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self.close()
        return None


class ANEDevice:
    """Context manager to manage the ANE DRM device."""

    def __init__(self, path: str = "/dev/dri/renderD129") -> None:
        self.path = path
        self.fd: Optional[int] = None

    def open(self) -> "ANEDevice":
        if self.fd is not None:
            raise RuntimeError("Device is already open")
        self.fd = os.open(self.path, os.O_RDWR)
        return self

    def close(self) -> None:
        if self.fd is None:
            return
        os.close(self.fd)
        self.fd = None

    def allocate_buffer(self, size: int) -> AneBuffer:
        if self.fd is None:
            raise RuntimeError("Device must be opened before allocating buffers")
        req = DrmAneBoInit()
        req.size = size
        drm_ioctl(self.fd, IOCTL_ANE_BO_INIT, req)
        return AneBuffer(self, size=size, handle=req.handle, offset=req.offset)

    def __enter__(self) -> "ANEDevice":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self.close()
        return None


def drm_ioctl(fd: int, request: int, obj: ctypes.Structure) -> int:
    """Thin wrapper over `ioctl` that returns the raw return code."""
    ret = libc.ioctl(fd, request, ctypes.byref(obj))
    if ret != 0:
        err = ctypes.get_errno()
        raise AneIoctlError(os.strerror(err))
    return ret


def zero_map(mm: mmap.mmap) -> None:
    """Zero a memory map using small chunks to avoid large allocations."""
    mm.seek(0)
    length = mm.size()
    chunk = 1 << 20
    offset = 0
    zero_block = b"\x00" * chunk
    while offset < length:
        step = min(chunk, length - offset)
        mm.write(zero_block[:step])
        offset += step
    mm.flush()
