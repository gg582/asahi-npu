import os, mmap, ctypes, struct
import numpy as np
from fcntl import ioctl

# --- Constants from kernel headers ---
ANE_TILE_COUNT = 0x20
DRM_COMMAND_BASE = 0x40
DRM_IOCTL_BASE = ord('d')

# IOCTL numbers
def _IOWR(nr, size):
    return (3 << 30) | (DRM_IOCTL_BASE << 8) | (size << 16) | nr

DRM_ANE_BO_INIT = 0x1
DRM_ANE_BO_FREE = 0x2
DRM_ANE_SUBMIT = 0x3

# struct drm_ane_bo_init { __u32 handle; __u32 pad; __u64 size; __u64 offset; };
# Size = 4 + 4 + 8 + 8 = 24
IOCTL_ANE_BO_INIT = _IOWR(DRM_COMMAND_BASE + DRM_ANE_BO_INIT, 24)

# struct drm_ane_submit { __u64 tsk_size; __u32 td_count; __u32 td_size; __u32 handles[32]; __u32 btsp_handle; __u32 pad; };
# Size = 8 + 4 + 4 + 128 + 4 + 4 = 152
IOCTL_ANE_SUBMIT = _IOWR(DRM_COMMAND_BASE + DRM_ANE_SUBMIT, 152)

# --- Register Offsets (from relu.py) ---
class reg:
    W0, W1, W2, W3, W4, W5, W6, W7, W8, W9 = 0x00, 0x04, 0x08, 0x0c, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24
    KernelDMA = 0x28
    CommonStream = 0x124
    InDim, OutDim, ChCfg, Cin, Cout = 0x128, 0x13c, 0x130, 0x134, 0x138
    ConvCfg, GroupConvCfg, TileCfg, Cfg = 0x144, 0x14c, 0x150, 0x15c
    TaskInfo, DPE = 0x160, 0x164
    SrcStream, SrcDMAConfig, SrcBaseAddr = 0x168, 0x16c, 0x174
    SrcRowStride, SrcPlaneStride, SrcDepthStride = 0x178, 0x17c, 0x180
    SrcFmt = 0x1a4
    SrcPadStream = 0x1AC
    L2Stream, L2Cfg, SourceCfg, SourceBase = 0x1DC, 0x1e0, 0x1e4, 0x1e8
    SourceChannelStride, SourceRowStride = 0x1ec, 0x1f0
    ResultCfg, ResultBase = 0x210, 0x214
    PEStream, PECfg, BiasScale, PreScale, FinalScale = 0x228, 0x22c, 0x230, 0x234, 0x238
    NEStream, KernelCfg, MACCfg, MatrixVectorBias, AccBias, PostScale = 0x23c, 0x240, 0x244, 0x248, 0x24c, 0x250
    DstStream, DstDMAConfig, DstBaseAddr, DstRowStride = 0x254, 0x258, 0x25c, 0x260
    DstPlaneStride, DstDepthStride, DstGroupStride, DstFmt = 0x264, 0x268, 0x26c, 0x270

def pack_reg(buf, offset, value):
    struct.pack_into('<I', buf, offset, value)

def stream_header(hw_addr, num_words):
    return ((num_words - 1) << 26) | hw_addr

def build_seg(seg_off, seg_len, word_packs):
    max_off = max(boff for boff, _ in word_packs) if word_packs else 0
    tmp = bytearray(max(max_off + 4, seg_off + seg_len))
    for boff, val in word_packs:
        pack_reg(tmp, boff, val)
    return bytes(tmp[seg_off:seg_off + seg_len])

libc = ctypes.CDLL(None, use_errno=True)
libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
libc.ioctl.restype = ctypes.c_int

def drm_ioctl(fd, request, obj):
    ret = libc.ioctl(fd, request, ctypes.byref(obj))
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return ret

class DrmAneBoInit(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
    ]

class DrmAneSubmit(ctypes.Structure):
    _fields_ = [
        ("tsk_size", ctypes.c_uint64),
        ("td_count", ctypes.c_uint32),
        ("td_size", ctypes.c_uint32),
        ("handles", ctypes.c_uint32 * 32),
        ("btsp_handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]

# --- Buffer Management ---
def allocate_buffer(fd, size):
    req = DrmAneBoInit(handle=0, pad=0, size=size, offset=0)
    print(f"Request: size={req.size}")
    drm_ioctl(fd, IOCTL_ANE_BO_INIT, req)
    print(f"Response: handle={req.handle}, size={req.size}, offset=0x{req.offset:016X}")
    buf = mmap.mmap(fd, req.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=req.offset)
    return req.handle, buf

# --- BTSP Payload (ReLU) ---
W = 77
ST = 192
CHANNELS = 1
HALF_ONE = 0x3C00
DMA_EOL = 0x80000000
DMA_ACTIVE = 0x40000000

def create_btsp_buf():
    buf = bytearray(0x4000)
    
    # Task Descriptor
    seg1 = build_seg(0, 44, [
        (reg.W0, (0 << 0) | (0x40 << 16) | (1 << 25)),
        (reg.W2, 1058),
        (reg.W4, 0xFFF86A),
        (reg.W6, (38 << 10) | (3 << 28)),
        (reg.W8, (5) | (1 << 5) | (4 << 12) | (1 << 17) | (1 << 24)),
        (reg.KernelDMA, stream_header(0x1F800, 62)),
    ])
    buf[0:44] = seg1
    
    # Firmware DMA context
    dma_ctx = struct.pack('>' + 'I' * 62, *([0]*2 + [DMA_EOL]*16 + [0]*16 + [DMA_ACTIVE]*16 + [DMA_EOL]*4 + [0]*8))
    buf[44:44+248] = dma_ctx
    
    # Common + TileDMA Src
    seg2 = build_seg(0x124, 184, [
        (reg.CommonStream, stream_header(0x00000, 16)),
        (reg.InDim, (1 << 16) | W),
        (reg.OutDim, (1 << 16) | W),
        (reg.ChCfg, (2) | (2 << 4)),
        (reg.Cin, CHANNELS),
        (reg.Cout, CHANNELS),
        (reg.ConvCfg, (1) | (1 << 5) | (1 << 13) | (1 << 15) | (1 << 28) | (1 << 30)),
        (reg.GroupConvCfg, (1) | (1 << 14) | (1 << 16)),
        (reg.TileCfg, 1),
        (reg.Cfg, (1 << 0) | (1 << 8) | (1 << 16) | (1 << 26)),
        (reg.TaskInfo, (1 << 20)),
        (reg.SrcStream, stream_header(0x13800, 28)),
        (reg.SrcDMAConfig, (1) | (8 << 4) | (8 << 8) | (3 << 12) | (3 << 16)),
        (reg.SrcBaseAddr, 0),
        (reg.SrcRowStride, ST),
        (reg.SrcPlaneStride, ST),
        (reg.SrcDepthStride, ST),
        (reg.SrcFmt, (1) | (3 << 4) | (2 << 12) | (1 << 24)),
        (reg.SrcPadStream, 0x00000100),
    ])
    buf[292:292+184] = seg2
    
    # L2
    seg3 = build_seg(0x1DC, 68, [
        (reg.L2Stream, stream_header(0x04800, 18)),
        (reg.SourceCfg, (2) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 8) | (1 << 20) | (1 << 22)),
        (reg.SourceChannelStride, 0xa0),
        (reg.SourceRowStride, 0xa0),
        (reg.ResultCfg, (2) | (2 << 2) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 8) | (1 << 20) | (1 << 22)),
        (reg.ResultBase, 0xa0),
    ])
    buf[476:476+68] = seg3
    
    # PE + NE
    seg4 = build_seg(0x228, 44, [
        (reg.PEStream, stream_header(0x08800, 4)),
        (reg.NEStream, stream_header(0x0C800, 5)),
        (reg.KernelCfg, (1 << 7)),
        (reg.MACCfg, (12) | (1 << 16) | (1 << 20)),
        (reg.PostScale, HALF_ONE),
    ])
    buf[552:552+44] = seg4
    
    # TileDMA Dst
    seg5 = build_seg(0x254, 32, [
        (reg.DstStream, stream_header(0x17800, 7)),
        (reg.DstDMAConfig, (1) | (12 << 4) | (1 << 26)),
        (reg.DstRowStride, ST),
        (reg.DstPlaneStride, ST),
        (reg.DstDepthStride, ST),
        (reg.DstFmt, (1) | (3 << 4) | (2 << 12) | (1 << 13) | (3 << 20) | (1 << 24)),
    ])
    buf[596:596+32] = seg5
    
    return buf

# --- Main ---
if __name__ == "__main__":
    device_path = "/dev/dri/renderD129"
    print(f"Opening {device_path}...")
    fd = os.open(device_path, os.O_RDWR)
    
    try:
        # Allocate buffers
        print("Allocating buffers...")
        out_handle, out_map = allocate_buffer(fd, 0x4000)
        src1_handle, src1_map = allocate_buffer(fd, 0x4000)
        btsp_handle, btsp_map = allocate_buffer(fd, 0x4000)
        
        # Prepare data
        print("Preparing data...")
        input_a = np.tile(np.array([-3.0, 5.0, -3.0, 5.0], dtype=np.float16), 2048)
        src1_map.write(input_a.tobytes())
        
        BTSP_BUF = create_btsp_buf()
        btsp_map.write(BTSP_BUF)
        
        # Submit task
        print("Submitting task...")
        submit = DrmAneSubmit()
        submit.tsk_size = 0x274
        submit.td_count = 1
        submit.td_size = 0x274
        
        # handles[0]=btsp_handle, handles[1]=0, handles[4]=out_handle, handles[5]=src1_handle
        submit.handles[0] = btsp_handle
        submit.handles[1] = 0
        submit.handles[4] = out_handle
        submit.handles[5] = src1_handle
        
        submit.btsp_handle = btsp_handle
        submit.pad = 0 # Bypassing ONNX parser
        
        print(f"IOCTL command: 0x{IOCTL_ANE_SUBMIT:08X}")
        print(f"Struct size: {ctypes.sizeof(submit)}")
        
        try:
            drm_ioctl(fd, IOCTL_ANE_SUBMIT, submit)
            print("Submission successful!")
        except Exception as e:
            print(f"Submission failed: {e}")
            import traceback
            traceback.print_exc()

        # Check output
        out_map.seek(0)
        output = np.frombuffer(out_map.read(W * 2), dtype=np.float16)
        print("output =", output)
        
    finally:
        os.close(fd)
