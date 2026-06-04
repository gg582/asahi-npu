// SPDX-License-Identifier: GPL-2.0-only OR MIT
/*
 * ONNX ingestion helpers for the Apple Neural Engine driver.
 */

#ifndef __ANE_ONNX_H__
#define __ANE_ONNX_H__

#include <linux/types.h>

#define ANE_KONNX_MAGIC "KONNX\0\0\0"
#define ANE_KONNX_VERSION_MAJOR 1
#define ANE_KONNX_VERSION_MINOR 0
#define ANE_KONNX_HEADER_SIZE 80

struct ane_konnx_header {
	u8 magic[8];
	__le16 version_major;
	__le16 version_minor;
	__le32 header_size;
	__le32 flags;

	__le32 microcode_offset;
	__le32 microcode_size;

	__le32 weights_offset;
	__le32 weights_size;

	__le32 tile_desc_offset;
	__le32 tile_desc_size;

	__le32 td_size;
	__le32 td_count;

	__le32 onnx_offset;
	__le32 onnx_size;

	__le32 reserved[5];
} __packed;

struct ane_onnx_payload {
        void *microcode;
        size_t microcode_size;
        void *weights;
        size_t weights_size;
        u32 td_size;
        u32 td_count;
};

int ane_onnx_translate(const void *data, size_t size,
                       struct ane_onnx_payload *payload);
void ane_onnx_payload_cleanup(struct ane_onnx_payload *payload);

#endif /* __ANE_ONNX_H__ */
