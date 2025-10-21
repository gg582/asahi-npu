// SPDX-License-Identifier: GPL-2.0-only OR MIT
/*
 * ONNX ingestion helpers for the Apple Neural Engine driver.
 */

#ifndef __ANE_ONNX_H__
#define __ANE_ONNX_H__

#include <linux/types.h>

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
