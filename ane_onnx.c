// SPDX-License-Identifier: GPL-2.0-only OR MIT
/*
 * Minimal ONNX metadata parser that extracts the ANE-specific payload stored in
 * metadata_props. The microcode and weights are stored as base64 encoded blobs
 * while the tile descriptor sizing information is stored as decimal strings.
 */

#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/kstrtox.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/types.h>

#include "ane_onnx.h"

struct pb_reader {
        const u8 *ptr;
        const u8 *end;
};

static int pb_get_varint(struct pb_reader *r, u64 *out)
{
        u64 result = 0;
        int shift = 0;

        while (r->ptr < r->end) {
                u8 byte = *r->ptr++;

                result |= (u64)(byte & 0x7f) << shift;
                if (!(byte & 0x80)) {
                        *out = result;
                        return 0;
                }
                shift += 7;
                if (shift >= 64)
                        return -EINVAL;
        }

        return -EINVAL;
}

static int pb_skip_bytes(struct pb_reader *r, size_t len)
{
        if (len > (size_t)(r->end - r->ptr))
                return -EINVAL;
        r->ptr += len;
        return 0;
}

static int pb_get_length_delimited(struct pb_reader *r, const u8 **data,
                                   size_t *len)
{
        u64 raw_len;
        int ret;

        ret = pb_get_varint(r, &raw_len);
        if (ret)
                return ret;

        if (raw_len > (u64)(r->end - r->ptr))
                return -EINVAL;

        *data = r->ptr;
        *len = raw_len;
        r->ptr += raw_len;

        return 0;
}

static int pb_skip_field(struct pb_reader *r, u32 wire_type)
{
        switch (wire_type) {
        case 0: { /* varint */
                u64 tmp;

                return pb_get_varint(r, &tmp);
        }
        case 1: /* 64-bit */
                return pb_skip_bytes(r, 8);
        case 2: { /* length-delimited */
                const u8 *dummy;
                size_t len;

                return pb_get_length_delimited(r, &dummy, &len);
        }
        case 5: /* 32-bit */
                return pb_skip_bytes(r, 4);
        default:
                return -EINVAL;
        }
}

static int base64_value(u8 ch)
{
        if (ch >= 'A' && ch <= 'Z')
                return ch - 'A';
        if (ch >= 'a' && ch <= 'z')
                return ch - 'a' + 26;
        if (ch >= '0' && ch <= '9')
                return ch - '0' + 52;
        if (ch == '+' || ch == '-')
                return 62;
        if (ch == '/' || ch == '_')
                return 63;
        if (ch == '=')
                return -2;
        if (ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ')
                return -3;
        return -1;
}

static int ane_base64_decode(const u8 *src, size_t len, u8 **out,
                             size_t *out_len)
{
        size_t cap = (len / 4) * 3 + 4;
        u8 *dst;
        u32 buffer = 0;
        int bits = 0;
        size_t produced = 0;
        int val;

        dst = kvzalloc(cap, GFP_KERNEL);
        if (!dst)
                return -ENOMEM;

        for (size_t i = 0; i < len; i++) {
                val = base64_value(src[i]);
                if (val == -3)
                        continue;
                if (val == -1) {
                        kvfree(dst);
                        return -EINVAL;
                }
                if (val == -2)
                        break;

                buffer = (buffer << 6) | val;
                bits += 6;
                if (bits >= 8) {
                        bits -= 8;
                        dst[produced++] = (buffer >> bits) & 0xff;
                }
        }

        *out = dst;
        *out_len = produced;
        return 0;
}

static int ane_parse_metadata_entry(const u8 *buf, size_t len,
                                    struct ane_onnx_payload *payload)
{
        struct pb_reader r = {
                .ptr = buf,
                .end = buf + len,
        };
        const u8 *key_ptr = NULL, *value_ptr = NULL;
        size_t key_len = 0, value_len = 0;
        char key[64];
        int ret;

        while (r.ptr < r.end) {
                u64 tag;
                ret = pb_get_varint(&r, &tag);
                if (ret)
                        return ret;

                u32 field = tag >> 3;
                u32 wire = tag & 7;

                if (wire != 2) {
                        ret = pb_skip_field(&r, wire);
                        if (ret)
                                return ret;
                        continue;
                }

                const u8 *data;
                size_t data_len;

                ret = pb_get_length_delimited(&r, &data, &data_len);
                if (ret)
                        return ret;

                if (field == 1) {
                        key_ptr = data;
                        key_len = data_len;
                } else if (field == 2) {
                        value_ptr = data;
                        value_len = data_len;
                }
        }

        if (!key_ptr || !value_ptr)
                return 0;

        if (key_len >= sizeof(key))
                return -EINVAL;

        memcpy(key, key_ptr, key_len);
        key[key_len] = '\0';

        if (!strcmp(key, "ane.microcode.b64")) {
                u8 *decoded;
                size_t decoded_len;

                ret = ane_base64_decode(value_ptr, value_len, &decoded,
                                        &decoded_len);
                if (ret)
                        return ret;

                kvfree(payload->microcode);
                payload->microcode = decoded;
                payload->microcode_size = decoded_len;
        } else if (!strcmp(key, "ane.weights.b64")) {
                u8 *decoded;
                size_t decoded_len;

                ret = ane_base64_decode(value_ptr, value_len, &decoded,
                                        &decoded_len);
                if (ret)
                        return ret;

                kvfree(payload->weights);
                payload->weights = decoded;
                payload->weights_size = decoded_len;
        } else if (!strcmp(key, "ane.td_size")) {
                char value[32];

                if (value_len >= sizeof(value))
                        return -EINVAL;
                memcpy(value, value_ptr, value_len);
                value[value_len] = '\0';

                ret = kstrtou32(value, 10, &payload->td_size);
                if (ret)
                        return ret;
        } else if (!strcmp(key, "ane.td_count")) {
                char value[32];

                if (value_len >= sizeof(value))
                        return -EINVAL;
                memcpy(value, value_ptr, value_len);
                value[value_len] = '\0';

                ret = kstrtou32(value, 10, &payload->td_count);
                if (ret)
                        return ret;
        }

        return 0;
}

int ane_onnx_translate(const void *data, size_t size,
                       struct ane_onnx_payload *payload)
{
        struct pb_reader r = {
                .ptr = data,
                .end = (const u8 *)data + size,
        };
        int ret;

        memset(payload, 0, sizeof(*payload));

        while (r.ptr < r.end) {
                u64 tag;

                ret = pb_get_varint(&r, &tag);
                if (ret)
                        goto fail;

                u32 field = tag >> 3;
                u32 wire = tag & 7;

                if (wire == 2) {
                        const u8 *chunk;
                        size_t chunk_len;

                        ret = pb_get_length_delimited(&r, &chunk, &chunk_len);
                        if (ret)
                                goto fail;

                        if (field == 14) {
                                ret = ane_parse_metadata_entry(chunk, chunk_len,
                                                               payload);
                                if (ret)
                                        goto fail;
                        }
                } else {
                        ret = pb_skip_field(&r, wire);
                        if (ret)
                                goto fail;
                }
        }

        if (!payload->microcode || !payload->microcode_size)
                ret = -EINVAL;
        else if (!payload->td_size || !payload->td_count)
                ret = -EINVAL;
        else
                ret = 0;

fail:
        if (ret)
                ane_onnx_payload_cleanup(payload);
        return ret;
}

void ane_onnx_payload_cleanup(struct ane_onnx_payload *payload)
{
        if (!payload)
                return;

        kvfree(payload->microcode);
        kvfree(payload->weights);
        memset(payload, 0, sizeof(*payload));
}
