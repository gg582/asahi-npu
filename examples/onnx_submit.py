#!/usr/bin/env python3
"""Submit an ONNX model to the Apple Neural Engine driver using the helper lib."""

from __future__ import annotations

import argparse
from pathlib import Path

from asahi_ane_llm import ANEDevice, parse_ane_metadata, submit_onnx_model


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
    metadata = parse_ane_metadata(model_bytes)

    with ANEDevice(args.device) as device:
        submission = submit_onnx_model(device, model_bytes, metadata)

    print(
        "Submission complete:\n"
        f"  microcode bytes : {submission.tsk_size}\n"
        f"  td entries      : {submission.td_count} x {submission.td_size} bytes",
    )


if __name__ == "__main__":
    main()
