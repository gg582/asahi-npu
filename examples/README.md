# Examples

This directory hosts Python helpers that exercise the ANE DRM driver.

## onnx_submit.py

Demonstrates how to push an ONNX model through the in-kernel ingestion path
using the reusable helpers from `asahi_ane_llm`.

```bash
pip install -e libs/asahi_ane_llm
python examples/onnx_submit.py path/to/model.onnx
```

## tinyllama_chat.py

Interactive TinyLlama chat demo. The script uploads a TinyLlama ONNX export to
ANE first and then launches a CPU fallback chat loop driven by Hugging Face
`transformers`.

```bash
pip install -e libs/asahi_ane_llm[chat]
python examples/tinyllama_chat.py \
    path/to/tinyllama-chat.onnx \
    --hf-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Use `--skip-cpu-fallback` if you only want to upload the ONNX model and defer
text generation to a hardware backed pipeline.
