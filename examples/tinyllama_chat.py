#!/usr/bin/env python3
"""Interactive TinyLlama chat demo built on top of the asahi-ane-llm helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from asahi_ane_llm import (
    ANEDevice,
    MissingAneMetadataError,
    parse_ane_metadata,
    submit_onnx_model,
    with_ane_metadata,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:  # pragma: no cover - optional dependency error path
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import onnxruntime_genai as ort_genai
except ImportError:  # pragma: no cover - optional dependency error path
    ort_genai = None  # type: ignore[assignment]


class TransformersGenerator:
    """Wrapper around a Hugging Face model for CPU fallback generation."""

    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "The transformers extras are required. Install them via 'pip install -e libs/asahi_ane_llm[chat]' and ensure PyTorch is available."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        pad_token = self.tokenizer.eos_token or self.tokenizer.pad_token
        if pad_token is None:
            raise RuntimeError("Tokenizer is missing both EOS and PAD tokens")
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(pad_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if torch is not None:
            self.device = torch.device("cpu")
            self.model.to(self.device)
        else:  # pragma: no cover - torch import failure
            self.device = None

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch is not None:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.pad_token_id,
            "top_p": top_p,
        }
        if temperature <= 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature

        output_tokens = self.model.generate(**inputs, **generation_kwargs)
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return output_text[len(prompt) :].strip()


class OnnxRuntimeGenerator:
    """Drive text generation through onnxruntime-genai using the ONNX export."""

    def __init__(self, model_path: Path, provider: str) -> None:
        if ort_genai is None:
            raise RuntimeError(
                "onnxruntime-genai is required for the ONNX fallback. Install it via "
                "'pip install onnxruntime-genai' or 'pip install -e libs/asahi_ane_llm[onnxruntime]'"
            )

        if not model_path.is_file():
            raise RuntimeError(f"Model file '{model_path}' does not exist")

        self.model = ort_genai.Model(str(model_path), provider=provider)
        self.tokenizer = ort_genai.Tokenizer(self.model)
        self.token_stream = ort_genai.TokenizerStream(self.tokenizer)
        self.generator = ort_genai.Generator(self.model, self.tokenizer)

    def _set_search_options(
        self, params: "ort_genai.GeneratorParams", max_new_tokens: int, temperature: float, top_p: float
    ) -> None:
        search_kwargs = {"max_length": max_new_tokens, "top_p": top_p}
        if temperature <= 0:
            search_kwargs["do_sample"] = False
            search_kwargs["temperature"] = 0.0
        else:
            search_kwargs["do_sample"] = True
            search_kwargs["temperature"] = temperature

        if hasattr(params, "set_search_options"):
            params.set_search_options(**search_kwargs)
            return

        # Fallback for older releases that expose attributes instead of the helper.
        for key, value in search_kwargs.items():
            if hasattr(params, key):
                setattr(params, key, value)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        params = ort_genai.GeneratorParams(self.model)
        self._set_search_options(params, max_new_tokens, temperature, top_p)

        if hasattr(params, "set_input_text"):
            params.set_input_text(prompt)
        else:
            encoded = self.tokenizer.encode(prompt)
            if hasattr(params, "set_input_ids"):
                params.set_input_ids(encoded)
            else:
                params.input_ids = encoded

        chunks: List[str] = []
        if hasattr(self.generator, "generate_next"):
            while True:
                chunk = self.generator.generate_next(params)
                if not chunk:
                    break
                decoded = self.token_stream.decode(chunk)
                if decoded:
                    chunks.append(decoded)
        else:  # pragma: no cover - compatibility with older releases
            tokens = self.generator.generate(params)
            if hasattr(tokens, "get"):
                tokens = tokens.get()
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            chunks.append(decoded)

        text = "".join(chunks)
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()


class TinyLlamaChatSession:
    """Maintains prompt history and delegates generation to a backend."""

    def __init__(self, generator: TransformersGenerator, system_prompt: str) -> None:
        self.generator = generator
        self.system_prompt = system_prompt
        self.history: List[Tuple[str, str]] = []

    def build_prompt(self, user_message: str) -> str:
        prompt_lines = [self.system_prompt]
        for role, message in self.history:
            prompt_lines.append(f"<{role}>: {message}")
        prompt_lines.append(f"<user>: {user_message}")
        prompt_lines.append("<assistant>:")
        return "\n".join(prompt_lines)

    def generate_response(
        self,
        user_message: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        prompt = self.build_prompt(user_message)
        completion = self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if "<assistant>:" in completion:
            response_text = completion.split("<assistant>:")[-1]
        else:
            response_text = completion
        response = response_text.strip()
        self.history.append(("user", user_message))
        self.history.append(("assistant", response))
        return response


def upload_model(
    model_path: Path,
    device_path: str,
    *,
    ane_microcode: Path | None,
    ane_weights: Path | None,
    ane_td_size: int | None,
    ane_td_count: int | None,
) -> bool:
    """Upload the ONNX payload to the ANE if metadata is available.

    Returns ``True`` when the submission succeeds and ``False`` when the model is
    missing ANE metadata and no conversion inputs were provided.
    """

    model_bytes = model_path.read_bytes()

    try:
        metadata = parse_ane_metadata(model_bytes)
    except MissingAneMetadataError:
        if not ane_microcode or ane_td_size is None or ane_td_count is None:
            print(
                "Skipping ANE upload: model is missing Asahi metadata and no "
                "conversion inputs were provided. Supply --ane-microcode, "
                "--ane-td-size, and --ane-td-count or pre-process the model "
                "with 'python -m asahi_ane_llm.tools.embed_metadata'.",
                file=sys.stderr,
            )
            return False

        weights_bytes = ane_weights.read_bytes() if ane_weights else None
        model_bytes = with_ane_metadata(
            model_bytes,
            microcode=ane_microcode.read_bytes(),
            td_size=ane_td_size,
            td_count=ane_td_count,
            weights=weights_bytes,
        )
        metadata = parse_ane_metadata(model_bytes)

    with ANEDevice(device_path) as device:
        submission = submit_onnx_model(device, model_bytes, metadata)

    print(
        "Model submission complete:\n"
        f"  microcode bytes : {submission.tsk_size}\n"
        f"  td entries      : {submission.td_count} x {submission.td_size} bytes\n"
        f"  btsp handle     : {submission.btsp_handle}"
    )
    return True


def interactive_chat(session: TinyLlamaChatSession, args: argparse.Namespace) -> None:
    print("Enter 'quit' or press Ctrl-D to exit.")
    while True:
        try:
            user_message = input("You: ")
        except EOFError:
            print()
            break
        if user_message.strip().lower() in {"quit", "exit"}:
            break
        response = session.generate_response(
            user_message,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"TinyLlama: {response}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to the TinyLlama ONNX model")
    parser.add_argument(
        "--device",
        default="/dev/dri/renderD129",
        help="Path to the ANE DRM render node (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model identifier used for the CPU fallback",
    )
    parser.add_argument(
        "--tokenizer",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Tokenizer identifier; defaults to the TinyLlama chat model",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are TinyLlama, a compact assistant running on Asahi's ANE stack.",
        help="System prompt that seeds the conversation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per turn",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for the CPU fallback",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter for the CPU fallback",
    )
    parser.add_argument(
        "--skip-cpu-fallback",
        action="store_true",
        help="Only upload the model without launching the interactive CPU session",
    )
    parser.add_argument(
        "--skip-ane-upload",
        action="store_true",
        help="Do not submit the model to the ANE; useful for plain ONNX testing",
    )
    parser.add_argument(
        "--fallback-backend",
        choices=["auto", "onnxruntime", "transformers"],
        default="auto",
        help=(
            "Backend used when ANE execution is unavailable. 'auto' tries onnxruntime-genai first "
            "and falls back to Hugging Face transformers."
        ),
    )
    parser.add_argument(
        "--onnx-provider",
        default="cpu",
        help="Execution provider passed to onnxruntime-genai (default: %(default)s)",
    )
    parser.add_argument(
        "--ane-microcode",
        type=Path,
        help="Microcode blob to embed when ANE metadata is missing",
    )
    parser.add_argument(
        "--ane-weights",
        type=Path,
        help="Optional ANE weights blob to embed when metadata is missing",
    )
    parser.add_argument(
        "--ane-td-size",
        type=int,
        help="Tile descriptor size to embed when metadata is missing",
    )
    parser.add_argument(
        "--ane-td-count",
        type=int,
        help="Tile descriptor count to embed when metadata is missing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        print(f"Model file '{args.model}' does not exist", file=sys.stderr)
        raise SystemExit(1)

    if args.ane_microcode and not args.ane_microcode.is_file():
        print(f"Microcode file '{args.ane_microcode}' does not exist", file=sys.stderr)
        raise SystemExit(1)
    if args.ane_weights and not args.ane_weights.is_file():
        print(f"Weights file '{args.ane_weights}' does not exist", file=sys.stderr)
        raise SystemExit(1)

    if args.ane_td_size is not None and args.ane_td_size <= 0:
        print("--ane-td-size must be a positive integer", file=sys.stderr)
        raise SystemExit(1)
    if args.ane_td_count is not None and args.ane_td_count <= 0:
        print("--ane-td-count must be a positive integer", file=sys.stderr)
        raise SystemExit(1)

    if not args.skip_ane_upload:
        submitted = upload_model(
            args.model,
            args.device,
            ane_microcode=args.ane_microcode,
            ane_weights=args.ane_weights,
            ane_td_size=args.ane_td_size,
            ane_td_count=args.ane_td_count,
        )
        if not submitted and args.skip_cpu_fallback:
            raise SystemExit(1)
    else:
        submitted = False

    if args.skip_cpu_fallback:
        return

    backend_order: List[str]
    if args.fallback_backend == "auto":
        backend_order = ["onnxruntime", "transformers"]
    else:
        backend_order = [args.fallback_backend]

    generator = None
    last_error: List[str] = []
    selected_backend = None
    for backend in backend_order:
        try:
            if backend == "onnxruntime":
                generator = OnnxRuntimeGenerator(args.model, args.onnx_provider)
            else:
                generator = TransformersGenerator(args.hf_model, args.tokenizer)
            selected_backend = backend
            break
        except RuntimeError as exc:
            last_error.append(f"{backend}: {exc}")
            if args.fallback_backend != "auto":
                break

    if generator is None:
        details = "\n".join(last_error) or "unknown error"
        print(
            "Unable to initialize the requested fallback backend(s):\n"
            f"{details}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.fallback_backend == "auto" and selected_backend != "onnxruntime" and last_error:
        print(
            "onnxruntime fallback unavailable; defaulting to transformers instead.\n"
            f"Details: {last_error[0]}",
            file=sys.stderr,
        )

    session = TinyLlamaChatSession(generator, args.system_prompt)
    interactive_chat(session, args)


if __name__ == "__main__":
    main()
