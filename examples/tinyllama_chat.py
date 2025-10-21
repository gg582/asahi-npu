#!/usr/bin/env python3
"""Interactive TinyLlama chat demo built on top of the asahi-ane-llm helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from asahi_ane_llm import ANEDevice, parse_ane_metadata, submit_onnx_model

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:  # pragma: no cover - optional dependency error path
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]


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


def upload_model(model_path: Path, device_path: str) -> None:
    model_bytes = model_path.read_bytes()
    metadata = parse_ane_metadata(model_bytes)

    with ANEDevice(device_path) as device:
        submission = submit_onnx_model(device, model_bytes, metadata)

    print(
        "Model submission complete:\n"
        f"  microcode bytes : {submission.tsk_size}\n"
        f"  td entries      : {submission.td_count} x {submission.td_size} bytes\n"
        f"  btsp handle     : {submission.btsp_handle}"
    )


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        print(f"Model file '{args.model}' does not exist", file=sys.stderr)
        raise SystemExit(1)

    upload_model(args.model, args.device)

    if args.skip_cpu_fallback:
        return

    generator = TransformersGenerator(args.hf_model, args.tokenizer)
    session = TinyLlamaChatSession(generator, args.system_prompt)
    interactive_chat(session, args)


if __name__ == "__main__":
    main()
