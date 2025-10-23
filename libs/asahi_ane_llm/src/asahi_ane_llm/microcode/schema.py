"""Schema definitions for assembling ANE microprograms.

The format intentionally mirrors the conventions used in `eiln/ane` so that
reverse-engineered opcode maps can be described in a small JSON document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class SchemaError(ValueError):
    """Raised when the schema description is invalid."""


@dataclass(frozen=True)
class FieldLayout:
    """Describe how a numeric field should be packed into a word."""

    name: str
    shift: int
    width: int
    signed: bool = False
    default: int | None = None

    @staticmethod
    def from_mapping(name: str, payload: Mapping[str, Any]) -> "FieldLayout":
        try:
            shift = int(payload["shift"])
            width = int(payload["width"])
        except KeyError as exc:  # pragma: no cover - defensive path
            raise SchemaError(f"Field '{name}' is missing the {exc.args[0]!r} key") from exc
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
            raise SchemaError(
                f"Field '{name}' must define integer 'shift' and 'width' values"
            ) from exc

        signed = bool(payload.get("signed", False))
        default_value = payload.get("default")
        if default_value is not None:
            try:
                default_value = int(default_value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
                raise SchemaError(
                    f"Field '{name}' default must be an integer value"
                ) from exc

        if width <= 0:
            raise SchemaError(f"Field '{name}' width must be positive")
        if shift < 0:
            raise SchemaError(f"Field '{name}' shift must be non-negative")

        return FieldLayout(
            name=name,
            shift=shift,
            width=width,
            signed=signed,
            default=default_value,
        )

    def encode(self, value: int) -> int:
        """Return the bit pattern for *value* positioned according to the layout."""

        max_unsigned = (1 << self.width) - 1
        if self.signed:
            min_value = -(1 << (self.width - 1))
            max_value = (1 << (self.width - 1)) - 1
            if not (min_value <= value <= max_value):
                raise SchemaError(
                    f"Field '{self.name}' expects a signed {self.width}-bit value; "
                    f"got {value}"
                )
            value &= max_unsigned
        else:
            if not (0 <= value <= max_unsigned):
                raise SchemaError(
                    f"Field '{self.name}' expects an unsigned {self.width}-bit value; "
                    f"got {value}"
                )

        return value << self.shift


@dataclass(frozen=True)
class InstructionLayout:
    """Describe how to build a single instruction word."""

    name: str
    opcode: int
    opcode_field: FieldLayout
    fields: Mapping[str, FieldLayout]
    literals: Mapping[str, int]

    @staticmethod
    def from_mapping(
        name: str, payload: Mapping[str, Any], opcode_field: FieldLayout
    ) -> "InstructionLayout":
        try:
            opcode_value = int(payload["opcode"])
        except KeyError as exc:  # pragma: no cover - defensive path
            raise SchemaError(f"Instruction '{name}' is missing the 'opcode' key") from exc
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
            raise SchemaError(
                f"Instruction '{name}' must define an integer 'opcode' value"
            ) from exc

        fields_payload = payload.get("fields", {})
        if not isinstance(fields_payload, Mapping):
            raise SchemaError(
                f"Instruction '{name}' must describe its fields using an object"
            )

        field_layouts = {
            field_name: FieldLayout.from_mapping(field_name, field_payload)
            for field_name, field_payload in fields_payload.items()
        }

        literals_payload = payload.get("literals", {})
        if not isinstance(literals_payload, Mapping):
            raise SchemaError(
                f"Instruction '{name}' literals must be provided as an object"
            )

        literals: dict[str, int] = {}
        for literal_name, raw_value in literals_payload.items():
            try:
                literals[literal_name] = int(raw_value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
                raise SchemaError(
                    f"Instruction '{name}' literal '{literal_name}' must be an integer"
                ) from exc

        return InstructionLayout(
            name=name,
            opcode=opcode_value,
            opcode_field=opcode_field,
            fields=field_layouts,
            literals=literals,
        )

    def assemble(self, parameters: Mapping[str, Any], *, word_mask: int) -> int:
        """Assemble the instruction into an integer word."""

        word = self.opcode_field.encode(self.opcode)
        seen_fields: set[str] = set()

        for name, layout in self.fields.items():
            if name in parameters:
                value_raw = parameters[name]
            else:
                if layout.default is None:
                    raise SchemaError(
                        f"Instruction '{self.name}' is missing the '{name}' parameter"
                    )
                value_raw = layout.default

            try:
                value = int(value_raw)
            except (TypeError, ValueError) as exc:
                raise SchemaError(
                    f"Instruction '{self.name}' parameter '{name}' must be an integer"
                ) from exc

            word |= layout.encode(value)
            seen_fields.add(name)

        for literal_name, literal_value in self.literals.items():
            field = self.fields.get(literal_name)
            if field is None:
                raise SchemaError(
                    f"Instruction '{self.name}' literal '{literal_name}' does not match "
                    "any defined field"
                )
            if literal_name in parameters:
                raise SchemaError(
                    f"Instruction '{self.name}' provides a literal for '{literal_name}' "
                    "but the program also sets it explicitly"
                )
            word |= field.encode(literal_value)

        if word & ~word_mask:
            raise SchemaError(
                f"Instruction '{self.name}' overflows the configured word size"
            )

        return word


@dataclass(frozen=True)
class MicrocodeSchema:
    """Top-level schema that describes the ANE microprogram encoding."""

    word_size: int
    endianness: str
    opcode_field: FieldLayout
    instructions: Mapping[str, InstructionLayout]

    @property
    def word_bytes(self) -> int:
        return self.word_size // 8

    @property
    def word_mask(self) -> int:
        return (1 << self.word_size) - 1

    @staticmethod
    def from_mapping(payload: Mapping[str, Any]) -> "MicrocodeSchema":
        try:
            word_size = int(payload["word_size"])
        except KeyError as exc:  # pragma: no cover - defensive path
            raise SchemaError("Schema is missing the 'word_size' property") from exc
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
            raise SchemaError("'word_size' must be an integer") from exc

        if word_size % 8:
            raise SchemaError("'word_size' must be a multiple of 8 bits")
        if word_size <= 0:
            raise SchemaError("'word_size' must be positive")

        opcode_payload = payload.get("opcode_field")
        if not isinstance(opcode_payload, Mapping):
            raise SchemaError("'opcode_field' must be provided as an object")
        opcode_field = FieldLayout.from_mapping("opcode", opcode_payload)

        endianness = str(payload.get("endianness", "little"))
        if endianness not in {"little", "big"}:
            raise SchemaError("'endianness' must be either 'little' or 'big'")

        instructions_payload = payload.get("instructions", {})
        if not isinstance(instructions_payload, Mapping):
            raise SchemaError("'instructions' must be provided as an object")

        instructions = {
            name: InstructionLayout.from_mapping(name, value, opcode_field)
            for name, value in instructions_payload.items()
        }
        if not instructions:
            raise SchemaError("Schema must define at least one instruction")

        return MicrocodeSchema(
            word_size=word_size,
            endianness=endianness,
            opcode_field=opcode_field,
            instructions=instructions,
        )

    def assemble_program(self, program: list[Mapping[str, Any]]) -> bytes:
        """Assemble a list of instruction descriptions into microcode bytes."""

        output = bytearray()
        mask = self.word_mask

        for index, entry in enumerate(program):
            if not isinstance(entry, Mapping):
                raise SchemaError(
                    f"Program entry {index} must be an object with an 'op' field"
                )

            op = entry.get("op")
            if not isinstance(op, str):
                raise SchemaError(f"Program entry {index} is missing the 'op' field")

            definition = self.instructions.get(op)
            if definition is None:
                available = ", ".join(sorted(self.instructions))
                raise SchemaError(
                    f"Program entry {index} references unknown instruction '{op}'. "
                    f"Available instructions: {available}"
                )

            repeat = entry.get("repeat", 1)
            try:
                repeat_count = int(repeat)
            except (TypeError, ValueError):
                raise SchemaError(
                    f"Program entry {index} repeat count must be an integer"
                ) from None
            if repeat_count <= 0:
                raise SchemaError(
                    f"Program entry {index} repeat count must be positive"
                )

            word = definition.assemble(entry, word_mask=mask)
            word_bytes = word.to_bytes(self.word_bytes, self.endianness)
            output.extend(word_bytes * repeat_count)

        return bytes(output)


__all__ = [
    "FieldLayout",
    "InstructionLayout",
    "MicrocodeSchema",
    "SchemaError",
]
