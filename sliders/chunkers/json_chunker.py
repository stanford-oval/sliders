from __future__ import annotations

import json
from typing import Any, Sequence


def serialize_message_array(messages: Sequence[dict[str, Any]]) -> str:
    """Serialize a sequence of messages as a compact JSON array string."""
    if not messages:
        return "[]"
    return "[{}]".format(",".join(json.dumps(message, ensure_ascii=False) for message in messages))


class JsonArrayChunker:
    """Chunker that preserves message boundaries within JSON arrays."""

    def __init__(self, messages: Sequence[dict[str, Any]], chunk_size: int):
        self.messages = list(messages)
        self.chunk_size = chunk_size

    def _make_chunk(self, content: str, indices: list[int], oversized: bool) -> dict[str, Any]:
        return {
            "content": content,
            "metadata": {
                "messages_in_chunk": len(indices),
                "message_indices": indices,
                "characters": len(content),
                "oversized_single": oversized,
            },
        }

    def chunk_text(
        self,
        text: str,
        replace_tables: bool = True,  # noqa: ARG002 - API compatibility
        tag_to_table: dict | None = None,  # noqa: ARG002 - API compatibility
    ) -> list[dict[str, Any]]:
        if not self.messages:
            return [self._make_chunk("[]", [], False)]

        chunks: list[dict[str, Any]] = []
        current_serialized: list[str] = []
        current_indices: list[int] = []
        current_sum_len = 0

        for idx, message in enumerate(self.messages):
            message_str = json.dumps(message, ensure_ascii=False)
            message_len = len(message_str)

            if message_len > self.chunk_size:
                if current_serialized:
                    chunk_content = f"[{','.join(current_serialized)}]"
                    chunks.append(self._make_chunk(chunk_content, current_indices, False))
                    current_serialized = []
                    current_indices = []
                    current_sum_len = 0
                chunks.append(self._make_chunk(message_str, [idx], True))
                continue

            new_count = len(current_serialized) + 1
            tentative_sum = current_sum_len + message_len
            tentative_length = tentative_sum + (new_count - 1) + 2  # commas + brackets

            if not current_serialized or tentative_length <= self.chunk_size:
                current_serialized.append(message_str)
                current_indices.append(idx)
                current_sum_len = tentative_sum
            else:
                chunk_content = f"[{','.join(current_serialized)}]"
                chunks.append(self._make_chunk(chunk_content, current_indices, False))
                current_serialized = [message_str]
                current_indices = [idx]
                current_sum_len = message_len

        if current_serialized:
            chunk_content = f"[{','.join(current_serialized)}]"
            chunks.append(self._make_chunk(chunk_content, current_indices, False))

        if not chunks:
            chunks.append(self._make_chunk("[]", [], False))

        return chunks
