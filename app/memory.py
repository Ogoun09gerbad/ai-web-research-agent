"""Conversational memory management — simple in-memory implementation."""

from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class SessionMemory:
    messages: list[dict[str, str]] = field(default_factory=list)

    def save_context(self, inputs: dict, outputs: dict) -> None:
        self.messages.append({"role": "user", "content": inputs["question"]})
        self.messages.append({"role": "assistant", "content": outputs["answer"]})

    def load_memory_variables(self, _: dict) -> dict:
        from langchain_core.messages import HumanMessage, AIMessage
        history = []
        for msg in self.messages:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        return {"history": history}

    def clear(self) -> None:
        self.messages.clear()


class MemoryManager:
    def __init__(self) -> None:
        self._memories: dict[str, SessionMemory] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> SessionMemory:
        with self._lock:
            if session_id not in self._memories:
                self._memories[session_id] = SessionMemory()
            return self._memories[session_id]

    def clear(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._memories:
                self._memories[session_id].clear()

    def history_as_dicts(self, session_id: str) -> list[dict[str, str]]:
        return list(self.get(session_id).messages)

    def save_turn(self, session_id: str, question: str, answer: str) -> None:
        self.get(session_id).save_context(
            {"question": question}, {"answer": answer}
        )
