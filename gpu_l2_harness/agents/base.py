from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRole:
    """Lightweight role descriptor for handoff-oriented agents."""

    name: str
    doc_path: str
    responsibility: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]

    def handoff_summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "doc_path": self.doc_path,
            "responsibility": self.responsibility,
            "reads": list(self.reads),
            "writes": list(self.writes),
        }

