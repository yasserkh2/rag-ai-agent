from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.graph.state import ChatState, Intent, TurnOutcome


@dataclass(frozen=True, slots=True)
class IntentDecision:
    intent: Intent
    confidence: float
    frustration_flag: bool = False
    escalation_reason: str | None = None

    def as_state_update(self) -> ChatState:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeBaseAnswer:
    final_response: str
    retrieval_query: str = ""
    retrieved_context: Sequence[str] = ()
    turn_outcome: TurnOutcome = "resolved"
    turn_failure_reason: str | None = None
    escalation_reason: str | None = None

    def as_state_update(self) -> ChatState:
        return {
            "final_response": self.final_response,
            "retrieval_query": self.retrieval_query,
            "retrieved_context": list(self.retrieved_context),
            "turn_outcome": self.turn_outcome,
            "turn_failure_reason": self.turn_failure_reason,
        }
