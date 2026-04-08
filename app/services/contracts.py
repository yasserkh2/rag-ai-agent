from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from app.graph.state import ChatState
from app.services.models import IntentDecision, KnowledgeBaseAnswer


class ConversationHistoryManager(Protocol):
    def normalize_query(self, query: str) -> str:
        ...

    def append_user_message(self, history: Iterable[str], message: str) -> list[str]:
        ...

    def append_assistant_message(
        self, history: Iterable[str], message: str
    ) -> list[str]:
        ...


class IntentClassifier(Protocol):
    def classify(self, state: ChatState) -> IntentDecision:
        ...


class IntentRouter(Protocol):
    def route(self, state: ChatState) -> str:
        ...


class KnowledgeBaseService(Protocol):
    def answer(self, state: ChatState) -> KnowledgeBaseAnswer:
        ...


class RetrievalQueryRewriter(Protocol):
    def rewrite(self, query: str, history: list[str]) -> str:
        ...


class ActionRequestService(Protocol):
    def handle_turn(self, state: ChatState) -> ChatState:
        ...


class EscalationService(Protocol):
    def build_response(self, state: ChatState) -> str:
        ...


class GeneralConversationService(Protocol):
    def build_response(self, state: ChatState) -> str:
        ...


class EscalationEvaluator(Protocol):
    def evaluate(self, state: ChatState) -> ChatState:
        ...
