from __future__ import annotations

from dataclasses import dataclass

from app.agents.action_agent import ActionRequestAgent
from app.agents.escalation_agent import HumanEscalationAgent
from app.agents.general_conversation_agent import GeneralConversationAgent
from app.agents.kb_agent import KnowledgeBaseAgent
from app.services.contracts import (
    ActionRequestService,
    EscalationService,
    GeneralConversationService,
    KnowledgeBaseService,
)


@dataclass(frozen=True, slots=True)
class AgentFactory:
    knowledge_base_service: KnowledgeBaseService
    action_request_service: ActionRequestService
    escalation_service: EscalationService
    general_conversation_service: GeneralConversationService

    def build_kb_agent(self) -> KnowledgeBaseAgent:
        return KnowledgeBaseAgent(self.knowledge_base_service)

    def build_action_agent(self) -> ActionRequestAgent:
        return ActionRequestAgent(self.action_request_service)

    def build_escalation_agent(self) -> HumanEscalationAgent:
        return HumanEscalationAgent(self.escalation_service)

    def build_general_conversation_agent(self) -> GeneralConversationAgent:
        return GeneralConversationAgent(self.general_conversation_service)
