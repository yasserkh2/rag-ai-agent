from __future__ import annotations

from dataclasses import dataclass

from app.agents import (
    AgentFactory,
    ActionRequestAgent,
    GeneralConversationAgent,
    HumanEscalationAgent,
    KnowledgeBaseAgent,
)
from app.llm.action_factory import ActionReplyGeneratorFactory
from app.llm.action_extraction import AppointmentExtractorFactory
from app.services.contracts import (
    ActionRequestService,
    ConversationHistoryManager,
    EscalationEvaluator,
    EscalationService,
    GeneralConversationService,
    IntentClassifier,
    IntentRouter,
    KnowledgeBaseService,
)
from app.services.action_request import AppointmentActionService
from app.services.booking_api import LocalMockBookingApiClient
from app.services.escalation import PostTurnEscalationEvaluator
from app.services.history import DefaultConversationHistoryManager
from app.services.intent import LlmIntentClassifier
from app.services.knowledge_base import RetrievalKnowledgeBaseService
from app.services.responses import (
    GeneralConversationService as DefaultGeneralConversationService,
    HumanEscalationService,
)
from app.services.router import DefaultIntentRouter


@dataclass(frozen=True, slots=True)
class GraphDependencies:
    history_manager: ConversationHistoryManager
    intent_classifier: IntentClassifier
    knowledge_base_service: KnowledgeBaseService
    action_request_service: ActionRequestService
    escalation_service: EscalationService
    general_conversation_service: GeneralConversationService
    escalation_evaluator: EscalationEvaluator
    intent_router: IntentRouter
    kb_agent: KnowledgeBaseAgent
    action_agent: ActionRequestAgent
    escalation_agent: HumanEscalationAgent
    general_conversation_agent: GeneralConversationAgent

    @classmethod
    def default(cls) -> "GraphDependencies":
        history_manager = DefaultConversationHistoryManager()
        knowledge_base_service = RetrievalKnowledgeBaseService()
        knowledge_base_service.warmup()
        action_request_service = AppointmentActionService(
            extractor=AppointmentExtractorFactory().build(),
            booking_api_client=LocalMockBookingApiClient(),
            response_generator=ActionReplyGeneratorFactory().build(),
        )
        escalation_service = HumanEscalationService()
        general_conversation_service = DefaultGeneralConversationService()
        escalation_evaluator = PostTurnEscalationEvaluator()
        agent_factory = AgentFactory(
            knowledge_base_service=knowledge_base_service,
            action_request_service=action_request_service,
            escalation_service=escalation_service,
            general_conversation_service=general_conversation_service,
        )
        return cls(
            history_manager=history_manager,
            intent_classifier=LlmIntentClassifier(),
            knowledge_base_service=knowledge_base_service,
            action_request_service=action_request_service,
            escalation_service=escalation_service,
            general_conversation_service=general_conversation_service,
            escalation_evaluator=escalation_evaluator,
            intent_router=DefaultIntentRouter(),
            kb_agent=agent_factory.build_kb_agent(),
            action_agent=agent_factory.build_action_agent(),
            escalation_agent=agent_factory.build_escalation_agent(),
            general_conversation_agent=agent_factory.build_general_conversation_agent(),
        )
