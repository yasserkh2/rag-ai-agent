from app.graph.state import ChatState
from app.observability import get_logger, summarize_state
from app.services.contracts import IntentRouter
from app.services.router import DefaultIntentRouter

logger = get_logger("graph.router")


class ActiveFlowRouter:
    def __call__(self, state: ChatState) -> str:
        if state.get("handoff_pending"):
            route = "human_escalation"
        elif state.get("active_action") == "appointment_scheduling":
            route = "action_request"
        else:
            route = "classify_intent"
        logger.info("active_flow route=%s state=%s", route, summarize_state(state))
        return route


class GraphRouter:
    def __init__(self, router: IntentRouter) -> None:
        self._router = router

    def __call__(self, state: ChatState) -> str:
        route = self._router.route(state)
        logger.info(
            "intent route=%s intent=%s confidence=%s",
            route,
            state.get("intent"),
            state.get("confidence"),
        )
        return route


_default_active_flow_router = ActiveFlowRouter()
_default_router = GraphRouter(DefaultIntentRouter())


class PostTurnRouter:
    def __call__(self, state: ChatState) -> str:
        if state.get("handoff_pending"):
            route = "human_escalation"
        else:
            route = "response"
        logger.info("post_turn route=%s state=%s", route, summarize_state(state))
        return route


class ServiceResultRouter:
    def __call__(self, state: ChatState) -> str:
        if state.get("handoff_pending"):
            route = "human_escalation"
        else:
            route = "evaluate_escalation"
        logger.info("service_result route=%s state=%s", route, summarize_state(state))
        return route


def route_active_flow(state: ChatState) -> str:
    return _default_active_flow_router(state)


def route_intent(state: ChatState) -> str:
    return _default_router(state)


def route_post_turn(state: ChatState) -> str:
    return PostTurnRouter()(state)


def route_service_result(state: ChatState) -> str:
    return ServiceResultRouter()(state)
