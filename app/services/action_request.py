from __future__ import annotations

import re
from typing import Any, Protocol, cast

from app.graph.state import ChatState, TurnOutcome
from app.llm.contracts import ActionReplyGenerator
from app.observability import get_logger, truncate_text
from app.services.action_models import (
    APPOINTMENT_SERVICE_OPTIONS,
    AppointmentActionReplyContext,
    AppointmentAvailabilityRequest,
    AppointmentAvailabilityResult,
    AppointmentBookingRequest,
    AppointmentBookingResult,
    AppointmentDateAvailabilityRequest,
    AppointmentDateAvailabilityResult,
    AppointmentExtraction,
    missing_appointment_fields,
)

_EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE)
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,98}[A-Za-z.]?$")

logger = get_logger("services.action_request")


class AppointmentExtractor(Protocol):
    def extract(
        self,
        user_query: str,
        conversation_history: list[str],
        current_slots: dict[str, str],
        offered_dates: list[str] | None = None,
        offered_times: list[str] | None = None,
        offered_services: list[str] | None = None,
        awaiting_confirmation: bool = False,
    ) -> AppointmentExtraction:
        ...


class AppointmentBookingApiClient(Protocol):
    def get_available_dates(
        self,
        request: AppointmentDateAvailabilityRequest,
    ) -> AppointmentDateAvailabilityResult:
        ...

    def get_availability(
        self,
        request: AppointmentAvailabilityRequest,
    ) -> AppointmentAvailabilityResult:
        ...

    def create_booking(
        self,
        request: AppointmentBookingRequest,
    ) -> AppointmentBookingResult:
        ...

    def get_booking(self, confirmation_id: str) -> AppointmentBookingResult:
        ...


class AppointmentActionService:
    def __init__(
        self,
        extractor: AppointmentExtractor,
        booking_api_client: AppointmentBookingApiClient,
        response_generator: ActionReplyGenerator,
    ) -> None:
        self._extractor = extractor
        self._booking_api_client = booking_api_client
        self._response_generator = response_generator

    def handle_turn(self, state: ChatState) -> ChatState:
        query = state.get("user_query", "").strip()
        history = list(state.get("history", []))
        current_slots = self._current_slots(state)
        available_dates = list(state.get("available_dates", []))
        available_slots = list(state.get("available_slots", []))
        date_confirmed = bool(state.get("date_confirmed"))
        time_confirmed = bool(state.get("time_confirmed"))
        awaiting_confirmation = bool(state.get("awaiting_confirmation"))
        invalid_field: str | None = None
        validation_error: str | None = None
        logger.info(
            "action turn start query='%s' slots=%s awaiting_confirmation=%s available_dates=%s available_slots=%s",
            truncate_text(query, 120),
            current_slots,
            awaiting_confirmation,
            available_dates,
            available_slots,
        )

        try:
            extraction = self._extract_with_llm(
                user_query=query,
                conversation_history=history,
                current_slots=current_slots,
                offered_dates=available_dates,
                offered_times=available_slots,
                awaiting_confirmation=awaiting_confirmation,
            )
            logger.info(
                "action extraction result service=%s selected_date=%s selected_time=%s confirmation=%s name=%s email=%s",
                extraction.selected_service,
                extraction.selected_date,
                extraction.selected_time,
                extraction.confirmation_intent,
                bool(extraction.name),
                bool(extraction.email),
            )
        except Exception as exc:
            logger.exception("action extraction failed: %s", exc)
            return self._state_update(
                current_slots=current_slots,
                available_dates=available_dates,
                available_slots=available_slots,
                date_confirmed=date_confirmed,
                time_confirmed=time_confirmed,
                awaiting_confirmation=awaiting_confirmation,
                booking_error="action_extraction_failed",
                turn_outcome="unresolved",
                turn_failure_reason="action_extraction_failed",
                escalation_reason=(
                    "I need to transfer this appointment request to a human "
                    "agent because I could not process the scheduling details."
                ),
                final_response=self._build_llm_error_reply(
                    stage="extract appointment details",
                    error=exc,
                ),
            )

        confirmation_ready = awaiting_confirmation and (
            self._next_required_field(
                current_slots=current_slots,
                date_confirmed=date_confirmed,
                time_confirmed=time_confirmed,
            )
            is None
        )
        explicit_change_request = extraction.confirmation_intent == "change"
        allow_service_update = (
            explicit_change_request
            or (not confirmation_ready and not current_slots.get("service"))
        )
        allow_date_update = explicit_change_request or (
            not confirmation_ready
            and (
                not current_slots.get("date")
                or not date_confirmed
            )
        )
        allow_time_update = explicit_change_request or (
            not confirmation_ready
            and (
                not current_slots.get("time")
                or not time_confirmed
            )
        )
        allow_name_update = (
            explicit_change_request
            or (not confirmation_ready and not current_slots.get("name"))
        )
        allow_email_update = (
            explicit_change_request
            or (not confirmation_ready and not current_slots.get("email"))
        )

        slot_updates = extraction.as_slot_updates()
        if not allow_service_update:
            slot_updates.pop("service", None)
        if not allow_date_update:
            slot_updates.pop("date", None)
        if not allow_time_update:
            slot_updates.pop("time", None)
        if not allow_name_update:
            slot_updates.pop("name", None)
        if not allow_email_update:
            slot_updates.pop("email", None)

        current_slots.update(slot_updates)
        if allow_service_update and extraction.selected_service in APPOINTMENT_SERVICE_OPTIONS:
            current_slots["service"] = extraction.selected_service
        elif allow_service_update and extraction.selected_service:
            invalid_field = "service"
            validation_error = "Please choose one of the listed services."
            logger.info("action service validation failed for service='%s'", extraction.selected_service)

        if allow_date_update and available_dates and extraction.selected_date and extraction.selected_date not in available_dates:
            invalid_field = "date"
            validation_error = "Please choose one of the available dates."
            logger.info("action service rejected selected date='%s'", extraction.selected_date)
        elif allow_date_update and extraction.selected_date:
            current_slots["date"] = extraction.selected_date
            available_dates = []
            date_confirmed = True
            time_confirmed = False
            availability = self._lookup_availability(
                current_slots=current_slots,
                time_preference=extraction.time_preference,
            )
            available_slots = availability.slots if availability else []
            if available_slots:
                return self._state_update(
                    current_slots=current_slots,
                    available_dates=[],
                    available_slots=available_slots,
                    date_confirmed=True,
                    time_confirmed=False,
                    awaiting_confirmation=False,
                    final_response=self._build_action_reply(
                        phase="choose_time",
                        user_query=query,
                        conversation_history=history,
                        current_slots=current_slots,
                        available_dates=[],
                        available_slots=available_slots,
                        date_confirmed=True,
                        time_confirmed=False,
                        awaiting_confirmation=False,
                    ),
                )
            available_slots = []
        elif allow_date_update and extraction.date:
            current_slots["date"] = extraction.date
            date_confirmed = False

        if allow_time_update and available_slots and extraction.selected_time and extraction.selected_time not in available_slots:
            invalid_field = "time"
            validation_error = "Please choose one of the available times."
            logger.info("action service rejected selected time='%s'", extraction.selected_time)
        elif allow_time_update and extraction.selected_time:
            current_slots["time"] = extraction.selected_time
            available_slots = []
            time_confirmed = True
        elif allow_time_update and extraction.time:
            time_confirmed = False

        (
            current_slots,
            invalid_field,
            validation_error,
        ) = self._validate_slot_values(
            current_slots=current_slots,
            invalid_field=invalid_field,
            validation_error=validation_error,
            extraction=extraction,
        )

        if awaiting_confirmation:
            next_required_field = self._next_required_field(
                current_slots=current_slots,
                date_confirmed=date_confirmed,
                time_confirmed=time_confirmed,
            )
            if next_required_field is None:
                if extraction.confirmation_intent == "confirm":
                    return self._book_appointment(current_slots=current_slots)
                if extraction.confirmation_intent == "change":
                    return self._state_update(
                        current_slots=current_slots,
                        available_dates=available_dates,
                        available_slots=available_slots,
                        date_confirmed=date_confirmed,
                        time_confirmed=time_confirmed,
                        awaiting_confirmation=False,
                        final_response=self._build_action_reply(
                            phase="change_request",
                            user_query=query,
                            conversation_history=history,
                            current_slots=current_slots,
                            available_dates=available_dates,
                            available_slots=available_slots,
                            date_confirmed=date_confirmed,
                            time_confirmed=time_confirmed,
                            awaiting_confirmation=False,
                            invalid_field=invalid_field,
                            validation_error=validation_error,
                        ),
                    )
                return self._state_update(
                    current_slots=current_slots,
                    available_dates=available_dates,
                    available_slots=available_slots,
                    date_confirmed=date_confirmed,
                    time_confirmed=time_confirmed,
                    awaiting_confirmation=True,
                    final_response=self._build_action_reply(
                        phase="awaiting_confirmation",
                        user_query=query,
                        conversation_history=history,
                        current_slots=current_slots,
                        available_dates=available_dates,
                        available_slots=available_slots,
                        date_confirmed=date_confirmed,
                        time_confirmed=time_confirmed,
                        awaiting_confirmation=True,
                        invalid_field=invalid_field,
                        validation_error=validation_error,
                    ),
                )
            if extraction.confirmation_intent == "confirm":
                awaiting_confirmation = False
            elif extraction.confirmation_intent == "change":
                return self._state_update(
                    current_slots=current_slots,
                    available_dates=available_dates,
                    available_slots=available_slots,
                    date_confirmed=date_confirmed,
                    time_confirmed=time_confirmed,
                    awaiting_confirmation=False,
                    final_response=self._build_action_reply(
                        phase="change_request",
                        user_query=query,
                        conversation_history=history,
                        current_slots=current_slots,
                        available_dates=available_dates,
                        available_slots=available_slots,
                        date_confirmed=date_confirmed,
                        time_confirmed=time_confirmed,
                        awaiting_confirmation=False,
                        invalid_field=invalid_field,
                        validation_error=validation_error,
                    ),
                )
            elif (
                extraction.as_slot_updates()
                or extraction.selected_service
                or extraction.selected_date
                or extraction.selected_time
            ):
                awaiting_confirmation = False
            else:
                return self._state_update(
                    current_slots=current_slots,
                    available_dates=available_dates,
                    available_slots=available_slots,
                    date_confirmed=date_confirmed,
                    time_confirmed=time_confirmed,
                    awaiting_confirmation=True,
                    final_response=self._build_action_reply(
                        phase="awaiting_confirmation",
                        user_query=query,
                        conversation_history=history,
                        current_slots=current_slots,
                        available_dates=available_dates,
                        available_slots=available_slots,
                        date_confirmed=date_confirmed,
                        time_confirmed=time_confirmed,
                        awaiting_confirmation=True,
                        invalid_field=invalid_field,
                        validation_error=validation_error,
                    ),
                )

        if self._should_validate_date_selection(
            current_slots=current_slots,
            date_confirmed=date_confirmed,
        ):
            date_availability = self._lookup_available_dates(current_slots=current_slots)
            available_dates = (
                date_availability.available_dates if date_availability else []
            )
            if available_dates:
                suggested_date = current_slots.pop("date", None)
                return self._state_update(
                    current_slots=current_slots,
                    available_dates=available_dates,
                    available_slots=[],
                    date_confirmed=False,
                    time_confirmed=False,
                    awaiting_confirmation=False,
                    final_response=self._build_action_reply(
                        phase="choose_date",
                        user_query=query,
                        conversation_history=history,
                        current_slots=current_slots,
                        available_dates=available_dates,
                        available_slots=[],
                        date_confirmed=False,
                        time_confirmed=False,
                        awaiting_confirmation=False,
                        suggested_date=suggested_date,
                        invalid_field=invalid_field,
                        validation_error=validation_error,
                    ),
                )

        if self._should_validate_time_selection(
            current_slots=current_slots,
            date_confirmed=date_confirmed,
            time_confirmed=time_confirmed,
        ):
            availability = self._lookup_availability(
                current_slots=current_slots,
                time_preference=extraction.time_preference,
            )
            available_slots = availability.slots if availability else []
            if available_slots:
                suggested_time = current_slots.pop("time", None)
                return self._state_update(
                    current_slots=current_slots,
                    available_dates=[],
                    available_slots=available_slots,
                    date_confirmed=True,
                    time_confirmed=False,
                    awaiting_confirmation=False,
                    final_response=self._build_action_reply(
                        phase="choose_time",
                        user_query=query,
                        conversation_history=history,
                        current_slots=current_slots,
                        available_dates=[],
                        available_slots=available_slots,
                        date_confirmed=True,
                        time_confirmed=False,
                        awaiting_confirmation=False,
                        suggested_time=suggested_time,
                        invalid_field=invalid_field,
                        validation_error=validation_error,
                    ),
                )

        missing_fields = missing_appointment_fields(current_slots)
        suggested_service = self._infer_service_from_history(
            user_query=query,
            conversation_history=history,
            current_slots=current_slots,
        )
        if (
            "date" in missing_fields
            and current_slots.get("service")
            and not available_dates
        ):
            date_availability = self._lookup_available_dates(current_slots=current_slots)
            available_dates = (
                date_availability.available_dates if date_availability else []
            )
            if available_dates:
                return self._state_update(
                    current_slots=current_slots,
                    available_dates=available_dates,
                    available_slots=[],
                    date_confirmed=False,
                    time_confirmed=False,
                    awaiting_confirmation=False,
                    final_response=self._build_action_reply(
                        phase="choose_date",
                        user_query=query,
                        conversation_history=history,
                        current_slots=current_slots,
                        available_dates=available_dates,
                        available_slots=[],
                        date_confirmed=False,
                        time_confirmed=False,
                        awaiting_confirmation=False,
                        suggested_service=suggested_service,
                        invalid_field=invalid_field,
                        validation_error=validation_error,
                    ),
                )

        if not missing_fields and date_confirmed and time_confirmed:
            return self._state_update(
                current_slots=current_slots,
                available_dates=[],
                available_slots=[],
                date_confirmed=date_confirmed,
                time_confirmed=time_confirmed,
                awaiting_confirmation=True,
                final_response=self._build_action_reply(
                    phase="confirm",
                    user_query=query,
                    conversation_history=history,
                    current_slots=current_slots,
                    available_dates=[],
                    available_slots=[],
                    date_confirmed=date_confirmed,
                    time_confirmed=time_confirmed,
                    awaiting_confirmation=True,
                    suggested_service=suggested_service,
                    invalid_field=invalid_field,
                    validation_error=validation_error,
                ),
            )

        if self._should_offer_slots(
            missing_fields=missing_fields,
            current_slots=current_slots,
            extraction=extraction,
            available_slots=available_slots,
        ):
            availability = self._lookup_availability(
                current_slots=current_slots,
                time_preference=extraction.time_preference,
            )
            available_slots = availability.slots if availability else []

            if available_slots:
                return self._state_update(
                    current_slots=current_slots,
                    available_dates=[],
                    available_slots=available_slots,
                    date_confirmed=True,
                    time_confirmed=False,
                    awaiting_confirmation=False,
                    final_response=self._build_action_reply(
                        phase="choose_time",
                        user_query=query,
                        conversation_history=history,
                        current_slots=current_slots,
                        available_dates=[],
                        available_slots=available_slots,
                        date_confirmed=True,
                        time_confirmed=False,
                        awaiting_confirmation=False,
                        suggested_service=suggested_service,
                        invalid_field=invalid_field,
                        validation_error=validation_error,
                    ),
                )

        return self._state_update(
            current_slots=current_slots,
            available_dates=available_dates if "date" not in missing_fields else [],
            available_slots=available_slots if "time" not in missing_fields else [],
            date_confirmed=date_confirmed if "date" not in missing_fields else False,
            time_confirmed=time_confirmed if "time" not in missing_fields else False,
            awaiting_confirmation=False,
            final_response=self._build_action_reply(
                phase="collecting",
                user_query=query,
                conversation_history=history,
                current_slots=current_slots,
                available_dates=available_dates if "date" not in missing_fields else [],
                available_slots=available_slots if "time" not in missing_fields else [],
                date_confirmed=date_confirmed if "date" not in missing_fields else False,
                time_confirmed=time_confirmed if "time" not in missing_fields else False,
                awaiting_confirmation=False,
                suggested_service=suggested_service,
                invalid_field=invalid_field,
                validation_error=validation_error,
            ),
        )

    def _book_appointment(self, current_slots: dict[str, str]) -> ChatState:
        logger.info("action booking attempt slots=%s", current_slots)
        try:
            booking_result = self._booking_api_client.create_booking(
                AppointmentBookingRequest(
                    service=current_slots["service"],
                    date=current_slots["date"],
                    time=current_slots["time"],
                    name=current_slots["name"],
                    email=current_slots["email"],
                    title=current_slots.get("service"),
                )
            )
        except Exception as exc:
            logger.exception("action booking failed: %s", exc)
            return self._state_update(
                current_slots=current_slots,
                available_dates=[],
                available_slots=[],
                date_confirmed=False,
                time_confirmed=False,
                awaiting_confirmation=False,
                booking_error="booking_request_failed",
                turn_outcome="unresolved",
                turn_failure_reason="booking_request_failed",
                escalation_reason=(
                    "I need to transfer this appointment request to a human "
                    "agent because the booking step could not be completed."
                ),
                final_response=self._build_action_reply(
                    phase="booking_error",
                    user_query="",
                    conversation_history=[],
                    current_slots=current_slots,
                    available_dates=[],
                    available_slots=[],
                    date_confirmed=False,
                    time_confirmed=False,
                    awaiting_confirmation=False,
                    booking_error="booking_request_failed",
                    invalid_field=None,
                    validation_error=None,
                ),
            )

        booking_result_payload: dict[str, Any] = {
            "confirmation_id": booking_result.confirmation_id,
            "service": booking_result.service,
            "date": booking_result.date,
            "time": booking_result.time,
            "name": booking_result.name,
            "email": booking_result.email,
            "saved_booking": booking_result.saved_booking,
        }
        logger.info(
            "action booking succeeded confirmation_id=%s",
            booking_result.confirmation_id,
        )
        return {
            "active_action": None,
            "appointment_slots": {},
            "missing_slots": [],
            "available_dates": [],
            "date_confirmed": False,
            "available_slots": [],
            "time_confirmed": False,
            "awaiting_confirmation": False,
            "booking_confirmation_id": booking_result.confirmation_id,
            "booking_result": booking_result_payload,
            "booking_error": None,
            "turn_outcome": "resolved",
            "turn_failure_reason": None,
            "escalation_reason": None,
            "final_response": self._build_action_reply(
                phase="booking_success",
                user_query="",
                conversation_history=[],
                current_slots={},
                available_dates=[],
                available_slots=[],
                date_confirmed=False,
                time_confirmed=False,
                awaiting_confirmation=False,
                booking_result=booking_result_payload,
                invalid_field=None,
                validation_error=None,
            ),
        }

    def _extract_with_llm(
        self,
        user_query: str,
        conversation_history: list[str],
        current_slots: dict[str, str],
        offered_dates: list[str] | None = None,
        offered_times: list[str] | None = None,
        awaiting_confirmation: bool = False,
    ) -> AppointmentExtraction:
        return self._extractor.extract(
            user_query=user_query,
            conversation_history=conversation_history,
            current_slots=current_slots,
            offered_dates=offered_dates,
            offered_times=offered_times,
            offered_services=list(APPOINTMENT_SERVICE_OPTIONS),
            awaiting_confirmation=awaiting_confirmation,
        )

    def _should_offer_slots(
        self,
        missing_fields: list[str],
        current_slots: dict[str, str],
        extraction: AppointmentExtraction,
        available_slots: list[str],
    ) -> bool:
        if "time" not in missing_fields:
            return False
        if not current_slots.get("service") or not current_slots.get("date"):
            return False
        if available_slots:
            return True
        return bool(extraction.time_preference)

    def _should_validate_time_selection(
        self,
        current_slots: dict[str, str],
        date_confirmed: bool,
        time_confirmed: bool,
    ) -> bool:
        return bool(
            current_slots.get("service")
            and current_slots.get("date")
            and date_confirmed
            and current_slots.get("time")
            and not time_confirmed
        )

    def _should_validate_date_selection(
        self,
        current_slots: dict[str, str],
        date_confirmed: bool,
    ) -> bool:
        return bool(
            current_slots.get("service")
            and current_slots.get("date")
            and not date_confirmed
        )

    def _validate_slot_values(
        self,
        current_slots: dict[str, str],
        invalid_field: str | None,
        validation_error: str | None,
        extraction: AppointmentExtraction,
    ) -> tuple[dict[str, str], str | None, str | None]:
        validated_slots = dict(current_slots)

        name = validated_slots.get("name")
        if name:
            normalized_name = self._normalize_name(name)
            if normalized_name is None:
                validated_slots.pop("name", None)
                if invalid_field is None and extraction.name:
                    invalid_field = "name"
                    validation_error = "Please share the name to use for the booking."
            else:
                validated_slots["name"] = normalized_name

        email = validated_slots.get("email")
        if email:
            normalized_email = self._normalize_email(email)
            if normalized_email is None:
                validated_slots.pop("email", None)
                if invalid_field is None and extraction.email:
                    invalid_field = "email"
                    validation_error = (
                        "Please provide a complete email address, for example name@example.com."
                    )
            else:
                validated_slots["email"] = normalized_email

        return validated_slots, invalid_field, validation_error

    def _next_required_field(
        self,
        current_slots: dict[str, str],
        date_confirmed: bool,
        time_confirmed: bool,
    ) -> str | None:
        if not current_slots.get("service"):
            return "service"
        if not current_slots.get("date") or not date_confirmed:
            return "date"
        if not current_slots.get("time") or not time_confirmed:
            return "time"
        if not current_slots.get("name"):
            return "name"
        if not current_slots.get("email"):
            return "email"
        return None

    def _build_action_reply(
        self,
        phase: str,
        user_query: str,
        conversation_history: list[str],
        current_slots: dict[str, str],
        available_dates: list[str],
        available_slots: list[str],
        date_confirmed: bool,
        time_confirmed: bool,
        awaiting_confirmation: bool,
        suggested_service: str | None = None,
        suggested_date: str | None = None,
        suggested_time: str | None = None,
        booking_result: dict[str, object] | None = None,
        booking_error: str | None = None,
        invalid_field: str | None = None,
        validation_error: str | None = None,
    ) -> str:
        context = AppointmentActionReplyContext(
            phase=phase,
            user_query=user_query,
            conversation_history=conversation_history,
            current_slots=dict(current_slots),
            missing_fields=missing_appointment_fields(current_slots),
            next_required_field=self._next_required_field(
                current_slots=current_slots,
                date_confirmed=date_confirmed,
                time_confirmed=time_confirmed,
            ),
            service_options=list(APPOINTMENT_SERVICE_OPTIONS),
            available_dates=list(available_dates),
            available_slots=list(available_slots),
            awaiting_confirmation=awaiting_confirmation,
            date_confirmed=date_confirmed,
            time_confirmed=time_confirmed,
            suggested_service=suggested_service,
            suggested_date=suggested_date,
            suggested_time=suggested_time,
            booking_result=dict(booking_result) if booking_result else None,
            booking_error=booking_error,
            invalid_field=invalid_field,
            validation_error=validation_error,
        )
        reply = self._response_generator.generate_reply(context).strip()
        if not reply:
            raise RuntimeError("Action reply generator returned an empty reply.")
        return reply

    @staticmethod
    def _build_llm_error_reply(stage: str, error: Exception) -> str:
        error_message = str(error).strip() or error.__class__.__name__
        return f"Action LLM failed to {stage}: {error_message}"

    def _current_slots(self, state: ChatState) -> dict[str, str]:
        raw_slots: object = state.get("appointment_slots", {})
        if not isinstance(raw_slots, dict):
            return {}
        raw_slots_dict = cast(dict[object, object], raw_slots)
        cleaned: dict[str, str] = {}
        for raw_key, raw_value in raw_slots_dict.items():
            if not isinstance(raw_key, str) or not isinstance(raw_value, str):
                continue
            key = raw_key.strip()
            value = raw_value.strip()
            if not key or not value:
                continue
            cleaned[key] = value
        return cleaned

    def _state_update(
        self,
        current_slots: dict[str, str],
        available_dates: list[str],
        available_slots: list[str],
        date_confirmed: bool,
        time_confirmed: bool,
        awaiting_confirmation: bool,
        final_response: str,
        booking_error: str | None = None,
        turn_outcome: TurnOutcome = "needs_input",
        turn_failure_reason: str | None = None,
        escalation_reason: str | None = None,
        invalid_field: str | None = None,
        validation_error: str | None = None,
    ) -> ChatState:
        update: ChatState = {
            "active_action": "appointment_scheduling",
            "appointment_slots": current_slots,
            "missing_slots": missing_appointment_fields(current_slots),
            "available_dates": available_dates,
            "available_slots": available_slots,
            "date_confirmed": date_confirmed,
            "time_confirmed": time_confirmed,
            "awaiting_confirmation": awaiting_confirmation,
            "booking_confirmation_id": None,
            "booking_result": None,
            "booking_error": booking_error,
            "turn_outcome": turn_outcome,
            "turn_failure_reason": turn_failure_reason,
            "escalation_reason": escalation_reason,
            "final_response": final_response,
        }
        logger.info(
            "action state update slots=%s missing=%s available_dates=%s available_slots=%s turn_outcome=%s invalid_field=%s booking_error=%s",
            current_slots,
            update["missing_slots"],
            available_dates,
            available_slots,
            turn_outcome,
            invalid_field,
            booking_error,
        )
        return update

    @staticmethod
    def _normalize_email(value: str) -> str | None:
        normalized = value.strip().lower()
        if not normalized or not _EMAIL_PATTERN.fullmatch(normalized):
            return None
        return normalized

    @staticmethod
    def _normalize_name(value: str) -> str | None:
        normalized = " ".join(value.strip().split())
        if not normalized or not _NAME_PATTERN.fullmatch(normalized):
            return None
        return normalized.title()

    def _lookup_availability(
        self,
        current_slots: dict[str, str],
        time_preference: str | None,
    ) -> AppointmentAvailabilityResult | None:
        try:
            logger.info(
                "action lookup availability service=%s date=%s time_preference=%s",
                current_slots["service"],
                current_slots["date"],
                time_preference,
            )
            result = self._booking_api_client.get_availability(
                AppointmentAvailabilityRequest(
                    service=current_slots["service"],
                    date=current_slots["date"],
                    time_preference=time_preference,
                )
            )
            logger.info("action availability result slots=%s", result.slots)
            return result
        except Exception as exc:
            logger.exception("action availability lookup failed: %s", exc)
            return None

    def _lookup_available_dates(
        self,
        current_slots: dict[str, str],
    ) -> AppointmentDateAvailabilityResult | None:
        try:
            logger.info(
                "action lookup available dates service=%s date_preference=%s",
                current_slots["service"],
                current_slots.get("date"),
            )
            result = self._booking_api_client.get_available_dates(
                AppointmentDateAvailabilityRequest(
                    service=current_slots["service"],
                    date_preference=current_slots.get("date"),
                )
            )
            logger.info("action available dates result=%s", result.available_dates)
            return result
        except Exception as exc:
            logger.exception("action available dates lookup failed: %s", exc)
            return None

    def _infer_service_from_history(
        self,
        user_query: str,
        conversation_history: list[str],
        current_slots: dict[str, str],
    ) -> str | None:
        if current_slots.get("service"):
            return None

        searchable_lines: list[str] = []
        if user_query.strip():
            searchable_lines.append(user_query)
        for raw_line in conversation_history[-8:]:
            user_message = self._extract_user_message(str(raw_line))
            if user_message:
                searchable_lines.append(user_message)
        lowered_options = {
            option.lower(): option for option in APPOINTMENT_SERVICE_OPTIONS
        }
        service_aliases: dict[str, str] = {
            "credentialing": "Credentialing and Provider Maintenance",
            "provider maintenance": "Credentialing and Provider Maintenance",
            "authorizations": "Authorizations and Benefits Verification",
            "authorization": "Authorizations and Benefits Verification",
            "benefits verification": "Authorizations and Benefits Verification",
            "medical billing": "Medical Billing and Denial Management",
            "denial management": "Medical Billing and Denial Management",
            "medical auditing": "Medical Auditing",
            "auditing": "Medical Auditing",
            "communication services": "Communication Services",
            "financial management": "Financial Management",
            "digital marketing": "Digital Marketing and Website Services",
            "website services": "Digital Marketing and Website Services",
        }

        for line in reversed(searchable_lines):
            normalized = str(line).strip().lower()
            if not normalized:
                continue
            if self._is_generic_services_request(normalized):
                continue
            for alias, option in service_aliases.items():
                if alias in normalized:
                    return option
            for option_lower, option in lowered_options.items():
                if option_lower in normalized:
                    return option
            for option in APPOINTMENT_SERVICE_OPTIONS:
                compact = option.lower().replace(" and ", " ")
                keywords = [token for token in compact.split() if len(token) > 4]
                if keywords and sum(1 for token in keywords if token in normalized) >= 2:
                    return option
        return None

    @staticmethod
    def _is_generic_services_request(text: str) -> bool:
        generic_signals = (
            "more about the services",
            "about the services",
            "about services",
            "our services",
            "the services",
            "all services",
            "available services",
            "service list",
        )
        return any(signal in text for signal in generic_signals)

    @staticmethod
    def _extract_user_message(line: str) -> str | None:
        normalized = line.strip()
        if not normalized:
            return None
        role_match = re.match(
            r"^(user|customer|client)\s*:\s*(.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if role_match:
            return role_match.group(2).strip()
        return None
