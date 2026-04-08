from __future__ import annotations

import unittest
from collections.abc import Callable

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
)
from app.services.action_request import AppointmentActionService


class StubAppointmentExtractor:
    def __init__(self, responses: dict[str, AppointmentExtraction] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "user_query": user_query,
                "conversation_history": list(conversation_history),
                "current_slots": dict(current_slots),
                "offered_dates": list(offered_dates or []),
                "offered_times": list(offered_times or []),
                "offered_services": list(offered_services or []),
                "awaiting_confirmation": awaiting_confirmation,
            }
        )
        return self._responses.get(user_query, AppointmentExtraction())


class FailingAppointmentExtractor:
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
        raise RuntimeError("extractor offline")


class StubBookingApiClient:
    def __init__(
        self,
        availability: AppointmentAvailabilityResult | None = None,
        date_availability: AppointmentDateAvailabilityResult | None = None,
        booking_result: AppointmentBookingResult | None = None,
        fail_booking: bool = False,
    ) -> None:
        self._date_availability = date_availability or AppointmentDateAvailabilityResult(
            service="Meeting",
            available_dates=["Tomorrow", "Next Tuesday", "Next Thursday"],
            date_preference="tomorrow",
        )
        self._availability = availability or AppointmentAvailabilityResult(
            service="Credentialing",
            date="Next Tuesday",
            slots=["01:00 PM", "02:30 PM", "04:00 PM"],
            time_preference="afternoon",
        )
        self._booking_result = booking_result or AppointmentBookingResult(
            success=True,
            confirmation_id="apt_1234567890",
            service="Credentialing",
            date="Next Tuesday",
            time="02:30 PM",
            name="Ahmed Hassan",
            email="ahmed@example.com",
            message="Booking created successfully.",
        )
        self._fail_booking = fail_booking
        self.date_availability_calls: list[AppointmentDateAvailabilityRequest] = []
        self.availability_calls: list[AppointmentAvailabilityRequest] = []
        self.booking_calls: list[AppointmentBookingRequest] = []

    def get_available_dates(
        self,
        request: AppointmentDateAvailabilityRequest,
    ) -> AppointmentDateAvailabilityResult:
        self.date_availability_calls.append(request)
        return self._date_availability

    def get_availability(
        self,
        request: AppointmentAvailabilityRequest,
    ) -> AppointmentAvailabilityResult:
        self.availability_calls.append(request)
        return self._availability

    def create_booking(
        self,
        request: AppointmentBookingRequest,
    ) -> AppointmentBookingResult:
        self.booking_calls.append(request)
        if self._fail_booking:
            raise RuntimeError("booking failed")
        return self._booking_result

    def get_booking(self, confirmation_id: str) -> AppointmentBookingResult:
        if self._booking_result.confirmation_id != confirmation_id:
            raise RuntimeError("booking not found")
        return self._booking_result


def _default_reply_factory(context: AppointmentActionReplyContext) -> str:
    if context.phase == "booking_success" and context.booking_result:
        return (
            f"Great! Your appointment for {context.booking_result['service']} on "
            f"{context.booking_result['date']} at {context.booking_result['time']} has been scheduled. "
            f"Confirmation ID: {context.booking_result['confirmation_id']}."
        )
    if context.phase == "booking_error":
        return "I could not complete the appointment booking right now."
    if context.validation_error and context.invalid_field == "email":
        return (
            f"{context.validation_error} What email address should I use for the confirmation?"
        )
    if context.validation_error and context.invalid_field == "name":
        return f"{context.validation_error} What name should I use for the booking?"
    if context.validation_error and context.invalid_field == "service":
        return "Please choose one of the listed services to continue."
    if context.validation_error and context.invalid_field == "date":
        return f"{context.validation_error} Please choose one of these dates: {', '.join(context.available_dates)}."
    if context.validation_error and context.invalid_field == "time":
        return f"{context.validation_error} Please choose one of these times: {', '.join(context.available_slots)}."
    if context.phase == "choose_date":
        date_text = ", ".join(context.available_dates)
        if context.suggested_date:
            return (
                f"I checked the available dates. Instead of booking {context.suggested_date} directly, "
                f"please choose one of these available dates: {date_text}."
            )
        return f"I found available dates: {date_text}. Which date would you like?"
    if context.phase == "choose_time":
        slot_text = ", ".join(context.available_slots)
        if context.suggested_time:
            return (
                f"I checked the available times. Instead of booking {context.suggested_time} directly, "
                f"please choose one of these available times: {slot_text}."
            )
        return f"I found available times: {slot_text}. Which time would you like?"
    if context.phase == "confirm":
        slots = context.current_slots
        return (
            "Please confirm your appointment details: "
            f"{slots['service']} on {slots['date']} at {slots['time']} for "
            f"{slots['name']} ({slots['email']})."
        )
    if context.phase == "awaiting_confirmation":
        return "Please confirm the appointment or tell me what to change."
    if context.phase == "change_request":
        return "Tell me which detail you want to change."

    if context.next_required_field == "service":
        service_text = "; ".join(APPOINTMENT_SERVICE_OPTIONS)
        if "date" in context.missing_fields:
            return (
                "For appointment scheduling, the available services are: "
                f"{service_text}. Which service would you like to book, and what date works for you?"
            )
        return (
            "For appointment scheduling, the available services are: "
            f"{service_text}. Which service would you like to book?"
        )
    if context.next_required_field == "date":
        return "What date would you like for the appointment?"
    if context.next_required_field == "time" and context.available_slots:
        return (
            "Please choose one of these available times: "
            f"{', '.join(context.available_slots)}."
        )
    if context.next_required_field == "time":
        return "What time would work best for the appointment?"
    if context.next_required_field == "name":
        return "What name should I use for the booking?"
    if context.next_required_field == "email":
        return "What email address should I use for the confirmation?"
    return "Please share the remaining appointment details."


class StubActionReplyGenerator:
    def __init__(
        self,
        reply: str | None = None,
        reply_factory: Callable[[AppointmentActionReplyContext], str] | None = None,
    ) -> None:
        self._reply = reply
        self._reply_factory = reply_factory or _default_reply_factory
        self.calls: list[AppointmentActionReplyContext] = []

    def generate_reply(self, context: AppointmentActionReplyContext) -> str:
        self.calls.append(context)
        if self._reply is not None:
            return self._reply
        return self._reply_factory(context)


class AppointmentActionServiceTests(unittest.TestCase):
    def test_initial_request_asks_for_service_and_date(self) -> None:
        service = AppointmentActionService(
            extractor=StubAppointmentExtractor(),
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn({"user_query": "I'd like to schedule an appointment."})

        self.assertEqual(result["active_action"], "appointment_scheduling")
        self.assertEqual(result["turn_outcome"], "needs_input")
        self.assertEqual(result["appointment_slots"], {})
        self.assertEqual(result["missing_slots"], ["service", "date", "time", "name", "email"])
        self.assertIn("available services are", result["final_response"])
        self.assertIn("Which service would you like to book", result["final_response"])

    def test_prompt_driven_reply_generator_can_override_default_reply(self) -> None:
        reply_generator = StubActionReplyGenerator("Generated action-agent reply.")
        service = AppointmentActionService(
            extractor=StubAppointmentExtractor(),
            booking_api_client=StubBookingApiClient(),
            response_generator=reply_generator,
        )

        result = service.handle_turn({"user_query": "I'd like to schedule an appointment."})

        self.assertEqual(result["final_response"], "Generated action-agent reply.")
        self.assertEqual(reply_generator.calls[0].phase, "collecting")

    def test_selected_service_fetches_and_shows_available_dates(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "Digital Marketing": AppointmentExtraction(
                    selected_service="Digital Marketing and Website Services",
                )
            }
        )
        booking_client = StubBookingApiClient(
            date_availability=AppointmentDateAvailabilityResult(
                service="Digital Marketing and Website Services",
                available_dates=["Next Thursday", "Next Friday", "Next Monday"],
                date_preference=None,
            )
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn({"user_query": "Digital Marketing"})

        self.assertEqual(
            result["appointment_slots"],
            {"service": "Digital Marketing and Website Services"},
        )
        self.assertEqual(
            result["available_dates"],
            ["Next Thursday", "Next Friday", "Next Monday"],
        )
        self.assertIn("available dates", result["final_response"])
        self.assertEqual(len(booking_client.date_availability_calls), 1)

    def test_date_is_validated_and_available_dates_are_shown_first(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "meeting tomorrow": AppointmentExtraction(
                    service="Meeting",
                    date="tomorrow",
                )
            }
        )
        booking_client = StubBookingApiClient()
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn({"user_query": "meeting tomorrow"})

        self.assertEqual(result["appointment_slots"], {"service": "Meeting"})
        self.assertEqual(result["available_dates"], ["Tomorrow", "Next Tuesday", "Next Thursday"])
        self.assertFalse(result["date_confirmed"])
        self.assertEqual(result["available_slots"], [])
        self.assertIn("available dates", result["final_response"])
        self.assertEqual(len(booking_client.date_availability_calls), 1)

    def test_weekday_preference_shows_available_days_instead_of_asking_for_exact_date(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "can it be thursday": AppointmentExtraction(
                    date="thursday",
                )
            }
        )
        booking_client = StubBookingApiClient(
            date_availability=AppointmentDateAvailabilityResult(
                service="Digital Marketing and Website Services",
                available_dates=["Next Thursday", "Next Friday", "Next Monday"],
                date_preference="thursday",
            )
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "can it be thursday",
                "appointment_slots": {
                    "service": "Digital Marketing and Website Services",
                },
            }
        )

        self.assertEqual(
            result["appointment_slots"],
            {"service": "Digital Marketing and Website Services"},
        )
        self.assertEqual(
            result["available_dates"],
            ["Next Thursday", "Next Friday", "Next Monday"],
        )
        self.assertFalse(result["date_confirmed"])
        self.assertIn("available dates", result["final_response"])
        self.assertEqual(
            booking_client.date_availability_calls[0].date_preference,
            "thursday",
        )

    def test_user_can_select_offered_date_before_times_are_shown(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "Tomorrow": AppointmentExtraction(selected_date="Tomorrow"),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "Tomorrow",
                "appointment_slots": {"service": "Meeting"},
                "available_dates": ["Tomorrow", "Next Tuesday", "Next Thursday"],
            }
        )

        self.assertEqual(
            result["appointment_slots"],
            {
                "service": "Meeting",
                "date": "Tomorrow",
            },
        )
        self.assertTrue(result["date_confirmed"])
        self.assertEqual(result["available_dates"], [])
        self.assertEqual(result["available_slots"], ["01:00 PM", "02:30 PM", "04:00 PM"])
        self.assertIn("I found available times", result["final_response"])

    def test_llm_handles_typo_in_offered_date(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "tommorow": AppointmentExtraction(selected_date="Tomorrow"),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "tommorow",
                "appointment_slots": {"service": "Meeting"},
                "available_dates": ["Tomorrow", "Next Tuesday", "Next Thursday"],
            }
        )

        self.assertEqual(result["appointment_slots"]["date"], "Tomorrow")
        self.assertTrue(result["date_confirmed"])
        self.assertIn("I found available times", result["final_response"])

    def test_llm_selected_date_drives_follow_up_choice(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "the first one please": AppointmentExtraction(selected_date="Tomorrow"),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "the first one please",
                "appointment_slots": {"service": "Meeting"},
                "available_dates": ["Tomorrow", "Next Tuesday", "Next Thursday"],
            }
        )

        self.assertEqual(result["appointment_slots"]["date"], "Tomorrow")
        self.assertTrue(result["date_confirmed"])
        self.assertEqual(result["available_slots"], ["01:00 PM", "02:30 PM", "04:00 PM"])

    def test_llm_confirmation_intent_can_confirm_booking(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "looks good": AppointmentExtraction(confirmation_intent="confirm"),
            }
        )
        booking_client = StubBookingApiClient()
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "looks good",
                "appointment_slots": {
                    "service": "Credentialing",
                    "date": "Next Tuesday",
                    "time": "02:30 PM",
                    "name": "Ahmed Hassan",
                    "email": "ahmed@example.com",
                },
                "date_confirmed": True,
                "time_confirmed": True,
                "awaiting_confirmation": True,
            }
        )

        self.assertEqual(result["booking_confirmation_id"], "apt_1234567890")
        self.assertEqual(result["turn_outcome"], "resolved")
        self.assertEqual(len(booking_client.booking_calls), 1)

    def test_service_date_and_vague_time_offer_available_dates_first(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "Credentialing next Tuesday afternoon": AppointmentExtraction(
                    service="Credentialing",
                    date="Next Tuesday",
                    time_preference="afternoon",
                )
            }
        )
        booking_client = StubBookingApiClient()
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn({"user_query": "Credentialing next Tuesday afternoon"})

        self.assertEqual(result["appointment_slots"], {"service": "Credentialing"})
        self.assertEqual(result["available_dates"], ["Tomorrow", "Next Tuesday", "Next Thursday"])
        self.assertFalse(result["date_confirmed"])
        self.assertEqual(result["available_slots"], [])
        self.assertEqual(result["missing_slots"], ["date", "time", "name", "email"])
        self.assertIn("available dates", result["final_response"])
        self.assertEqual(len(booking_client.date_availability_calls), 1)

    def test_user_can_select_offered_time_slot(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "02:30 PM": AppointmentExtraction(selected_time="02:30 PM"),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "02:30 PM",
                "appointment_slots": {"service": "Credentialing", "date": "Next Tuesday"},
                "date_confirmed": True,
                "available_slots": ["01:00 PM", "02:30 PM", "04:00 PM"],
            }
        )

        self.assertEqual(
            result["appointment_slots"],
            {
                "service": "Credentialing",
                "date": "Next Tuesday",
                "time": "02:30 PM",
            },
        )
        self.assertEqual(result["missing_slots"], ["name", "email"])
        self.assertEqual(result["available_slots"], [])
        self.assertTrue(result["time_confirmed"])
        self.assertIn("What name should I use for the booking?", result["final_response"])

    def test_collects_remaining_fields_and_moves_to_confirmation(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "My name is Ahmed Hassan and my email is ahmed@example.com": AppointmentExtraction(
                    name="Ahmed Hassan",
                    email="ahmed@example.com",
                )
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "My name is Ahmed Hassan and my email is ahmed@example.com",
                "appointment_slots": {
                    "service": "Credentialing",
                    "date": "Next Tuesday",
                    "time": "02:30 PM",
                },
                "date_confirmed": True,
                "time_confirmed": True,
            }
        )

        self.assertTrue(result["awaiting_confirmation"])
        self.assertEqual(result["missing_slots"], [])
        self.assertIn("Please confirm your appointment details", result["final_response"])
        self.assertIn("Ahmed Hassan", result["final_response"])

    def test_confirmation_yes_requires_llm_confirmation_intent(self) -> None:
        booking_client = StubBookingApiClient()
        service = AppointmentActionService(
            extractor=StubAppointmentExtractor(),
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "yes",
                "appointment_slots": {
                    "service": "Credentialing",
                    "date": "Next Tuesday",
                    "time": "02:30 PM",
                    "name": "Ahmed Hassan",
                    "email": "ahmed@example.com",
                },
                "date_confirmed": True,
                "time_confirmed": True,
                "awaiting_confirmation": True,
            }
        )

        self.assertTrue(result["awaiting_confirmation"])
        self.assertEqual(len(booking_client.booking_calls), 0)
        self.assertIn("confirm the appointment", result["final_response"])

    def test_booking_result_contains_saved_booking_details(self) -> None:
        booking_client = StubBookingApiClient(
            booking_result=AppointmentBookingResult(
                success=True,
                confirmation_id="apt_1234567890",
                service="Credentialing",
                date="Next Tuesday",
                time="02:30 PM",
                name="Ahmed Hassan",
                email="ahmed@example.com",
                message="Booking created successfully.",
                saved_booking={
                    "confirmation_id": "apt_1234567890",
                    "service": "Credentialing",
                    "date": "Next Tuesday",
                    "time": "02:30 PM",
                    "name": "Ahmed Hassan",
                    "email": "ahmed@example.com",
                    "status": "confirmed",
                },
            )
        )
        extractor = StubAppointmentExtractor(
            {
                "confirm it": AppointmentExtraction(confirmation_intent="confirm"),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "confirm it",
                "appointment_slots": {
                    "service": "Credentialing",
                    "date": "Next Tuesday",
                    "time": "02:30 PM",
                    "name": "Ahmed Hassan",
                    "email": "ahmed@example.com",
                },
                "date_confirmed": True,
                "time_confirmed": True,
                "awaiting_confirmation": True,
            }
        )

        self.assertEqual(result["booking_result"]["confirmation_id"], "apt_1234567890")
        self.assertEqual(result["booking_result"]["service"], "Credentialing")

    def test_affirmative_confirmation_books_when_llm_marks_confirmation_intent(self) -> None:
        booking_client = StubBookingApiClient()
        extractor = StubAppointmentExtractor(
            {
                "yes confirmed": AppointmentExtraction(confirmation_intent="confirm"),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "yes confirmed",
                "appointment_slots": {
                    "service": "Digital Marketing and Website Services",
                    "date": "Next Monday",
                    "time": "10:30 AM",
                    "name": "Yasser Khira",
                    "email": "yasserkhira64@gmail.com",
                },
                "date_confirmed": True,
                "time_confirmed": True,
                "awaiting_confirmation": True,
            }
        )

        self.assertIsNone(result["active_action"])
        self.assertEqual(result["booking_confirmation_id"], "apt_1234567890")
        self.assertEqual(len(booking_client.booking_calls), 1)

    def test_sure_confirmation_books_without_reopening_slots(self) -> None:
        booking_client = StubBookingApiClient()
        extractor = StubAppointmentExtractor(
            {
                "sure": AppointmentExtraction(
                    confirmation_intent="confirm",
                    selected_date="Next Thursday",
                ),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "sure",
                "appointment_slots": {
                    "service": "Digital Marketing and Website Services",
                    "date": "Next Thursday",
                    "time": "09:00 AM",
                    "name": "Yasser Khira",
                    "email": "yasserkhira@gmail.com",
                },
                "date_confirmed": True,
                "time_confirmed": True,
                "awaiting_confirmation": True,
                "available_dates": ["Next Monday", "Next Tuesday", "Next Thursday"],
            }
        )

        self.assertIsNone(result["active_action"])
        self.assertEqual(result["booking_confirmation_id"], "apt_1234567890")
        self.assertEqual(len(booking_client.booking_calls), 1)

    def test_confirmation_no_keeps_flow_open_for_changes(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "no": AppointmentExtraction(confirmation_intent="change"),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "no",
                "appointment_slots": {
                    "service": "Credentialing",
                    "date": "Next Tuesday",
                    "time": "02:30 PM",
                    "name": "Ahmed Hassan",
                    "email": "ahmed@example.com",
                },
                "date_confirmed": True,
                "time_confirmed": True,
                "awaiting_confirmation": True,
            }
        )

        self.assertFalse(result["awaiting_confirmation"])
        self.assertEqual(result["appointment_slots"]["time"], "02:30 PM")
        self.assertIn("detail you want to change", result["final_response"])

    def test_available_services_question_stays_inside_action_flow(self) -> None:
        service = AppointmentActionService(
            extractor=StubAppointmentExtractor(),
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "what are the available services",
                "active_action": "appointment_scheduling",
            }
        )

        self.assertEqual(result["active_action"], "appointment_scheduling")
        self.assertIn("available services are", result["final_response"])
        self.assertIn("Credentialing and Provider Maintenance", result["final_response"])
        self.assertIn("Which service would you like to book", result["final_response"])

    def test_booking_failure_returns_llm_generated_error_message(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "confirm it": AppointmentExtraction(confirmation_intent="confirm"),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=StubBookingApiClient(fail_booking=True),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "confirm it",
                "appointment_slots": {
                    "service": "Credentialing",
                    "date": "Next Tuesday",
                    "time": "02:30 PM",
                    "name": "Ahmed Hassan",
                    "email": "ahmed@example.com",
                },
                "date_confirmed": True,
                "time_confirmed": True,
                "awaiting_confirmation": True,
            }
        )

        self.assertEqual(result["booking_error"], "booking_request_failed")
        self.assertEqual(result["turn_outcome"], "unresolved")
        self.assertEqual(result["turn_failure_reason"], "booking_request_failed")
        self.assertIn("transfer this appointment request", result["escalation_reason"])
        self.assertIn("could not complete the appointment booking", result["final_response"])
        self.assertEqual(result["appointment_slots"]["service"], "Credentialing")

    def test_specific_time_is_validated_against_api_slots_before_confirmation(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "2:00 PM": AppointmentExtraction(time="2:00 PM"),
            }
        )
        booking_client = StubBookingApiClient()
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=booking_client,
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "2:00 PM",
                "appointment_slots": {
                    "service": "Credentialing",
                    "date": "Next Tuesday",
                },
                "date_confirmed": True,
            }
        )

        self.assertEqual(result["appointment_slots"], {"service": "Credentialing", "date": "Next Tuesday"})
        self.assertEqual(result["available_slots"], ["01:00 PM", "02:30 PM", "04:00 PM"])
        self.assertFalse(result["time_confirmed"])
        self.assertIn("Instead of booking 2:00 PM directly", result["final_response"])
        self.assertEqual(len(booking_client.availability_calls), 1)

    def test_extractor_failure_is_returned_directly_instead_of_fallback_prompt(self) -> None:
        service = AppointmentActionService(
            extractor=FailingAppointmentExtractor(),
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn({"user_query": "today"})

        self.assertEqual(result["booking_error"], "action_extraction_failed")
        self.assertEqual(result["turn_outcome"], "unresolved")
        self.assertEqual(result["turn_failure_reason"], "action_extraction_failed")
        self.assertIn("transfer this appointment request", result["escalation_reason"])
        self.assertEqual(
            result["final_response"],
            "Action LLM failed to extract appointment details: extractor offline",
        )

    def test_invalid_email_does_not_advance_to_confirmation(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "gmail.com": AppointmentExtraction(email="gmail.com"),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "gmail.com",
                "appointment_slots": {
                    "service": "Digital Marketing and Website Services",
                    "date": "Next Monday",
                    "time": "09:00 AM",
                    "name": "Yasser",
                },
                "date_confirmed": True,
                "time_confirmed": True,
            }
        )

        self.assertEqual(
            result["appointment_slots"],
            {
                "service": "Digital Marketing and Website Services",
                "date": "Next Monday",
                "time": "09:00 AM",
                "name": "Yasser",
            },
        )
        self.assertFalse(result["awaiting_confirmation"])
        self.assertEqual(result["missing_slots"], ["email"])
        self.assertIn("complete email address", result["final_response"])
        self.assertIn("What email address should I use", result["final_response"])

    def test_service_does_not_confirm_until_date_and_time_are_confirmed(self) -> None:
        service = AppointmentActionService(
            extractor=StubAppointmentExtractor(),
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "ready",
                "appointment_slots": {
                    "service": "Digital Marketing and Website Services",
                    "date": "Next Monday",
                    "time": "09:00 AM",
                    "name": "Yasser",
                    "email": "yasser@example.com",
                },
                "date_confirmed": False,
                "time_confirmed": True,
            }
        )

        self.assertFalse(result["awaiting_confirmation"])
        self.assertIn("please choose one of these available dates", result["final_response"])

    def test_name_turn_ignores_stale_date_extraction_and_moves_to_email(self) -> None:
        extractor = StubAppointmentExtractor(
            {
                "yasser": AppointmentExtraction(
                    name="yasser",
                    selected_date="Next Thursday",
                ),
            }
        )
        service = AppointmentActionService(
            extractor=extractor,
            booking_api_client=StubBookingApiClient(),
            response_generator=StubActionReplyGenerator(),
        )

        result = service.handle_turn(
            {
                "user_query": "yasser",
                "appointment_slots": {
                    "service": "Digital Marketing and Website Services",
                    "date": "Next Thursday",
                    "time": "04:00 PM",
                },
                "available_dates": [],
                "available_slots": [],
                "date_confirmed": True,
                "time_confirmed": True,
            }
        )

        self.assertEqual(
            result["appointment_slots"],
            {
                "service": "Digital Marketing and Website Services",
                "date": "Next Thursday",
                "time": "04:00 PM",
                "name": "Yasser",
            },
        )
        self.assertFalse(result["awaiting_confirmation"])
        self.assertEqual(result["missing_slots"], ["email"])
        self.assertIn("What email address should I use", result["final_response"])


if __name__ == "__main__":
    unittest.main()
