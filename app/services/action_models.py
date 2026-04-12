from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_APPOINTMENT_FIELDS = ("service", "date", "time", "name", "email")
APPOINTMENT_SERVICE_OPTIONS = (
    "Credentialing and Provider Maintenance",
    "Authorizations and Benefits Verification",
    "Medical Billing and Denial Management",
    "Medical Auditing",
    "Communication Services",
    "Financial Management",
    "Digital Marketing and Website Services",
)


@dataclass(frozen=True, slots=True)
class AppointmentExtraction:
    service: str | None = None
    date: str | None = None
    time: str | None = None
    time_preference: str | None = None
    selected_date: str | None = None
    selected_time: str | None = None
    selected_service: str | None = None
    confirmation_intent: str | None = None
    name: str | None = None
    email: str | None = None

    def as_slot_updates(self) -> dict[str, str]:
        updates: dict[str, str] = {}
        for field_name in REQUIRED_APPOINTMENT_FIELDS:
            value = getattr(self, field_name)
            if value:
                updates[field_name] = value.strip()
        return updates


@dataclass(frozen=True, slots=True)
class AppointmentActionDecision:
    phase: str
    operation: str | None = None
    slot_updates: dict[str, str] | None = None
    clear_slots: list[str] | None = None
    time_preference: str | None = None
    date_confirmed: bool | None = None
    time_confirmed: bool | None = None
    awaiting_confirmation: bool | None = None

    def as_slot_updates(self) -> dict[str, str]:
        return {
            str(key): str(value).strip()
            for key, value in (self.slot_updates or {}).items()
            if str(value).strip()
        }

    def slots_to_clear(self) -> list[str]:
        return [str(field_name).strip() for field_name in (self.clear_slots or []) if str(field_name).strip()]


@dataclass(frozen=True, slots=True)
class AppointmentDateAvailabilityRequest:
    service: str
    date_preference: str | None = None


@dataclass(frozen=True, slots=True)
class AppointmentDateAvailabilityResult:
    service: str
    available_dates: list[str]
    date_preference: str | None = None


@dataclass(frozen=True, slots=True)
class AppointmentAvailabilityRequest:
    service: str
    date: str
    time_preference: str | None = None


@dataclass(frozen=True, slots=True)
class AppointmentAvailabilityResult:
    service: str
    date: str
    slots: list[str]
    time_preference: str | None = None


@dataclass(frozen=True, slots=True)
class AppointmentBookingRequest:
    service: str
    date: str
    time: str
    name: str
    email: str
    title: str | None = None


@dataclass(frozen=True, slots=True)
class AppointmentBookingResult:
    success: bool
    confirmation_id: str | None
    service: str
    date: str
    time: str
    name: str
    email: str
    message: str | None = None
    saved_booking: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AppointmentActionReplyContext:
    phase: str
    user_query: str
    conversation_history: list[str]
    current_slots: dict[str, str]
    missing_fields: list[str]
    next_required_field: str | None
    service_options: list[str]
    available_dates: list[str]
    available_slots: list[str]
    awaiting_confirmation: bool
    date_confirmed: bool
    time_confirmed: bool
    suggested_service: str | None = None
    suggested_date: str | None = None
    suggested_time: str | None = None
    booking_result: dict[str, Any] | None = None
    booking_error: str | None = None
    invalid_field: str | None = None
    validation_error: str | None = None


@dataclass(frozen=True, slots=True)
class AppointmentActionPlanningContext:
    user_query: str
    conversation_history: list[str]
    current_slots: dict[str, str]
    missing_fields: list[str]
    service_options: list[str]
    available_dates: list[str]
    available_slots: list[str]
    date_confirmed: bool
    time_confirmed: bool
    awaiting_confirmation: bool
    suggested_date: str | None = None
    suggested_time: str | None = None


def missing_appointment_fields(appointment_slots: dict[str, str]) -> list[str]:
    return [
        field_name
        for field_name in REQUIRED_APPOINTMENT_FIELDS
        if not appointment_slots.get(field_name, "").strip()
    ]
