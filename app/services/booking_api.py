from __future__ import annotations

import json
from urllib import parse, request

from app.mock_api.booking_api import ensure_mock_booking_api_server_started
from app.services.action_request import AppointmentBookingApiClient
from app.services.action_models import (
    AppointmentAvailabilityRequest,
    AppointmentAvailabilityResult,
    AppointmentBookingRequest,
    AppointmentBookingResult,
    AppointmentDateAvailabilityRequest,
    AppointmentDateAvailabilityResult,
)


class LocalMockBookingApiClient(AppointmentBookingApiClient):
    def __init__(self, base_url: str | None = None, timeout_seconds: int = 10) -> None:
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    def get_available_dates(
        self,
        request_data: AppointmentDateAvailabilityRequest,
    ) -> AppointmentDateAvailabilityResult:
        base_url = self._ensure_base_url()
        query = parse.urlencode(
            {
                "service": request_data.service,
                "date_preference": request_data.date_preference or "",
            }
        )
        with request.urlopen(
            f"{base_url}/available-dates?{query}",
            timeout=self._timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return AppointmentDateAvailabilityResult(
            service=str(payload["service"]),
            available_dates=[
                str(value) for value in payload.get("available_dates", [])
            ],
            date_preference=str(payload.get("date_preference") or "") or None,
        )

    def get_availability(
        self,
        request_data: AppointmentAvailabilityRequest,
    ) -> AppointmentAvailabilityResult:
        base_url = self._ensure_base_url()
        query = parse.urlencode(
            {
                "service": request_data.service,
                "date": request_data.date,
                "time_preference": request_data.time_preference or "",
            }
        )
        with request.urlopen(
            f"{base_url}/availability?{query}",
            timeout=self._timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return AppointmentAvailabilityResult(
            service=str(payload["service"]),
            date=str(payload["date"]),
            slots=[str(value) for value in payload.get("slots", [])],
            time_preference=str(payload.get("time_preference") or "") or None,
        )

    def create_booking(
        self,
        request_data: AppointmentBookingRequest,
    ) -> AppointmentBookingResult:
        base_url = self._ensure_base_url()
        http_request = request.Request(
            f"{base_url}/bookings",
            data=json.dumps(
                {
                    "service": request_data.service,
                    "date": request_data.date,
                    "time": request_data.time,
                    "name": request_data.name,
                    "email": request_data.email,
                    "title": request_data.title or "",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return AppointmentBookingResult(
            success=bool(payload["success"]),
            confirmation_id=str(payload.get("confirmation_id") or ""),
            service=str(payload["service"]),
            date=str(payload["date"]),
            time=str(payload["time"]),
            name=str(payload["name"]),
            email=str(payload["email"]),
            message=str(payload.get("message") or "") or None,
            saved_booking=payload.get("saved_booking"),
        )

    def get_booking(self, confirmation_id: str) -> AppointmentBookingResult:
        base_url = self._ensure_base_url()
        with request.urlopen(
            f"{base_url}/bookings/{parse.quote(confirmation_id)}",
            timeout=self._timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return AppointmentBookingResult(
            success=bool(payload["success"]),
            confirmation_id=str(payload.get("confirmation_id") or ""),
            service=str(payload["service"]),
            date=str(payload["date"]),
            time=str(payload["time"]),
            name=str(payload["name"]),
            email=str(payload["email"]),
            message=str(payload.get("message") or "") or None,
            saved_booking=payload.get("saved_booking"),
        )

    def _ensure_base_url(self) -> str:
        if self._base_url is None:
            self._base_url = ensure_mock_booking_api_server_started()
        return self._base_url
