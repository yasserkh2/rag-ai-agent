from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib import parse

_SERVER_LOCK = Lock()
_SERVER_BASE_URL: str | None = None
_BOOKINGS_LOCK = Lock()

_STORE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "booking_store.json"
)
_DEFAULT_START_DATE = "2026-04-15"
_DEFAULT_DAYS = 31


@dataclass(frozen=True, slots=True)
class SlotEntry:
    time: str
    state: str
    title: str


def _default_times() -> list[str]:
    times: list[str] = []
    hour = 9
    minute = 0
    while hour < 17 or (hour == 17 and minute == 0):
        display_hour = hour % 12 or 12
        suffix = "AM" if hour < 12 else "PM"
        times.append(f"{display_hour:02d}:{minute:02d} {suffix}")
        minute += 30
        if minute >= 60:
            minute = 0
            hour += 1
    return times


def _seed_slots() -> dict[str, dict[str, dict[str, str]]]:
    from datetime import date, timedelta

    start = date.fromisoformat(_DEFAULT_START_DATE)
    slots: dict[str, dict[str, dict[str, str]]] = {}
    for offset in range(_DEFAULT_DAYS):
        day = (start + timedelta(days=offset)).isoformat()
        slots[day] = {
            time: {"state": "free", "title": ""}
            for time in _default_times()
        }
    return slots


def _load_store() -> dict[str, object]:
    if not _STORE_PATH.exists():
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        store = {"slots": _seed_slots(), "bookings": {}}
        _STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")
        return store

    raw = _STORE_PATH.read_text(encoding="utf-8")
    try:
        store = json.loads(raw)
    except json.JSONDecodeError:
        store = {"slots": _seed_slots(), "bookings": {}}
    if not isinstance(store, dict):
        store = {"slots": _seed_slots(), "bookings": {}}
    store.setdefault("slots", _seed_slots())
    store.setdefault("bookings", {})
    return store


def _save_store(store: dict[str, object]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def ensure_mock_booking_api_server_started() -> str:
    global _SERVER_BASE_URL
    with _SERVER_LOCK:
        if _SERVER_BASE_URL is not None:
            return _SERVER_BASE_URL

        server = ThreadingHTTPServer(("127.0.0.1", 0), _BookingApiHandler)
        host, port = server.server_address
        _SERVER_BASE_URL = f"http://{host}:{port}"
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return _SERVER_BASE_URL


class _BookingApiHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed_url = parse.urlparse(self.path)
        if parsed_url.path.startswith("/bookings/"):
            confirmation_id = parsed_url.path.rsplit("/", 1)[-1].strip()
            booking = _get_saved_booking(confirmation_id)
            if booking is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "booking_not_found"})
                return

            self._write_json(
                HTTPStatus.OK,
                {
                    "success": True,
                    "confirmation_id": confirmation_id,
                    "service": str(booking["service"]),
                    "date": str(booking["date"]),
                    "time": str(booking["time"]),
                    "name": str(booking["name"]),
                    "email": str(booking["email"]),
                    "message": "Booking retrieved successfully.",
                    "saved_booking": booking,
                },
            )
            return

        if parsed_url.path == "/available-dates":
            query = parse.parse_qs(parsed_url.query)
            service = _first_value(query, "service")
            date_preference = _first_value(query, "date_preference")
            if not service:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "service_required"},
                )
                return

            self._write_json(
                HTTPStatus.OK,
                {
                    "service": service,
                    "date_preference": date_preference,
                    "available_dates": _generate_available_dates(
                        service=service,
                        date_preference=date_preference,
                    ),
                },
            )
            return

        if parsed_url.path != "/availability":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        query = parse.parse_qs(parsed_url.query)
        service = _first_value(query, "service")
        date = _first_value(query, "date")
        time_preference = _first_value(query, "time_preference")

        if not service or not date:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "service_and_date_required"},
            )
            return

        slots = _generate_available_slots(
            service=service,
            date=date,
            time_preference=time_preference,
        )
        self._write_json(
            HTTPStatus.OK,
            {
                "service": service,
                "date": date,
                "time_preference": time_preference,
                "slots": slots,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/bookings":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        payload = self._read_json_body()
        required_fields = ("service", "date", "time", "name", "email")
        missing_fields = [
            field_name
            for field_name in required_fields
            if not str(payload.get(field_name, "")).strip()
        ]
        if missing_fields:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "missing_fields", "fields": missing_fields},
            )
            return

        confirmation_id, booking_record = persist_booking(payload)
        self._write_json(
            HTTPStatus.OK,
            {
                "success": True,
                "confirmation_id": confirmation_id,
                "service": str(booking_record["service"]),
                "date": str(booking_record["date"]),
                "time": str(booking_record["time"]),
                "name": str(booking_record["name"]),
                "email": str(booking_record["email"]),
                "message": "Booking created successfully.",
                "saved_booking": booking_record,
            },
        )

    def do_DELETE(self) -> None:
        if not self.path.startswith("/bookings/"):
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        confirmation_id = self.path.rsplit("/", 1)[-1].strip()
        deleted = delete_booking(confirmation_id)
        if not deleted:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "booking_not_found"})
            return

        self._write_json(
            HTTPStatus.OK,
            {"success": True, "confirmation_id": confirmation_id, "message": "Booking deleted successfully."},
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json_body(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _generate_available_slots(
    service: str,
    date: str,
    time_preference: str | None,
) -> list[str]:
    store = _load_store()
    slots_by_date = store.get("slots", {})
    if not isinstance(slots_by_date, dict):
        return []
    day_slots = slots_by_date.get(date, {})
    if not isinstance(day_slots, dict):
        return []

    free_slots = [
        time
        for time, slot in day_slots.items()
        if isinstance(slot, dict) and slot.get("state") == "free"
    ]

    normalized_preference = (time_preference or "").strip().lower()
    if normalized_preference in {"morning", "afternoon", "evening"}:
        filtered: list[str] = []
        for slot in free_slots:
            minutes = _time_to_minutes(slot)
            if minutes is None:
                continue
            if normalized_preference == "morning" and minutes < 12 * 60:
                filtered.append(slot)
            elif normalized_preference == "afternoon" and 12 * 60 <= minutes < 17 * 60:
                filtered.append(slot)
            elif normalized_preference == "evening" and minutes >= 17 * 60:
                filtered.append(slot)
        free_slots = filtered

    return sorted(free_slots)


def _generate_available_dates(
    service: str,
    date_preference: str | None,
) -> list[str]:
    store = _load_store()
    slots_by_date = store.get("slots", {})
    if not isinstance(slots_by_date, dict):
        return []
    available_dates = []
    for day, slots in slots_by_date.items():
        if not isinstance(slots, dict):
            continue
        if any(
            isinstance(slot, dict) and slot.get("state") == "free"
            for slot in slots.values()
        ):
            available_dates.append(day)

    preferred = _format_date_label(date_preference)
    if preferred and preferred in available_dates:
        available_dates.remove(preferred)
        available_dates.insert(0, preferred)

    return available_dates


def _build_confirmation_id(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(
        "|".join(
            [
                str(payload["service"]),
                str(payload["date"]),
                str(payload["time"]),
                str(payload["name"]),
                str(payload["email"]),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return f"apt_{digest[:10]}"


def persist_booking(payload: dict[str, object]) -> tuple[str, dict[str, object]]:
    confirmation_id = _build_confirmation_id(payload)
    booking_record = {
        "confirmation_id": confirmation_id,
        "service": str(payload["service"]).strip(),
        "date": str(payload["date"]).strip(),
        "time": str(payload["time"]).strip(),
        "name": str(payload["name"]).strip(),
        "email": str(payload["email"]).strip(),
        "title": str(payload.get("title") or "").strip(),
        "status": "confirmed",
    }
    _save_booking(confirmation_id, booking_record)
    return confirmation_id, booking_record


def _save_booking(confirmation_id: str, booking_record: dict[str, object]) -> None:
    with _BOOKINGS_LOCK:
        store = _load_store()
        bookings = store.get("bookings")
        slots_by_date = store.get("slots")
        if not isinstance(bookings, dict) or not isinstance(slots_by_date, dict):
            store = {"slots": _seed_slots(), "bookings": {}}
            bookings = store["bookings"]
            slots_by_date = store["slots"]

        date = str(booking_record.get("date") or "").strip()
        time = str(booking_record.get("time") or "").strip()
        day_slots = slots_by_date.get(date)
        if isinstance(day_slots, dict) and time in day_slots:
            slot = day_slots.get(time)
            if isinstance(slot, dict) and slot.get("state") == "free":
                slot["state"] = "booked"
                slot["title"] = str(booking_record.get("title") or "")

        bookings[confirmation_id] = dict(booking_record)
        _save_store(store)


def _get_saved_booking(confirmation_id: str) -> dict[str, object] | None:
    with _BOOKINGS_LOCK:
        store = _load_store()
        bookings = store.get("bookings", {})
        if not isinstance(bookings, dict):
            return None
        booking = bookings.get(confirmation_id)
        return dict(booking) if isinstance(booking, dict) else None


def get_saved_booking(confirmation_id: str) -> dict[str, object] | None:
    return _get_saved_booking(confirmation_id)


def delete_booking(confirmation_id: str) -> bool:
    with _BOOKINGS_LOCK:
        store = _load_store()
        bookings = store.get("bookings", {})
        slots_by_date = store.get("slots", {})
        if not isinstance(bookings, dict) or not isinstance(slots_by_date, dict):
            return False
        booking = bookings.pop(confirmation_id, None)
        if not isinstance(booking, dict):
            return False
        date = str(booking.get("date") or "").strip()
        time = str(booking.get("time") or "").strip()
        day_slots = slots_by_date.get(date)
        if isinstance(day_slots, dict) and time in day_slots:
            slot = day_slots.get(time)
            if isinstance(slot, dict):
                slot["state"] = "free"
                slot["title"] = ""
        _save_store(store)
        return True


def _format_date_label(date_text: str | None) -> str | None:
    if not date_text:
        return None
    words = [word.capitalize() for word in date_text.strip().split()]
    return " ".join(words) or None


def _time_to_minutes(value: str) -> int | None:
    try:
        time_part, suffix = value.strip().split()
        hour_text, minute_text = time_part.split(":")
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError:
        return None
    suffix = suffix.upper()
    if suffix not in {"AM", "PM"}:
        return None
    if hour < 1 or hour > 12 or minute not in {0, 30}:
        return None
    if suffix == "AM":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return hour * 60 + minute


def _first_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key, [])
    if not values:
        return None
    value = values[0].strip()
    return value or None
