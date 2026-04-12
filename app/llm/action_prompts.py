from __future__ import annotations

import json

from app.services.action_models import AppointmentActionReplyContext

DEFAULT_ACTION_AGENT_SYSTEM_PROMPT = (
    "You are COB Company's appointment scheduling assistant. "
    "Own the next step in the appointment flow from the provided state. "
    "Use recent conversation to sound contextual and human, not robotic. "
    "Use the known slot values exactly as provided. "
    "If service options, available dates, or available times are provided, guide the user to choose from them and do not invent new options. "
    "When asking for service, briefly acknowledge what the user already said and ask which service best matches their goal. "
    "If suggested_service is provided, explicitly mention it and ask whether to continue with it or choose another option. "
    "If suggested_service is null, do not assume one specific service. Ask a neutral service-selection question. "
    "Ask for only one kind of missing information at a time. "
    "Use next_required_field as the single source of truth for what to ask next. "
    "If validation_error is present, briefly explain it and re-ask only for invalid_field. "
    "If awaiting_confirmation is true and next_required_field is null, ask clearly for confirmation or a change request. "
    "Never ask for confirmation unless next_required_field is null. "
    "Never ask for fields that are already collected and confirmed. "
    "If booking_result is present, confirm success clearly and include the confirmation ID. "
    "If booking_error is present, explain the failure plainly instead of pretending the flow succeeded. "
    "Be concise, natural, and helpful. "
    "Do not mention internal state names or JSON."
)


def build_action_agent_user_prompt(context: AppointmentActionReplyContext) -> str:
    history_block = "\n".join(context.conversation_history[-6:]) or "[no prior conversation]"
    current_slots_json = json.dumps(context.current_slots, sort_keys=True)
    missing_fields_json = json.dumps(context.missing_fields)
    service_options_json = json.dumps(context.service_options)
    available_dates_json = json.dumps(context.available_dates)
    available_slots_json = json.dumps(context.available_slots)
    booking_result_json = json.dumps(context.booking_result or {}, sort_keys=True)
    suggested_service_json = json.dumps(context.suggested_service)
    suggested_date_json = json.dumps(context.suggested_date)
    suggested_time_json = json.dumps(context.suggested_time)

    return (
        "Generate the next assistant reply for the appointment action flow.\n\n"
        f"Phase: {context.phase}\n"
        f"Latest user message: {context.user_query}\n"
        f"Recent conversation:\n{history_block}\n\n"
        f"Current slots: {current_slots_json}\n"
        f"Missing fields: {missing_fields_json}\n"
        f"Next required field: {json.dumps(context.next_required_field)}\n"
        f"Service options: {service_options_json}\n"
        f"Available dates: {available_dates_json}\n"
        f"Available times: {available_slots_json}\n"
        f"Suggested service from history: {suggested_service_json}\n"
        f"Awaiting confirmation: {json.dumps(context.awaiting_confirmation)}\n"
        f"Date confirmed: {json.dumps(context.date_confirmed)}\n"
        f"Time confirmed: {json.dumps(context.time_confirmed)}\n"
        f"Suggested date awaiting validation: {suggested_date_json}\n"
        f"Suggested time awaiting validation: {suggested_time_json}\n"
        f"Booking result: {booking_result_json}\n"
        f"Booking error: {json.dumps(context.booking_error)}\n"
        f"Invalid field to re-collect: {json.dumps(context.invalid_field)}\n"
        f"Validation error: {json.dumps(context.validation_error)}\n\n"
        "Write the exact next assistant message for the user."
    )
