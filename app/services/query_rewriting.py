from __future__ import annotations

import re


_SERVICE_NAMES = (
    "Credentialing and Provider Maintenance",
    "Authorizations and Benefits Verification",
    "Medical Billing and Denial Management",
    "Medical Auditing",
    "Customer Care",
    "Digital Marketing and Website Services",
    "Financial Management",
    "Communication Services",
)
_FRAGMENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^what\s+this\s+service\s+usually\s+includes\??$", re.IGNORECASE),
        "What does {service} usually include?",
    ),
    (
        re.compile(r"^what\s+does\s+this\s+service\s+usually\s+include\??$", re.IGNORECASE),
        "What does {service} usually include?",
    ),
    (
        re.compile(r"^what\s+should\s+we\s+prepare\??$", re.IGNORECASE),
        "What should we prepare before discussing {service}?",
    ),
    (
        re.compile(r"^what\s+about\s+intake\??$", re.IGNORECASE),
        "What information should we collect during intake for {service}?",
    ),
    (
        re.compile(r"^when\s+should\s+that\s+be\s+escalated\??$", re.IGNORECASE),
        "When should a conversation about {service} be escalated to a human?",
    ),
    (
        re.compile(r"^when\s+should\s+this\s+be\s+escalated\??$", re.IGNORECASE),
        "When should a conversation about {service} be escalated to a human?",
    ),
)


class DefaultRetrievalQueryRewriter:
    def rewrite(self, query: str, history: list[str]) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            return normalized_query

        service_name = self._find_recent_service_name(history=history)
        if not service_name:
            return normalized_query

        for pattern, template in _FRAGMENT_PATTERNS:
            if pattern.match(normalized_query):
                return template.format(service=service_name)

        return normalized_query

    def _find_recent_service_name(self, history: list[str]) -> str | None:
        recent_messages = reversed(history[-6:])
        for message in recent_messages:
            message_text = message.strip()
            if not message_text:
                continue
            message_lower = message_text.lower()
            for service_name in _SERVICE_NAMES:
                if service_name.lower() in message_lower:
                    return service_name
        return None
