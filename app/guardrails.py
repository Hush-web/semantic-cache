import re
from typing import Dict, List


# Injection patterns as regex (case-insensitive matching handled below)
INJECTION_PATTERNS: List[str] = [
    # Instruction override
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+the\s+above",
    r"disregard\s+(all\s+)?(previous|prior|earlier)",
    r"forget\s+(all\s+)?(previous|prior|earlier|your)",
    r"override\s+your\s+instructions",

    # System prompt extraction
    r"(reveal|show|print|repeat|display)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"what\s+(were|are)\s+your\s+instructions",
    r"what\s+is\s+your\s+system\s+prompt",

    # Role hijacking
    r"you\s+are\s+now\s+",
    r"act\s+as\s+if\s+you\s+(have\s+no|are\s+not)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"from\s+now\s+on\s+you\s+are",
    r"new\s+persona",

    # Tool manipulation
    r"call\s+the\s+delete\s+function",
    r"execute\s+this\s+code",
    r"run\s+this\s+command",
]


SUSPICIOUS_PATTERNS: List[str] = [
    r"base64",
    r"eval\(",
    r"exec\(",
    r"system\(",
    r"<script",
    r"javascript:",
]


def _count_matches(text: str, patterns: List[str]) -> List[str]:
    """Return all patterns that match the text."""
    matches = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(pattern)
    return matches


def scan_input(text: str) -> Dict:
    """
    Scan a prompt for injection and suspicious patterns.

    Returns:
        score: 0-100 risk score
        blocked: True if score >= 75
        flagged: True if score > 0
        matches: list of matched patterns
        category: "clean" | "flagged" | "blocked"
    """
    if not text or not isinstance(text, str):
        return {
            "score": 0,
            "blocked": False,
            "flagged": False,
            "matches": [],
            "category": "clean",
        }

    injection_hits = _count_matches(text, INJECTION_PATTERNS)
    suspicious_hits = _count_matches(text, SUSPICIOUS_PATTERNS)

    # Score: 40 per injection, 10 per suspicious
    score = min(len(injection_hits) * 40 + len(suspicious_hits) * 10, 100)

    blocked = score >= 75
    flagged = score > 0

    if blocked:
        category = "blocked"
    elif flagged:
        category = "flagged"
    else:
        category = "clean"

    return {
        "score": score,
        "blocked": blocked,
        "flagged": flagged,
        "matches": injection_hits + [f"suspicious:{p}" for p in suspicious_hits],
        "category": category,
    }


def scan_messages(messages: List[Dict]) -> Dict:
    """
    Scan a full list of chat messages.
    Combines the risk from all messages.
    """
    if not messages:
        return {
            "score": 0,
            "blocked": False,
            "flagged": False,
            "matches": [],
            "category": "clean",
        }

    all_matches: List[str] = []
    max_score = 0

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            result = scan_input(content)
            all_matches.extend(result["matches"])
            max_score = max(max_score, result["score"])

    blocked = max_score >= 75
    flagged = max_score > 0

    if blocked:
        category = "blocked"
    elif flagged:
        category = "flagged"
    else:
        category = "clean"

    return {
        "score": max_score,
        "blocked": blocked,
        "flagged": flagged,
        "matches": all_matches,
        "category": category,
    }