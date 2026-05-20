"""Notification service — structured alert dispatch with DuckDB persistence."""
from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Any

from core.database import db
from core.logger import get_logger

_logger = get_logger("vigilant.notifications")

_LOG_BORDER = "=" * 60


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


def send_alert(
    message: str,
    level: AlertLevel = AlertLevel.INFO,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Dispatch a structured alert: log it visibly and persist to DuckDB.

    Returns the generated alert_id so callers can cross-reference.
    """
    metadata = metadata or {}
    alert_id = str(uuid.uuid4())

    _emit_log(message, level, metadata)
    _persist(alert_id, message, level, metadata)

    return alert_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _emit_log(message: str, level: AlertLevel, metadata: dict[str, Any]) -> None:
    meta_str = f" | metadata={metadata}" if metadata else ""
    body = f"\n{_LOG_BORDER}\n[ALERT:{level.value}] {message}{meta_str}\n{_LOG_BORDER}"

    if level == AlertLevel.CRITICAL:
        _logger.error(body)
    elif level == AlertLevel.WARNING:
        _logger.warning(body)
    else:
        _logger.info(body)


def _persist(
    alert_id: str,
    message: str,
    level: AlertLevel,
    metadata: dict[str, Any],
) -> None:
    db.execute(
        """
        INSERT INTO alerts (alert_id, level, message, metadata)
        VALUES (?, ?, ?, ?)
        """,
        [alert_id, level.value, message, json.dumps(metadata)],
    )
