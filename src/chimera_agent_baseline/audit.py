"""
audit.py
Immutable audit logging for every agent action.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


AUDIT_PATH = Path(os.environ.get("AUDIT_LOG_PATH", "logs/audit.jsonl"))


def log_entry(
    state: Dict[str, Any],
    agent: str,
    action: str,
    output: Dict[str, Any] | None = None,
) -> List[Dict]:
    """
    Append an audit entry to the state's audit log and to disk.
    Returns the updated audit log list.
    """
    entry = {
        "case_id": state.get("case_id", "unknown"),
        "agent": agent,
        "action": action,
        "output": output,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Append to disk (immutable, append-only)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return state.get("audit_log", []) + [entry]