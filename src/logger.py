from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd


LOG_COLUMNS = [
    "timestamp",
    "source",
    "source_type",
    "alert_count",
    "total_detections",
    "risk_summary",
    "alert_modes",
    "message",
    "risk_details_json",
]


@dataclass
class AlertLogger:
    log_path: Path

    def write_event(
        self,
        source_name: str,
        alert_count: int,
        total_detections: int,
        message: str,
        source_type: str = "image",
        risk_summary: str = "",
        risk_details: list[dict[str, Any]] | None = None,
        alert_modes: list[str] | None = None,
    ) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_name,
            "source_type": source_type,
            "alert_count": alert_count,
            "total_detections": total_detections,
            "risk_summary": risk_summary,
            "alert_modes": ", ".join(alert_modes or ["visual", "text"]),
            "message": message,
            "risk_details_json": json.dumps(risk_details or [], ensure_ascii=False),
        }
        self._ensure_current_log_format()
        frame = pd.DataFrame([row], columns=LOG_COLUMNS)
        frame.to_csv(
            self.log_path,
            mode="a",
            header=not self.log_path.exists(),
            index=False,
            encoding="utf-8-sig",
        )

    def read_events(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame()
        self._ensure_current_log_format()
        if not self.log_path.exists():
            return pd.DataFrame(columns=LOG_COLUMNS)
        try:
            return pd.read_csv(self.log_path, encoding="utf-8-sig")
        except pd.errors.ParserError:
            self._backup_legacy_log()
            return pd.DataFrame(columns=LOG_COLUMNS)

    def _ensure_current_log_format(self) -> None:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return

        first_line = self.log_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
        expected_header = ",".join(LOG_COLUMNS)
        if first_line != expected_header:
            self._backup_legacy_log()

    def _backup_legacy_log(self) -> None:
        if not self.log_path.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.log_path.with_name(f"{self.log_path.stem}_legacy_backup_{timestamp}.csv")
        shutil.move(str(self.log_path), str(backup_path))
