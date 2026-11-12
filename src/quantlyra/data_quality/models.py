from dataclasses import dataclass, field
from enum import Enum

from typing import Any

class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class QualityIssue:
    rule_id: str
    severity: Severity
    message: str
    affected_rows: int
    column: str | None = None
    row_indices: list[Any] = field(default_factory=list)


@dataclass
class QualityReport:
    table_name: str
    row_count: int
    issues: list[QualityIssue] = field(default_factory=list)
    checked_rules: int = 0

    @property
    def error_count(self) -> int:
        return sum(
            issue.severity == Severity.ERROR
            for issue in self.issues
        )

    @property
    def warning_count(self) -> int:
        return sum(
            issue.severity == Severity.WARNING
            for issue in self.issues
        )

    @property
    def info_count(self) -> int:
        return sum(
            issue.severity == Severity.INFO
            for issue in self.issues
        )

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0
