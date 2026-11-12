from dataclasses import dataclass, field
from typing import Any

from .models import Severity


@dataclass
class QualityRule:
    rule_id: str
    severity: Severity = Severity.ERROR
    description: str = ""


@dataclass
class MissingValuesRule(QualityRule):
    column: str = ""
    max_missing_percentage: float = 0.0


@dataclass
class DuplicateRowsRule(QualityRule):
    columns: list[str] | None = None


@dataclass
class NumericRangeRule(QualityRule):
    column: str = ""
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class CategoricalRule(QualityRule):
    column: str = ""
    allowed_values: set[Any] = field(default_factory=set)


@dataclass
class DataTypeRule(QualityRule):
    column: str = ""
    expected_dtype: str = ""


@dataclass
class UniqueValuesRule(QualityRule):
    column: str = ""


@dataclass
class LogicalRule(QualityRule):
    columns: list[str] = field(default_factory=list)
    condition: str = ""
