from dataclasses import dataclass, field
from typing import Any


@dataclass
class NumericStats:
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    quantiles: dict[str, float | None] = field(default_factory=dict)


@dataclass
class CategoricalStats:
    top_values: dict[Any, int] = field(default_factory=dict)


@dataclass
class DatetimeStats:
    min_date: Any | None = None
    max_date: Any | None = None
    date_range: Any | None = None


@dataclass
class ColumnProfile:
    name: str
    dtype: str

    non_null_count: int
    missing_count: int
    missing_percentage: float

    unique_count: int
    unique_percentage: float

    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    datetime_stats: DatetimeStats | None = None


@dataclass
class TableProfile:
    table_name: str

    row_count: int
    column_count: int

    duplicate_rows: int
    memory_usage: int

    columns: list[ColumnProfile] = field(default_factory=list)


@dataclass
class SourceProfile:
    file_id: str
    name: str
    format: str
    tables: dict[str, TableProfile] = field(default_factory=dict)


@dataclass
class CollectionProfile:
    collection_id: str
    sources: dict[str, SourceProfile] = field(default_factory=dict)
