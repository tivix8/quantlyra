from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
import math

@dataclass(frozen=True)
class TableRef:
    source_id: str
    table_name: str
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("Source ID cannot be empty")

        if not self.table_name.strip():
            raise ValueError("Table name cannot be empty")


@dataclass(frozen=True)
class ColumnPair:
    left_column: str
    right_column: str


@dataclass
class ColumnMatch:
    left_column: str
    right_column: str
    name_similarity: float
    type_compatible: bool
    score: float
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.left_column.strip() or not self.right_column.strip():
            raise ValueError("Column names cannot be empty")

        if not 0.0 <= self.name_similarity <= 100.0:
            raise ValueError("Name similarity must be between 0 and 100")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")


class RelationshipStatus(str, Enum):
    SUGGESTED = "suggested"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class RelationshipCardinality(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class RelationshipOrigin(str, Enum):
    AUTOMATIC = "automatic"
    USER = "user"
    AI_AGENT = "ai_agent"
    COMPANY_DOCUMENT = "company_document"


@dataclass
class TableRelationship:
    left_table: TableRef
    right_table: TableRef
    key_pairs: list[ColumnPair]

    relationship_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    status: RelationshipStatus = RelationshipStatus.SUGGESTED
    cardinality: RelationshipCardinality | None = None
    origin: RelationshipOrigin = RelationshipOrigin.AUTOMATIC

    score: float | None = None
    evidence: list[str] = field(default_factory=list)

    key_schema_fingerprint: str | None = None
    is_stale: bool = False

    def __post_init__(self) -> None:
        self._validate_tables()
        self._validate_key_pairs()
        self._validate_score()

    def _validate_tables(self) -> None:
        same_source = (
            self.left_table.source_id == self.right_table.source_id
        )
        same_table = (
            self.left_table.table_name == self.right_table.table_name
        )

        if same_source and same_table:
            raise ValueError(
                "Relationship cannot reference the same table"
            )

    def _validate_key_pairs(self) -> None:
        if not self.key_pairs:
            raise ValueError("At least one key pair is required")

        seen_pairs = set()

        for pair in self.key_pairs:
            left_column = pair.left_column.strip()
            right_column = pair.right_column.strip()

            if not left_column or not right_column:
                raise ValueError("Column names cannot be empty")

            normalized_pair = (
                left_column,
                right_column,
            )

            if normalized_pair in seen_pairs:
                raise ValueError("Duplicate key pairs are not allowed")

            seen_pairs.add(normalized_pair)

    def _validate_score(self) -> None:
        if self.score is not None:
            if not 0.0 <= self.score <= 1.0:
                raise ValueError(
                    "Score must be between 0.0 and 1.0"
                )


@dataclass
class RelationshipDetection:
    left_column: str
    right_column: str

    overlap_count: int
    left_coverage: float
    right_coverage: float

    left_unique: bool
    right_unique: bool

    left_null_ratio: float
    right_null_ratio: float

    cardinality: RelationshipCardinality | None
    score: float
    evidence: list[str] = field(default_factory=list)


    def __post_init__(self) -> None:
        if not self.left_column.strip() or not self.right_column.strip():
            raise ValueError("Column names cannot be empty")

        if self.overlap_count < 0:
            raise ValueError("Overlap count cannot be negative")

        metrics = {
            "left_coverage": self.left_coverage,
            "right_coverage": self.right_coverage,
            "left_null_ratio": self.left_null_ratio,
            "right_null_ratio": self.right_null_ratio,
            "score": self.score,
        }

        for name, value in metrics.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

@dataclass
class RelationshipEvent:
    relationship_id: str
    event_type: str
    previous_status: RelationshipStatus | None
    new_status: RelationshipStatus | None
    origin: RelationshipOrigin
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_id: str = field(default_factory=lambda: str(uuid4()))
    details: str | None = None