from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import pandas as pd

@dataclass
class CleaningOperation:
    operation_type: str
    operation_id: str = field(
        default_factory=lambda: str(uuid4())
    )
    column: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)



@dataclass
class CleaningStep:
    operation: CleaningOperation
    affected_rows: int
    description: str
    executed_at: datetime = field(
        default_factory=datetime.now
    )


@dataclass
class CleaningResult:
    original_dataframe: pd.DataFrame
    cleaned_dataframe: pd.DataFrame
    history: list[CleaningStep] = field(default_factory=list)

    @property
    def operations_count(self) -> int:
        return len(self.history)

    @property
    def total_affected_rows(self) -> int:
        return sum(
            step.affected_rows
            for step in self.history
        )
