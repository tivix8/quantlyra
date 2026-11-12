from copy import deepcopy

import pandas as pd

from .cleaner import clean_dataframe
from .models import CleaningResult, CleaningStep


def replay_cleaning_history(
    original_dataframe: pd.DataFrame,
    history: list[CleaningStep],
) -> CleaningResult:
    """Replays a sequence of cleaning operations."""

    if not isinstance(original_dataframe, pd.DataFrame):
        raise TypeError(
            "original_dataframe must be a pandas DataFrame"
        )

    if not isinstance(history, list):
        raise TypeError("history must be a list")

    operations = []

    for step in history:
        if not isinstance(step, CleaningStep):
            raise TypeError(
                "Each history item must be a CleaningStep instance"
            )

        operations.append(deepcopy(step.operation))

    return clean_dataframe(
        original_dataframe,
        operations,
    )


def undo_last_operation(
    cleaning_result: CleaningResult,
) -> CleaningResult:
    """Undoes the last operation by replaying previous steps."""

    if not isinstance(cleaning_result, CleaningResult):
        raise TypeError(
            "cleaning_result must be a CleaningResult instance"
        )

    if not cleaning_result.history:
        return replay_cleaning_history(
            cleaning_result.original_dataframe,
            [],
        )

    previous_history = cleaning_result.history[:-1]

    return replay_cleaning_history(
        cleaning_result.original_dataframe,
        previous_history,
    )
