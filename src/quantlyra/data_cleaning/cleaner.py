from typing import Any
from copy import deepcopy

import pandas as pd

from .models import (
    CleaningOperation,
    CleaningResult,
    CleaningStep,
)
from .operations import (
    clip_numeric_values,
    convert_dtype,
    drop_duplicates,
    drop_missing_values,
    fill_missing_values,
    normalize_text,
)


def _execute_operation(
    dataframe: pd.DataFrame,
    operation: CleaningOperation,
) -> tuple[pd.DataFrame, int]:
    """Executes a single cleaning operation."""

    parameters = dict(operation.parameters)
    operation_type = operation.operation_type
    column = operation.column

    if operation_type == "drop_missing_values":
        result = drop_missing_values(
            dataframe,
            columns=parameters.pop("columns", None),
            how=parameters.pop("how", "any"),
        )

    elif operation_type == "fill_missing_values":
        if column is None:
            raise ValueError("Column is required for fill_missing_values")

        result = fill_missing_values(
            dataframe,
            column=column,
            strategy=parameters.pop("strategy", "constant"),
            value=parameters.pop("value", None),
        )

    elif operation_type == "drop_duplicates":
        result = drop_duplicates(
            dataframe,
            subset=parameters.pop("subset", None),
            keep=parameters.pop("keep", "first"),
        )

    elif operation_type == "convert_dtype":
        if column is None:
            raise ValueError("Column is required for convert_dtype")

        if "dtype" not in parameters:
            raise ValueError("Parameter 'dtype' is required")

        result = convert_dtype(
            dataframe,
            column=column,
            dtype=parameters.pop("dtype"),
            errors=parameters.pop("errors", "raise"),
        )

    elif operation_type == "normalize_text":
        if column is None:
            raise ValueError("Column is required for normalize_text")

        result = normalize_text(
            dataframe,
            column=column,
            strip=parameters.pop("strip", True),
            lowercase=parameters.pop("lowercase", False),
            uppercase=parameters.pop("uppercase", False),
            collapse_whitespace=parameters.pop(
                "collapse_whitespace", True
            ),
        )

    elif operation_type == "clip_numeric_values":
        if column is None:
            raise ValueError(
                "Column is required for clip_numeric_values"
            )

        result = clip_numeric_values(
            dataframe,
            column=column,
            min_value=parameters.pop("min_value", None),
            max_value=parameters.pop("max_value", None),
        )

    else:
        raise ValueError(
            f"Unsupported cleaning operation: {operation_type}"
        )

    if parameters:
        raise ValueError(
            f"Unexpected parameters for {operation_type}: "
            f"{list(parameters.keys())}"
        )

    return result


def clean_dataframe(
    dataframe: pd.DataFrame,
    operations: list[CleaningOperation],
) -> CleaningResult:
    """Executes cleaning operations sequentially."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    original_dataframe = dataframe.copy(deep=True)
    current_dataframe = dataframe.copy(deep=True)

    history: list[CleaningStep] = []

    for operation in operations:
        if not isinstance(operation, CleaningOperation):
            raise TypeError(
                "Each operation must be a CleaningOperation instance"
            )

        current_dataframe, affected_rows = _execute_operation(
            current_dataframe,
            operation,
        )

        history.append(
            CleaningStep(
                operation=deepcopy(operation),
                affected_rows=affected_rows,
                description=f"Executed {operation.operation_type}",
            )
        )

    return CleaningResult(
        original_dataframe=original_dataframe,
        cleaned_dataframe=current_dataframe,
        history=history,
    )
