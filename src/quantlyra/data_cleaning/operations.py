from typing import Any
import pandas as pd
from pandas.api import types as ptypes


def _validate_column(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found")


def drop_missing_values(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    how: str = "any",
) -> tuple[pd.DataFrame, int]:
    """Removes rows with missing values."""
    if how not in {"any", "all"}:
        raise ValueError("how must be 'any' or 'all'")
    if columns is not None:
        for column in columns:
            _validate_column(dataframe, column)
    result = dataframe.copy(deep=True)
    original_rows = len(result)
    result = result.dropna(subset=columns, how=how)
    affected_rows = original_rows - len(result)
    return result, affected_rows


def fill_missing_values(
    dataframe: pd.DataFrame,
    column: str,
    strategy: str = "constant",
    value: Any = None,
) -> tuple[pd.DataFrame, int]:
    """Fills missing values using the selected strategy."""
    _validate_column(dataframe, column)
    if strategy not in {"mean", "median", "mode", "constant"}:
        raise ValueError(
            "Unsupported strategy. Choose mean, median, mode or constant"
        )
    result = dataframe.copy(deep=True)
    series = result[column]
    missing_mask = series.isna()
    affected_rows = int(missing_mask.sum())
    if affected_rows == 0:
        return result, 0
    if strategy == "constant":
        if value is None:
            raise ValueError(
                "A value must be provided for constant strategy"
            )
        fill_value = value
    elif strategy == "mean":
        if not ptypes.is_numeric_dtype(series.dtype) or ptypes.is_bool_dtype(series.dtype):
            raise TypeError("Mean strategy requires a numeric column")
        fill_value = series.mean()
    elif strategy == "median":
        if not ptypes.is_numeric_dtype(series.dtype) or ptypes.is_bool_dtype(series.dtype):
            raise TypeError("Median strategy requires a numeric column")
        fill_value = series.median()
    else:
        modes = series.mode(dropna=True)
        if modes.empty:
            raise ValueError(
                "Cannot calculate mode for a column with no non-null values"
            )
        fill_value = modes.iloc[0]
    result[column] = series.fillna(fill_value)
    return result, affected_rows


def drop_duplicates(
    dataframe: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, int]:
    """Removes duplicate rows."""
    if keep not in {"first", "last", False}:
        raise ValueError("keep must be 'first', 'last' or False")
    if subset is not None:
        for column in subset:
            _validate_column(dataframe, column)
    result = dataframe.copy(deep=True)
    original_rows = len(result)
    result = result.drop_duplicates(
        subset=subset,
        keep=keep,
    )
    affected_rows = original_rows - len(result)
    return result, affected_rows


def convert_dtype(
    dataframe: pd.DataFrame,
    column: str,
    dtype: str,
    errors: str = "raise",
) -> tuple[pd.DataFrame, int]:
    """Converts the data type of a column."""
    _validate_column(dataframe, column)

    if errors not in {"raise", "coerce"}:
        raise ValueError("errors must be 'raise' or 'coerce'")

    if dtype not in {"int", "float", "string", "datetime", "bool"}:
        raise ValueError(f"Unsupported dtype: {dtype}")

    result = dataframe.copy(deep=True)
    original = result[column]

    if dtype == "datetime":
        converted = pd.to_datetime(original, errors=errors)

    elif dtype == "int":
        converted = pd.to_numeric(original, errors=errors)

        fractional = converted.notna() & (converted % 1 != 0)

        if fractional.any():
            if errors == "raise":
                raise ValueError(
                    "Cannot convert non-integer values to int"
                )
            converted = converted.mask(fractional)

        converted = converted.astype(
            "Int64" if errors == "coerce" else "int64"
        )

    elif dtype == "float":
        converted = pd.to_numeric(
            original,
            errors=errors,
        ).astype(float)

    elif dtype == "string":
        converted = original.astype("string")

    else:
        if not original.dropna().isin([True, False, 0, 1]).all():
            raise ValueError(
                "Boolean conversion accepts only True, False, 0 and 1"
            )

        converted = original.astype(
            "boolean" if errors == "coerce" else bool
        )

    if dtype in {"datetime", "int", "float", "string", "bool"}:
        affected_rows = int(original.notna().sum())

    if errors == "coerce" and dtype in {"int", "float", "datetime"}:
        invalid_values = original.notna() & converted.isna()
        affected_rows = int(invalid_values.sum())

    result[column] = converted

    return result, affected_rows

def normalize_text(
    dataframe: pd.DataFrame,
    column: str,
    strip: bool = True,
    lowercase: bool = False,
    uppercase: bool = False,
    collapse_whitespace: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Normalizes text values."""
    _validate_column(dataframe, column)
    if lowercase and uppercase:
        raise ValueError(
            "lowercase and uppercase cannot both be True"
        )
    result = dataframe.copy(deep=True)
    original = result[column]
    if not original.dropna().map(
        lambda value: isinstance(value, str)
    ).all():
        raise TypeError(
            f"Column '{column}' must contain only strings and missing values"
        )
    normalized = original.copy()
    if strip:
        normalized = normalized.str.strip()
    if collapse_whitespace:
        normalized = normalized.str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    if lowercase:
        normalized = normalized.str.lower()
    if uppercase:
        normalized = normalized.str.upper()
    changed_mask = (
        original.notna() & original.ne(normalized).fillna(False)
    )
    affected_rows = int(changed_mask.sum())
    result[column] = normalized
    return result, affected_rows


def clip_numeric_values(
    dataframe: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[pd.DataFrame, int]:
    """Clips numeric values to specified boundaries."""
    _validate_column(dataframe, column)
    if min_value is None and max_value is None:
        raise ValueError("At least one boundary must be provided")
    if (
        min_value is not None
        and max_value is not None
        and min_value > max_value
    ):
        raise ValueError("min_value cannot exceed max_value")
    series = dataframe[column]
    if (
        not ptypes.is_numeric_dtype(series.dtype)
        or ptypes.is_bool_dtype(series.dtype)
    ):
        raise TypeError(
            f"Column '{column}' must have a numeric dtype"
        )
    result = dataframe.copy(deep=True)
    clipped = series.clip(
        lower=min_value,
        upper=max_value,
    )
    affected_rows = int(
        (series.notna() & series.ne(clipped)).sum()
    )
    result[column] = clipped
    return result, affected_rows
