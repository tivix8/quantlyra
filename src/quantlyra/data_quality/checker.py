from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from .models import QualityIssue, QualityReport
from .models import Severity
from .rules import (
    QualityRule,
    MissingValuesRule,
    DuplicateRowsRule,
    NumericRangeRule,
    CategoricalRule,
    DataTypeRule,
    UniqueValuesRule,
)


def _validate_column(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found")


def _validate_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    for column in columns:
        _validate_column(dataframe, column)


def _create_issue(
    rule: QualityRule,
    message: str,
    row_indices: list[Any],
    column: str | None = None,
) -> QualityIssue:
    return QualityIssue(
        rule_id=rule.rule_id,
        severity=rule.severity,
        message=message,
        affected_rows=len(row_indices),
        column=column,
        row_indices=row_indices,
    )


def _check_missing_values(
    dataframe: pd.DataFrame,
    rule: MissingValuesRule,
) -> QualityIssue | None:
    _validate_column(dataframe, rule.column)

    if not 0 <= rule.max_missing_percentage <= 100:
        raise ValueError(
            "max_missing_percentage must be between 0 and 100"
        )

    missing_mask = dataframe[rule.column].isna()
    missing_count = int(missing_mask.sum())

    if len(dataframe) == 0:
        missing_percentage = 0.0
    else:
        missing_percentage = missing_count / len(dataframe) * 100

    if missing_percentage <= rule.max_missing_percentage:
        return None

    row_indices = dataframe.index[missing_mask].tolist()

    return _create_issue(
        rule=rule,
        message=(
            f"Column '{rule.column}' contains "
            f"{missing_percentage:.2f}% missing values. "
            f"Allowed: {rule.max_missing_percentage:.2f}%"
        ),
        row_indices=row_indices,
        column=rule.column,
    )


def _check_duplicate_rows(
    dataframe: pd.DataFrame,
    rule: DuplicateRowsRule,
) -> QualityIssue | None:
    if rule.columns is not None:
        _validate_columns(dataframe, rule.columns)

    duplicate_mask = dataframe.duplicated(
        subset=rule.columns,
        keep=False,
    )

    row_indices = dataframe.index[duplicate_mask].tolist()

    if not row_indices:
        return None

    description = (
        "entire rows"
        if rule.columns is None
        else f"columns {rule.columns}"
    )

    return _create_issue(
        rule=rule,
        message=f"Duplicate values detected in {description}",
        row_indices=row_indices,
    )


def _check_numeric_range(
    dataframe: pd.DataFrame,
    rule: NumericRangeRule,
) -> QualityIssue | None:
    _validate_column(dataframe, rule.column)

    if rule.min_value is None and rule.max_value is None:
        return None

    if (
        rule.min_value is not None
        and rule.max_value is not None
        and rule.min_value > rule.max_value
    ):
        raise ValueError("min_value cannot exceed max_value")

    series = dataframe[rule.column]

    if not ptypes.is_numeric_dtype(series.dtype) or ptypes.is_bool_dtype(
        series.dtype
    ):
        raise TypeError(
            f"Column '{rule.column}' must have a numeric dtype"
        )

    invalid_mask = pd.Series(False, index=dataframe.index)

    if rule.min_value is not None:
        invalid_mask |= series.notna() & (series < rule.min_value)

    if rule.max_value is not None:
        invalid_mask |= series.notna() & (series > rule.max_value)

    row_indices = dataframe.index[invalid_mask].tolist()

    if not row_indices:
        return None

    return _create_issue(
        rule=rule,
        message=(
            f"Column '{rule.column}' contains values "
            f"outside the allowed range "
            f"[{rule.min_value}, {rule.max_value}]"
        ),
        row_indices=row_indices,
        column=rule.column,
    )


def _check_categorical(
    dataframe: pd.DataFrame,
    rule: CategoricalRule,
) -> QualityIssue | None:
    _validate_column(dataframe, rule.column)

    invalid_mask = (
        dataframe[rule.column].notna()
        & ~dataframe[rule.column].isin(rule.allowed_values)
    )

    row_indices = dataframe.index[invalid_mask].tolist()

    if not row_indices:
        return None

    return _create_issue(
        rule=rule,
        message=(
            f"Column '{rule.column}' contains "
            f"values outside the allowed categories"
        ),
        row_indices=row_indices,
        column=rule.column,
    )


def _matches_expected_dtype(
    series: pd.Series,
    expected_dtype: str,
) -> bool:
    dtype = expected_dtype.lower()

    if dtype == "numeric":
        return (
            ptypes.is_numeric_dtype(series.dtype)
            and not ptypes.is_bool_dtype(series.dtype)
        )

    if dtype == "integer":
        return (
            ptypes.is_integer_dtype(series.dtype)
            and not ptypes.is_bool_dtype(series.dtype)
        )

    if dtype == "float":
        return ptypes.is_float_dtype(series.dtype)

    if dtype == "datetime":
        return ptypes.is_datetime64_any_dtype(series.dtype)

    if dtype == "boolean":
        return ptypes.is_bool_dtype(series.dtype)

    if dtype == "string":
        non_null = series.dropna()

        return (
            ptypes.is_string_dtype(series.dtype)
            and non_null.map(lambda value: isinstance(value, str)).all()
        )

    return str(series.dtype).lower() == dtype


def _check_data_type(
    dataframe: pd.DataFrame,
    rule: DataTypeRule,
) -> QualityIssue | None:
    _validate_column(dataframe, rule.column)

    if not rule.expected_dtype:
        raise ValueError("expected_dtype cannot be empty")

    series = dataframe[rule.column]

    if _matches_expected_dtype(series, rule.expected_dtype):
        return None

    row_indices = dataframe.index.tolist()

    return _create_issue(
        rule=rule,
        message=(
            f"Column '{rule.column}' has dtype '{series.dtype}', "
            f"expected '{rule.expected_dtype}'"
        ),
        row_indices=row_indices,
        column=rule.column,
    )


def _check_unique_values(
    dataframe: pd.DataFrame,
    rule: UniqueValuesRule,
) -> QualityIssue | None:
    _validate_column(dataframe, rule.column)

    series = dataframe[rule.column]

    duplicate_mask = (
        series.notna()
        & series.duplicated(keep=False)
    )

    row_indices = dataframe.index[duplicate_mask].tolist()

    if not row_indices:
        return None

    return _create_issue(
        rule=rule,
        message=(
            f"Column '{rule.column}' contains duplicate values"
        ),
        row_indices=row_indices,
        column=rule.column,
    )


def check_dataframe(
    dataframe: pd.DataFrame,
    rules: list[QualityRule],
    table_name: str = "unknown",
) -> QualityReport:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "DataFrame contains duplicate column names"
        )

    report = QualityReport(
        table_name=table_name,
        row_count=len(dataframe),
    )

    rule_handlers = {
        MissingValuesRule: _check_missing_values,
        DuplicateRowsRule: _check_duplicate_rows,
        NumericRangeRule: _check_numeric_range,
        CategoricalRule: _check_categorical,
        DataTypeRule: _check_data_type,
        UniqueValuesRule: _check_unique_values,
    }

    for rule in rules:
        handler = rule_handlers.get(type(rule))

        if handler is None:
            raise NotImplementedError(
                f"Rule type '{type(rule).__name__}' "
                "is not supported yet"
            )

        issue = handler(dataframe, rule)

        report.checked_rules += 1

        if issue is not None:
            report.issues.append(issue)

    return report
