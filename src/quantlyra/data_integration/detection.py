import pandas as pd

from quantlyra.data_integration.models import (
    ColumnMatch,
    RelationshipCardinality,
    RelationshipDetection,
)


def _calculate_overlap(
    left: pd.Series,
    right: pd.Series,
) -> tuple[int, float, float]:
    left_values = set(left.dropna().unique())
    right_values = set(right.dropna().unique())

    overlap = left_values & right_values

    left_coverage = (
        len(overlap) / len(left_values)
        if left_values
        else 0.0
    )

    right_coverage = (
        len(overlap) / len(right_values)
        if right_values
        else 0.0
    )

    return (
        len(overlap),
        left_coverage,
        right_coverage,
    )


def _calculate_uniqueness(series: pd.Series) -> bool:

    values = series.dropna()

    return values.is_unique


def _calculate_null_ratio(series: pd.Series) -> float:

    if len(series) == 0:
        return 0.0

    return float(series.isna().mean())


def _determine_cardinality(
    left_unique: bool,
    right_unique: bool,
) -> RelationshipCardinality:

    if left_unique and right_unique:
        return RelationshipCardinality.ONE_TO_ONE

    if left_unique and not right_unique:
        return RelationshipCardinality.ONE_TO_MANY

    if not left_unique and right_unique:
        return RelationshipCardinality.MANY_TO_ONE

    return RelationshipCardinality.MANY_TO_MANY


def _calculate_detection_score(
    left_coverage: float,
    right_coverage: float,
    left_null_ratio: float,
    right_null_ratio: float,
) -> float:

    coverage_score = (
        left_coverage + right_coverage
    ) / 2

    quality_score = 1 - (
        left_null_ratio + right_null_ratio
    ) / 2

    score = (
        coverage_score * 0.8
        + quality_score * 0.2
    )

    return round(score, 4)


def _validate_inputs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    matches: list[ColumnMatch],
) -> None:

    if not isinstance(left, pd.DataFrame):
        raise TypeError("Left input must be a pandas DataFrame")

    if not isinstance(right, pd.DataFrame):
        raise TypeError("Right input must be a pandas DataFrame")

    if not isinstance(matches, list):
        raise TypeError("Matches must be a list")

    if any(not isinstance(match, ColumnMatch) for match in matches):
        raise TypeError("All matches must be ColumnMatch instances")

    if not left.columns.is_unique or not right.columns.is_unique:
        raise ValueError("DataFrames must have unique column names")

    for match in matches:
        if match.left_column not in left.columns:
            raise ValueError(
                f"Column '{match.left_column}' not found in left DataFrame"
            )

        if match.right_column not in right.columns:
            raise ValueError(
                f"Column '{match.right_column}' not found in right DataFrame"
            )


def detect_relationships(
    left: pd.DataFrame,
    right: pd.DataFrame,
    matches: list[ColumnMatch],
) -> list[RelationshipDetection]:

    _validate_inputs(left, right, matches)

    detections = []

    for match in matches:
        left_series = left[match.left_column]
        right_series = right[match.right_column]

        overlap_count, left_coverage, right_coverage = (
            _calculate_overlap(left_series, right_series)
        )

        left_unique = _calculate_uniqueness(left_series)
        right_unique = _calculate_uniqueness(right_series)

        left_null_ratio = _calculate_null_ratio(left_series)
        right_null_ratio = _calculate_null_ratio(right_series)

        cardinality = _determine_cardinality(
            left_unique,
            right_unique,
        )

        score = _calculate_detection_score(
            left_coverage,
            right_coverage,
            left_null_ratio,
            right_null_ratio,
        )

        evidence = [
            f"Common unique values: {overlap_count}",
            f"Left coverage: {left_coverage:.2%}",
            f"Right coverage: {right_coverage:.2%}",
            f"Left uniqueness: {left_unique}",
            f"Right uniqueness: {right_unique}",
            f"Left null ratio: {left_null_ratio:.2%}",
            f"Right null ratio: {right_null_ratio:.2%}",
        ]

        if overlap_count == 0:
            evidence.append("No common non-null values found")

        detections.append(
            RelationshipDetection(
                left_column=match.left_column,
                right_column=match.right_column,
                overlap_count=overlap_count,
                left_coverage=left_coverage,
                right_coverage=right_coverage,
                left_unique=left_unique,
                right_unique=right_unique,
                left_null_ratio=left_null_ratio,
                right_null_ratio=right_null_ratio,
                cardinality=cardinality,
                score=score,
                evidence=evidence,
            )
        )

    return sorted(
        detections,
        key=lambda detection: detection.score,
        reverse=True,
    )
