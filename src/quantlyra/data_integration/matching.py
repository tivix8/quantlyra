
import re

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)
from rapidfuzz import fuzz

from quantlyra.data_integration.models import ColumnMatch


SYNONYMS = {
    "client": "customer",
    "clients": "customer",
    "customers": "customer",
}


def _tokenize_column_name(name: str) -> list[str]:
    """Splits a column name into normalized tokens."""

    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    tokens = re.findall(r"[a-z0-9]+", name.lower())

    return [
        SYNONYMS.get(token, token)
        for token in tokens
    ]


def _calculate_name_similarity(
    left_name: str,
    right_name: str,
) -> float:
    """Calculates token-based similarity between column names."""

    left_tokens = _tokenize_column_name(left_name)
    right_tokens = _tokenize_column_name(right_name)

    if not left_tokens or not right_tokens:
        return 0.0

    similarities = []

    for left_token in left_tokens:
        for right_token in right_tokens:
            similarity = fuzz.ratio(left_token, right_token)

            if similarity >= 85:
                similarities.append(
                    (similarity, left_token, right_token)
                )

    similarities.sort(reverse=True)

    matched_left = set()
    matched_right = set()
    matched_score = 0.0

    for similarity, left_token, right_token in similarities:
        if left_token in matched_left:
            continue

        if right_token in matched_right:
            continue

        matched_left.add(left_token)
        matched_right.add(right_token)
        matched_score += similarity / 100.0

    total_tokens = len(left_tokens) + len(right_tokens)

    return round(
        (2 * matched_score / total_tokens) * 100,
        2,
    )


def _get_type_family(dtype) -> str:
    """Returns a general data type category."""

    if is_bool_dtype(dtype):
        return "boolean"

    if is_numeric_dtype(dtype):
        return "numeric"

    if is_datetime64_any_dtype(dtype):
        return "datetime"

    if is_timedelta64_dtype(dtype):
        return "timedelta"

    if (
        isinstance(dtype, pd.CategoricalDtype)
        or is_string_dtype(dtype)
        or is_object_dtype(dtype)
    ):
        return "text"

    return str(dtype)


def _are_types_compatible(left_dtype, right_dtype) -> bool:
    """Checks whether two pandas data types are compatible."""

    return (
        _get_type_family(left_dtype)
        == _get_type_family(right_dtype)
    )


def _calculate_score(
    name_similarity: float,
    type_compatible: bool,
) -> float:
    """Calculates a heuristic matching score."""

    name_score = name_similarity / 100.0
    type_score = 1.0 if type_compatible else 0.0

    return round(
        name_score * 0.8 + type_score * 0.2,
        4,
    )


def match_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    min_name_similarity: float = 55.0,
) -> list[ColumnMatch]:
    """
    Finds potential matching columns between two DataFrames.

    Compares token-based column name similarity and data types.
    """

    if not isinstance(left, pd.DataFrame):
        raise TypeError("Left input must be a pandas DataFrame")

    if not isinstance(right, pd.DataFrame):
        raise TypeError("Right input must be a pandas DataFrame")

    if not 0.0 <= min_name_similarity <= 100.0:
        raise ValueError(
            "Minimum name similarity must be between 0 and 100"
        )

    if any(not isinstance(column, str) for column in left.columns):
        raise TypeError("All left column names must be strings")

    if any(not isinstance(column, str) for column in right.columns):
        raise TypeError("All right column names must be strings")

    matches = []

    for left_column in left.columns:
        for right_column in right.columns:
            name_similarity = _calculate_name_similarity(
                left_column,
                right_column,
            )

            if name_similarity < min_name_similarity:
                continue

            type_compatible = _are_types_compatible(
                left[left_column].dtype,
                right[right_column].dtype,
            )

            score = _calculate_score(
                name_similarity,
                type_compatible,
            )

            evidence = [
                f"Name similarity: {name_similarity:.2f}%",
                (
                    "Compatible data types"
                    if type_compatible
                    else "Incompatible data types"
                ),
            ]

            matches.append(
                ColumnMatch(
                    left_column=left_column,
                    right_column=right_column,
                    name_similarity=name_similarity,
                    type_compatible=type_compatible,
                    score=score,
                    evidence=evidence,
                )
            )

    return sorted(
        matches,
        key=lambda match: match.score,
        reverse=True,
    )
