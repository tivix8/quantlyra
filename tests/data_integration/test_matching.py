
import pandas as pd
import pytest

from quantlyra.data_integration.matching import (
    match_columns,
    _calculate_name_similarity,
    _get_type_family,
)
from quantlyra.data_integration.models import ColumnMatch


@pytest.fixture
def customers():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "email": ["a@mail.com", "b@mail.com", "c@mail.com"],
        "full_name": ["Alex", "John", "Mike"],
    })


@pytest.fixture
def orders():
    return pd.DataFrame({
        "client_id": [1, 2, 3],
        "customer_email": ["a@mail.com", "b@mail.com", "c@mail.com"],
        "order_id": [101, 102, 103],
    })


class TestColumnNameSimilarity:

    def test_exact_match(self):
        similarity = _calculate_name_similarity(
            "customer_id",
            "customer_id",
        )

        assert similarity == 100.0

    def test_synonym_match(self):
        similarity = _calculate_name_similarity(
            "customer_id",
            "client_id",
        )

        assert similarity == 100.0

    def test_partial_match(self):
        similarity = _calculate_name_similarity(
            "email",
            "customer_email",
        )

        assert similarity == pytest.approx(66.67, abs=0.01)

    def test_unrelated_names(self):
        similarity = _calculate_name_similarity(
            "product_name",
            "order_date",
        )

        assert similarity == 0.0

    def test_typo_match(self):
        similarity = _calculate_name_similarity(
            "customer_id",
            "custmer_id",
        )

        assert similarity > 90.0

    def test_empty_names(self):
        assert _calculate_name_similarity("", "customer_id") == 0.0
        assert _calculate_name_similarity("customer_id", "") == 0.0


class TestTypeFamily:

    @pytest.mark.parametrize(
        "series,expected",
        [
            (pd.Series([1, 2, 3]), "numeric"),
            (pd.Series([1.5, 2.5]), "numeric"),
            (pd.Series(["a", "b"]), "text"),
            (pd.Series([True, False]), "boolean"),
            (
                pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02"])),
                "datetime",
            ),
            (
                pd.Series(pd.to_timedelta(["1 day", "2 days"])),
                "timedelta",
            ),
        ],
    )
    def test_get_type_family(self, series, expected):
        assert _get_type_family(series.dtype) == expected


class TestMatchColumns:

    def test_finds_synonym_match(self, customers, orders):
        matches = match_columns(customers, orders)

        pair = next(
            (
                match
                for match in matches
                if match.left_column == "customer_id"
                and match.right_column == "client_id"
            ),
            None,
        )

        assert pair is not None
        assert pair.name_similarity == 100.0
        assert pair.type_compatible is True
        assert pair.score == 1.0

    def test_finds_partial_match(self, customers, orders):
        matches = match_columns(customers, orders)

        pair = next(
            (
                match
                for match in matches
                if match.left_column == "email"
                and match.right_column == "customer_email"
            ),
            None,
        )

        assert pair is not None
        assert pair.name_similarity == pytest.approx(66.67, abs=0.01)

    def test_rejects_unrelated_columns(self):
        left = pd.DataFrame({
            "customer_id": [1, 2, 3],
        })

        right = pd.DataFrame({
            "product_name": ["A", "B", "C"],
        })

        matches = match_columns(left, right)

        assert matches == []

    def test_respects_similarity_threshold(self):
        left = pd.DataFrame({
            "email": ["a@mail.com"],
        })

        right = pd.DataFrame({
            "customer_email": ["a@mail.com"],
        })

        matches = match_columns(
            left,
            right,
            min_name_similarity=80.0,
        )

        assert matches == []

    def test_accepts_zero_threshold(self):
        left = pd.DataFrame({
            "customer_id": [1],
        })

        right = pd.DataFrame({
            "product_name": ["A"],
        })

        matches = match_columns(
            left,
            right,
            min_name_similarity=0.0,
        )

        assert len(matches) == 1

    def test_empty_left_dataframe(self, orders):
        left = pd.DataFrame()

        assert match_columns(left, orders) == []

    def test_empty_right_dataframe(self, customers):
        right = pd.DataFrame()

        assert match_columns(customers, right) == []

    def test_both_dataframes_empty(self):
        left = pd.DataFrame()
        right = pd.DataFrame()

        assert match_columns(left, right) == []

    def test_incompatible_types_reduce_score(self):
        left = pd.DataFrame({
            "customer_id": [1, 2, 3],
        })

        right = pd.DataFrame({
            "customer_id": ["1", "2", "3"],
        })

        matches = match_columns(left, right)

        assert len(matches) == 1
        assert matches[0].type_compatible is False
        assert matches[0].score == 0.8

    def test_compatible_types_increase_score(self):
        left = pd.DataFrame({
            "customer_id": [1, 2, 3],
        })

        right = pd.DataFrame({
            "customer_id": [4, 5, 6],
        })

        matches = match_columns(left, right)

        assert matches[0].type_compatible is True
        assert matches[0].score == 1.0

    def test_results_are_sorted_by_score(self):
        left = pd.DataFrame({
            "customer_id": [1, 2],
            "email": ["a", "b"],
        })

        right = pd.DataFrame({
            "client_id": [1, 2],
            "customer_email": ["a", "b"],
        })

        matches = match_columns(left, right)

        scores = [match.score for match in matches]

        assert scores == sorted(scores, reverse=True)

    def test_invalid_left_input(self):
        with pytest.raises(TypeError, match="Left input"):
            match_columns([], pd.DataFrame())

    def test_invalid_right_input(self):
        with pytest.raises(TypeError, match="Right input"):
            match_columns(pd.DataFrame(), [])

    @pytest.mark.parametrize(
        "threshold",
        [-1.0, 100.1, 101.0],
    )
    def test_rejects_invalid_threshold(self, threshold):
        with pytest.raises(ValueError, match="similarity"):
            match_columns(
                pd.DataFrame(),
                pd.DataFrame(),
                min_name_similarity=threshold,
            )

    def test_rejects_non_string_column_names(self):
        left = pd.DataFrame({
            123: [1, 2, 3],
        })

        right = pd.DataFrame({
            "customer_id": [1, 2, 3],
        })

        with pytest.raises(TypeError, match="column names"):
            match_columns(left, right)


class TestColumnMatch:

    def test_column_match_creation(self):
        match = ColumnMatch(
            left_column="customer_id",
            right_column="client_id",
            name_similarity=100.0,
            type_compatible=True,
            score=1.0,
            evidence=["Names match"],
        )

        assert match.left_column == "customer_id"
        assert match.right_column == "client_id"
        assert match.name_similarity == 100.0
        assert match.type_compatible is True
        assert match.score == 1.0

    @pytest.mark.parametrize(
        "similarity",
        [-1.0, 100.1],
    )
    def test_rejects_invalid_name_similarity(self, similarity):
        with pytest.raises(ValueError):
            ColumnMatch(
                left_column="customer_id",
                right_column="client_id",
                name_similarity=similarity,
                type_compatible=True,
                score=0.8,
            )

    @pytest.mark.parametrize(
        "score",
        [-0.1, 1.1],
    )
    def test_rejects_invalid_score(self, score):
        with pytest.raises(ValueError):
            ColumnMatch(
                left_column="customer_id",
                right_column="client_id",
                name_similarity=90.0,
                type_compatible=True,
                score=score,
            )

    def test_rejects_empty_column_names(self):
        with pytest.raises(ValueError):
            ColumnMatch(
                left_column="",
                right_column="client_id",
                name_similarity=100.0,
                type_compatible=True,
                score=1.0,
            )
