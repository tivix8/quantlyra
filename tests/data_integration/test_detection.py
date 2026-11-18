
import pandas as pd
import pytest

from quantlyra.data_integration.detection import (
    detect_relationships,
    _calculate_overlap,
    _calculate_uniqueness,
    _calculate_null_ratio,
    _determine_cardinality,
)
from quantlyra.data_integration.models import (
    ColumnMatch,
    RelationshipCardinality,
)


@pytest.fixture
def matches():
    return [
        ColumnMatch(
            left_column="customer_id",
            right_column="customer_id",
            name_similarity=100.0,
            type_compatible=True,
            score=1.0,
        )
    ]


class TestCalculateOverlap:

    def test_full_overlap(self):
        left = pd.Series([1, 2, 3])
        right = pd.Series([1, 2, 3])

        result = _calculate_overlap(left, right)

        assert result == (3, 1.0, 1.0)

    def test_partial_overlap(self):
        left = pd.Series([1, 2, 3, 4])
        right = pd.Series([3, 4, 5, 6])

        overlap, left_coverage, right_coverage = (
            _calculate_overlap(left, right)
        )

        assert overlap == 2
        assert left_coverage == 0.5
        assert right_coverage == 0.5

    def test_no_overlap(self):
        left = pd.Series([1, 2, 3])
        right = pd.Series([4, 5, 6])

        assert _calculate_overlap(left, right) == (0, 0.0, 0.0)

    def test_ignores_missing_values(self):
        left = pd.Series([1, 2, None])
        right = pd.Series([1, None, 3])

        overlap, left_coverage, right_coverage = (
            _calculate_overlap(left, right)
        )

        assert overlap == 1
        assert left_coverage == 0.5
        assert right_coverage == 0.5

    def test_empty_series(self):
        left = pd.Series([], dtype="float64")
        right = pd.Series([1, 2, 3])

        assert _calculate_overlap(left, right) == (0, 0.0, 0.0)

    def test_duplicate_values_do_not_increase_overlap(self):
        left = pd.Series([1, 1, 2, 2])
        right = pd.Series([1, 1, 2, 3])

        assert _calculate_overlap(left, right) == (2, 1.0, 2 / 3)


class TestCalculateUniqueness:

    def test_unique_values(self):
        series = pd.Series([1, 2, 3])

        assert _calculate_uniqueness(series) is True

    def test_duplicate_values(self):
        series = pd.Series([1, 2, 2, 3])

        assert _calculate_uniqueness(series) is False

    def test_missing_values_are_ignored(self):
        series = pd.Series([1, 2, None])

        assert _calculate_uniqueness(series) is True

    def test_empty_series(self):
        series = pd.Series([], dtype="float64")

        assert _calculate_uniqueness(series) is True


class TestCalculateNullRatio:

    def test_no_missing_values(self):
        series = pd.Series([1, 2, 3])

        assert _calculate_null_ratio(series) == 0.0

    def test_partial_missing_values(self):
        series = pd.Series([1, None, 3, None])

        assert _calculate_null_ratio(series) == 0.5

    def test_all_missing_values(self):
        series = pd.Series([None, None, None])

        assert _calculate_null_ratio(series) == 1.0

    def test_empty_series(self):
        series = pd.Series([], dtype="float64")

        assert _calculate_null_ratio(series) == 0.0


class TestDetermineCardinality:

    @pytest.mark.parametrize(
        "left_unique,right_unique,expected",
        [
            (True, True, RelationshipCardinality.ONE_TO_ONE),
            (True, False, RelationshipCardinality.ONE_TO_MANY),
            (False, True, RelationshipCardinality.MANY_TO_ONE),
            (False, False, RelationshipCardinality.MANY_TO_MANY),
        ],
    )
    def test_cardinality(
        self,
        left_unique,
        right_unique,
        expected,
    ):
        assert _determine_cardinality(
            left_unique,
            right_unique,
        ) == expected


class TestDetectRelationships:

    def test_detects_one_to_many_relationship(self, matches):
        left = pd.DataFrame({
            "customer_id": [1, 2, 3],
        })

        right = pd.DataFrame({
            "customer_id": [1, 1, 2, 3],
        })

        results = detect_relationships(left, right, matches)

        assert len(results) == 1

        result = results[0]

        assert result.overlap_count == 3
        assert result.left_coverage == 1.0
        assert result.right_coverage == 1.0
        assert result.left_unique is True
        assert result.right_unique is False
        assert result.cardinality == RelationshipCardinality.ONE_TO_MANY
        assert result.score == 1.0

    def test_detects_zero_overlap(self, matches):
        left = pd.DataFrame({
            "customer_id": [1, 2, 3],
        })

        right = pd.DataFrame({
            "customer_id": [4, 5, 6],
        })

        results = detect_relationships(left, right, matches)

        assert len(results) == 1
        assert results[0].overlap_count == 0
        assert results[0].left_coverage == 0.0
        assert results[0].right_coverage == 0.0
        assert "No common non-null values found" in results[0].evidence

    def test_detects_missing_values(self, matches):
        left = pd.DataFrame({
            "customer_id": [1, 2, None],
        })

        right = pd.DataFrame({
            "customer_id": [1, 2, 3, None],
        })

        results = detect_relationships(left, right, matches)

        assert results[0].left_null_ratio == pytest.approx(1 / 3)
        assert results[0].right_null_ratio == 0.25

    def test_handles_empty_dataframes(self, matches):
        left = pd.DataFrame({
            "customer_id": pd.Series(dtype="float64"),
        })

        right = pd.DataFrame({
            "customer_id": pd.Series(dtype="float64"),
        })

        results = detect_relationships(left, right, matches)

        assert len(results) == 1
        assert results[0].overlap_count == 0
        assert results[0].score == 0.2

    def test_returns_empty_list_without_matches(self):
        left = pd.DataFrame({"customer_id": [1, 2]})
        right = pd.DataFrame({"customer_id": [1, 2]})

        assert detect_relationships(left, right, []) == []

    def test_rejects_invalid_left_input(self, matches):
        with pytest.raises(TypeError, match="Left input"):
            detect_relationships([], pd.DataFrame(), matches)

    def test_rejects_invalid_right_input(self, matches):
        with pytest.raises(TypeError, match="Right input"):
            detect_relationships(pd.DataFrame(), [], matches)

    def test_rejects_invalid_matches(self):
        with pytest.raises(TypeError, match="Matches"):
            detect_relationships(
                pd.DataFrame(),
                pd.DataFrame(),
                None,
            )

    def test_rejects_invalid_match_items(self):
        with pytest.raises(TypeError, match="ColumnMatch"):
            detect_relationships(
                pd.DataFrame(),
                pd.DataFrame(),
                ["invalid"],
            )

    def test_rejects_missing_left_column(self, matches):
        left = pd.DataFrame({"other_id": [1, 2]})
        right = pd.DataFrame({"customer_id": [1, 2]})

        with pytest.raises(ValueError, match="not found in left"):
            detect_relationships(left, right, matches)

    def test_rejects_missing_right_column(self, matches):
        left = pd.DataFrame({"customer_id": [1, 2]})
        right = pd.DataFrame({"other_id": [1, 2]})

        with pytest.raises(ValueError, match="not found in right"):
            detect_relationships(left, right, matches)

    def test_rejects_duplicate_column_names(self, matches):
        left = pd.DataFrame(
            [[1, 2]],
            columns=["customer_id", "customer_id"],
        )

        right = pd.DataFrame({"customer_id": [1, 2]})

        with pytest.raises(ValueError, match="unique column names"):
            detect_relationships(left, right, matches)

    def test_results_are_sorted_by_score(self):
        left = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "email": ["a", "b", "c"],
        })

        right = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "email": ["x", "y", "z"],
        })

        matches = [
            ColumnMatch(
                left_column="customer_id",
                right_column="customer_id",
                name_similarity=100.0,
                type_compatible=True,
                score=1.0,
            ),
            ColumnMatch(
                left_column="email",
                right_column="email",
                name_similarity=100.0,
                type_compatible=True,
                score=1.0,
            ),
        ]

        results = detect_relationships(left, right, matches)

        scores = [result.score for result in results]

        assert scores == sorted(scores, reverse=True)


class TestRelationshipDetectionModel:

    def test_rejects_invalid_coverage(self):
        from quantlyra.data_integration.models import RelationshipDetection

        with pytest.raises(ValueError, match="left_coverage"):
            RelationshipDetection(
                left_column="customer_id",
                right_column="customer_id",
                overlap_count=1,
                left_coverage=1.5,
                right_coverage=1.0,
                left_unique=True,
                right_unique=True,
                left_null_ratio=0.0,
                right_null_ratio=0.0,
                cardinality=RelationshipCardinality.ONE_TO_ONE,
                score=1.0,
            )

    def test_rejects_invalid_score(self):
        from quantlyra.data_integration.models import RelationshipDetection

        with pytest.raises(ValueError, match="score"):
            RelationshipDetection(
                left_column="customer_id",
                right_column="customer_id",
                overlap_count=1,
                left_coverage=1.0,
                right_coverage=1.0,
                left_unique=True,
                right_unique=True,
                left_null_ratio=0.0,
                right_null_ratio=0.0,
                cardinality=RelationshipCardinality.ONE_TO_ONE,
                score=float("nan"),
            )
