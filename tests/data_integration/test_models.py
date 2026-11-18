from dataclasses import FrozenInstanceError

import pytest

from quantlyra.data_integration.models import (
    ColumnPair,
    RelationshipCardinality,
    RelationshipOrigin,
    RelationshipStatus,
    TableRef,
    TableRelationship,
)


@pytest.fixture
def table_refs():
    left = TableRef(
        source_id="source_1",
        table_name="customers",
    )
    right = TableRef(
        source_id="source_2",
        table_name="orders",
    )
    return left, right


@pytest.fixture
def key_pairs():
    return [
        ColumnPair(
            left_column="customer_id",
            right_column="customer_id",
        )
    ]


@pytest.fixture
def relationship(table_refs, key_pairs):
    left, right = table_refs
    return TableRelationship(
        left_table=left,
        right_table=right,
        key_pairs=key_pairs,
    )


class TestTableRef:

    def test_table_ref_creation(self):
        table = TableRef(
            source_id="source_1",
            table_name="customers",
        )

        assert table.source_id == "source_1"
        assert table.table_name == "customers"
        assert table.fingerprint is None

    def test_table_ref_with_fingerprint(self):
        table = TableRef(
            source_id="source_1",
            table_name="customers",
            fingerprint="abc123",
        )

        assert table.fingerprint == "abc123"

    def test_table_ref_is_immutable(self):
        table = TableRef(
            source_id="source_1",
            table_name="customers",
        )

        with pytest.raises(FrozenInstanceError):
            table.table_name = "orders"

    @pytest.mark.parametrize(
        "source_id,table_name",
        [
            ("", "customers"),
            ("   ", "customers"),
            ("source_1", ""),
            ("source_1", "   "),
        ],
    )
    def test_table_ref_rejects_empty_identifiers(
        self,
        source_id,
        table_name,
    ):
        with pytest.raises(ValueError):
            TableRef(
                source_id=source_id,
                table_name=table_name,
            )


class TestColumnPair:

    def test_column_pair_creation(self):
        pair = ColumnPair(
            left_column="customer_id",
            right_column="client_id",
        )

        assert pair.left_column == "customer_id"
        assert pair.right_column == "client_id"

    def test_column_pair_is_immutable(self):
        pair = ColumnPair(
            left_column="customer_id",
            right_column="client_id",
        )

        with pytest.raises(FrozenInstanceError):
            pair.left_column = "id"


class TestTableRelationship:

    def test_relationship_creation(self, relationship):
        assert relationship.status == RelationshipStatus.SUGGESTED
        assert relationship.cardinality is None
        assert relationship.origin == RelationshipOrigin.AUTOMATIC
        assert relationship.score is None
        assert relationship.evidence == []
        assert relationship.relationship_id

    def test_relationship_with_custom_values(self, table_refs, key_pairs):
        left, right = table_refs

        relationship = TableRelationship(
            left_table=left,
            right_table=right,
            key_pairs=key_pairs,
            status=RelationshipStatus.CONFIRMED,
            cardinality=RelationshipCardinality.MANY_TO_ONE,
            origin=RelationshipOrigin.USER,
            score=0.95,
            evidence=[
                "Column names match",
                "High value overlap",
            ],
        )

        assert relationship.status == RelationshipStatus.CONFIRMED
        assert relationship.cardinality == RelationshipCardinality.MANY_TO_ONE
        assert relationship.origin == RelationshipOrigin.USER
        assert relationship.score == 0.95
        assert len(relationship.evidence) == 2

    def test_relationship_generates_unique_ids(self, table_refs, key_pairs):
        left, right = table_refs

        first = TableRelationship(left, right, key_pairs)
        second = TableRelationship(left, right, key_pairs)

        assert first.relationship_id != second.relationship_id

    def test_relationship_requires_key_pairs(self, table_refs):
        left, right = table_refs

        with pytest.raises(
            ValueError,
            match="At least one key pair",
        ):
            TableRelationship(
                left_table=left,
                right_table=right,
                key_pairs=[],
            )

    @pytest.mark.parametrize(
        "score",
        [-0.1, 1.1, -1.0, 2.0],
    )
    def test_relationship_rejects_invalid_score(
        self,
        table_refs,
        key_pairs,
        score,
    ):
        left, right = table_refs

        with pytest.raises(ValueError, match="Score must be"):
            TableRelationship(
                left_table=left,
                right_table=right,
                key_pairs=key_pairs,
                score=score,
            )

    @pytest.mark.parametrize(
        "score",
        [0.0, 0.5, 1.0],
    )
    def test_relationship_accepts_valid_score(
        self,
        table_refs,
        key_pairs,
        score,
    ):
        left, right = table_refs

        relationship = TableRelationship(
            left_table=left,
            right_table=right,
            key_pairs=key_pairs,
            score=score,
        )

        assert relationship.score == score

    @pytest.mark.parametrize(
        "left_column,right_column",
        [
            ("", "customer_id"),
            ("customer_id", ""),
            ("", ""),
            ("   ", "customer_id"),
            ("customer_id", "   "),
        ],
    )
    def test_relationship_rejects_empty_column_names(
        self,
        table_refs,
        left_column,
        right_column,
    ):
        left, right = table_refs

        pair = ColumnPair(
            left_column=left_column,
            right_column=right_column,
        )

        with pytest.raises(
            ValueError,
            match="Column names cannot be empty",
        ):
            TableRelationship(
                left_table=left,
                right_table=right,
                key_pairs=[pair],
            )

    def test_relationship_rejects_same_table(self, key_pairs):
        table = TableRef(
            source_id="source_1",
            table_name="customers",
        )

        with pytest.raises(
            ValueError,
            match="same table",
        ):
            TableRelationship(
                left_table=table,
                right_table=table,
                key_pairs=key_pairs,
            )

    def test_relationship_rejects_same_table_with_different_fingerprints(
        self,
        key_pairs,
    ):
        left = TableRef(
            source_id="source_1",
            table_name="customers",
            fingerprint="version_1",
        )
        right = TableRef(
            source_id="source_1",
            table_name="customers",
            fingerprint="version_2",
        )

        with pytest.raises(ValueError, match="same table"):
            TableRelationship(
                left_table=left,
                right_table=right,
                key_pairs=key_pairs,
            )

    def test_relationship_rejects_duplicate_key_pairs(self, table_refs):
        left, right = table_refs

        pairs = [
            ColumnPair("customer_id", "customer_id"),
            ColumnPair("customer_id", "customer_id"),
        ]

        with pytest.raises(
            ValueError,
            match="Duplicate key pairs",
        ):
            TableRelationship(
                left_table=left,
                right_table=right,
                key_pairs=pairs,
            )

    def test_relationship_accepts_different_key_pairs(self, table_refs):
        left, right = table_refs

        pairs = [
            ColumnPair("customer_id", "customer_id"),
            ColumnPair("email", "client_email"),
        ]

        relationship = TableRelationship(
            left_table=left,
            right_table=right,
            key_pairs=pairs,
        )

        assert len(relationship.key_pairs) == 2

    def test_relationship_allows_same_table_name_in_different_sources(
        self,
        key_pairs,
    ):
        left = TableRef(
            source_id="source_1",
            table_name="customers",
        )
        right = TableRef(
            source_id="source_2",
            table_name="customers",
        )

        relationship = TableRelationship(
            left_table=left,
            right_table=right,
            key_pairs=key_pairs,
        )

        assert relationship.left_table != relationship.right_table
