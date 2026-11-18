import pandas as pd
import pytest

from quantlyra.data_integration.models import (
    ColumnPair,
    RelationshipCardinality,
    RelationshipOrigin,
    RelationshipStatus,
    TableRef,
    TableRelationship,
)
from quantlyra.data_integration.relationship_manager import (
    RelationshipManager,
)
from quantlyra.data_integration.storage import RelationshipStorage


@pytest.fixture
def dataframes():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    })

    orders = pd.DataFrame({
        "customer_id": [1, 1, 2, 3],
        "amount": [100, 200, 150, 300],
    })

    return customers, orders


@pytest.fixture
def manager(tmp_path):
    storage = RelationshipStorage(
        tmp_path / "relationships.json"
    )

    return RelationshipManager(storage)


def create_relationship(
    manager,
    dataframes,
    left_column="customer_id",
    right_column="customer_id",
):
    left, right = dataframes

    key_pairs = [
        ColumnPair(left_column, right_column)
    ]

    fingerprint = manager.build_key_schema_fingerprint(
        left,
        right,
        key_pairs,
    )

    return TableRelationship(
        left_table=TableRef("source_1", "customers"),
        right_table=TableRef("source_1", "orders"),
        key_pairs=key_pairs,
        cardinality=RelationshipCardinality.ONE_TO_MANY,
        origin=RelationshipOrigin.AUTOMATIC,
        score=1.0,
        evidence=["Full value overlap"],
        key_schema_fingerprint=fingerprint,
    )


def test_add_relationship(manager, dataframes):
    relationship = create_relationship(manager, dataframes)

    saved = manager.add_relationship(relationship)

    assert saved.status == RelationshipStatus.SUGGESTED
    assert len(manager.get_relationships()) == 1


def test_add_duplicate_relationship(manager, dataframes):
    first = create_relationship(manager, dataframes)
    second = create_relationship(manager, dataframes)

    saved_first = manager.add_relationship(first)
    saved_second = manager.add_relationship(second)

    assert saved_first.relationship_id == saved_second.relationship_id
    assert len(manager.get_relationships()) == 1


def test_confirm_relationship(manager, dataframes):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    confirmed = manager.confirm_relationship(
        relationship.relationship_id
    )

    assert confirmed.status == RelationshipStatus.CONFIRMED


def test_reject_relationship(manager, dataframes):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    rejected = manager.reject_relationship(
        relationship.relationship_id
    )

    assert rejected.status == RelationshipStatus.REJECTED


def test_rejected_relationship_remains_rejected(
    manager,
    dataframes,
):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.reject_relationship(relationship.relationship_id)

    repeated = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    assert repeated.status == RelationshipStatus.REJECTED


def test_confirmed_relationship_remains_confirmed(
    manager,
    dataframes,
):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.confirm_relationship(relationship.relationship_id)

    repeated = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    assert repeated.status == RelationshipStatus.CONFIRMED


def test_cannot_confirm_rejected_relationship(
    manager,
    dataframes,
):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.reject_relationship(relationship.relationship_id)

    with pytest.raises(ValueError):
        manager.confirm_relationship(relationship.relationship_id)


def test_cannot_reject_confirmed_relationship(
    manager,
    dataframes,
):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.confirm_relationship(relationship.relationship_id)

    with pytest.raises(ValueError):
        manager.reject_relationship(relationship.relationship_id)


def test_get_relationship(manager, dataframes):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    result = manager.get_relationship(
        relationship.relationship_id
    )

    assert result.relationship_id == relationship.relationship_id


def test_get_unknown_relationship(manager):
    with pytest.raises(KeyError):
        manager.get_relationship("unknown")


def test_get_confirmed_relationships(manager, dataframes):
    first = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    second = manager.add_relationship(
        TableRelationship(
            left_table=TableRef("source_1", "products"),
            right_table=TableRef("source_1", "orders"),
            key_pairs=[
                ColumnPair("product_id", "product_id")
            ],
            key_schema_fingerprint="different-schema",
        )
    )

    manager.confirm_relationship(first.relationship_id)

    confirmed = manager.get_confirmed_relationships()

    assert len(confirmed) == 1
    assert confirmed[0].relationship_id == first.relationship_id


def test_history_records_creation_and_confirmation(
    manager,
    dataframes,
):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.confirm_relationship(relationship.relationship_id)

    history = manager.get_history(relationship.relationship_id)

    assert len(history) == 2
    assert history[0].event_type == "created"
    assert history[1].event_type == "confirmed"


def test_history_records_rejection(manager, dataframes):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.reject_relationship(relationship.relationship_id)

    history = manager.get_history(relationship.relationship_id)

    assert history[-1].new_status == RelationshipStatus.REJECTED


def test_revalidate_unchanged_schema(manager, dataframes):
    relationship = create_relationship(manager, dataframes)

    saved = manager.add_relationship(relationship)

    result = manager.revalidate_relationship(
        saved.relationship_id,
        saved.key_schema_fingerprint,
    )

    assert result.is_stale is False


def test_revalidate_changed_schema(manager, dataframes):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.confirm_relationship(relationship.relationship_id)

    result = manager.revalidate_relationship(
        relationship.relationship_id,
        "changed-schema",
    )

    assert result.is_stale is True
    assert result.status == RelationshipStatus.CONFIRMED


def test_stale_relationship_not_returned_by_default(
    manager,
    dataframes,
):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.revalidate_relationship(
        relationship.relationship_id,
        "changed-schema",
    )

    assert manager.get_relationships() == []
    assert len(
        manager.get_relationships(include_stale=True)
    ) == 1


def test_cannot_confirm_stale_relationship(
    manager,
    dataframes,
):
    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.revalidate_relationship(
        relationship.relationship_id,
        "changed-schema",
    )

    with pytest.raises(ValueError):
        manager.confirm_relationship(relationship.relationship_id)


def test_persistence_after_restart(tmp_path, dataframes):
    path = tmp_path / "relationships.json"

    manager = RelationshipManager(
        RelationshipStorage(path)
    )

    relationship = manager.add_relationship(
        create_relationship(manager, dataframes)
    )

    manager.confirm_relationship(relationship.relationship_id)

    restored_manager = RelationshipManager(
        RelationshipStorage(path)
    )

    restored = restored_manager.get_relationship(
        relationship.relationship_id
    )

    assert restored.status == RelationshipStatus.CONFIRMED

    history = restored_manager.get_history(
        relationship.relationship_id
    )

    assert len(history) == 2


def test_key_schema_fingerprint_ignores_data_changes(
    manager,
    dataframes,
):
    customers, orders = dataframes

    original = manager.build_key_schema_fingerprint(
        customers,
        orders,
        [ColumnPair("customer_id", "customer_id")],
    )

    updated_customers = customers.copy()
    updated_customers.loc[0, "customer_id"] = 100

    updated_orders = orders.copy()
    updated_orders.loc[0, "customer_id"] = 100

    updated = manager.build_key_schema_fingerprint(
        updated_customers,
        updated_orders,
        [ColumnPair("customer_id", "customer_id")],
    )

    assert original == updated


def test_key_schema_fingerprint_ignores_unrelated_columns(
    manager,
    dataframes,
):
    customers, orders = dataframes

    original = manager.build_key_schema_fingerprint(
        customers,
        orders,
        [ColumnPair("customer_id", "customer_id")],
    )

    customers["new_column"] = ["a", "b", "c"]
    orders["another_column"] = [1, 2, 3, 4]

    updated = manager.build_key_schema_fingerprint(
        customers,
        orders,
        [ColumnPair("customer_id", "customer_id")],
    )

    assert original == updated


def test_key_schema_fingerprint_changes_when_key_dtype_changes(
    manager,
    dataframes,
):
    customers, orders = dataframes

    original = manager.build_key_schema_fingerprint(
        customers,
        orders,
        [ColumnPair("customer_id", "customer_id")],
    )

    customers["customer_id"] = customers["customer_id"].astype(str)

    updated = manager.build_key_schema_fingerprint(
        customers,
        orders,
        [ColumnPair("customer_id", "customer_id")],
    )

    assert original != updated


def test_key_schema_fingerprint_missing_column(
    manager,
    dataframes,
):
    customers, orders = dataframes

    with pytest.raises(ValueError):
        manager.build_key_schema_fingerprint(
            customers,
            orders,
            [ColumnPair("missing", "customer_id")],
        )


def test_reversed_relationship_is_not_duplicated(
    manager,
    dataframes,
):
    customers, orders = dataframes

    fingerprint = manager.build_key_schema_fingerprint(
        customers,
        orders,
        [ColumnPair("customer_id", "customer_id")],
    )

    first = TableRelationship(
        left_table=TableRef("source_1", "customers"),
        right_table=TableRef("source_1", "orders"),
        key_pairs=[ColumnPair("customer_id", "customer_id")],
        key_schema_fingerprint=fingerprint,
    )

    second = TableRelationship(
        left_table=TableRef("source_1", "orders"),
        right_table=TableRef("source_1", "customers"),
        key_pairs=[ColumnPair("customer_id", "customer_id")],
        key_schema_fingerprint=fingerprint,
    )

    manager.add_relationship(first)
    result = manager.add_relationship(second)

    assert len(manager.get_relationships()) == 1
    assert result.relationship_id == first.relationship_id
