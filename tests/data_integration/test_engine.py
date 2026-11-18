import pandas as pd
import pytest

from quantlyra.data_integration.engine import (
    AmbiguousJoinPathError,
    IntegrationConfig,
    IntegrationEngine,
    IntegrationValidationError,
    JoinExpansionError,
    JoinStep,
    JoinType,
    NullKeyPolicy,
    RelationshipNotUsableError,
)
from quantlyra.data_integration.graph import RelationshipGraph
from quantlyra.data_integration.models import (
    ColumnPair,
    RelationshipCardinality,
    RelationshipOrigin,
    RelationshipStatus,
    TableRef,
    TableRelationship,
)
from quantlyra.data_integration.relationship_manager import RelationshipManager


class MemoryStorage:
    def __init__(self):
        self.relationships = []
        self.events = []

    def load(self):
        return self.relationships, self.events

    def save(self, relationships, events):
        self.relationships = list(relationships)
        self.events = list(events)


def setup_engine(relations):
    manager = RelationshipManager(MemoryStorage())
    for relationship in relations:
        manager.add_relationship(relationship)
        manager.confirm_relationship(relationship.relationship_id)
    graph = RelationshipGraph(manager)
    graph.sync()
    return IntegrationEngine(graph, manager)


def relationship(left, right, left_key="id", right_key="id", relationship_id=None):
    result = TableRelationship(
        left_table=left,
        right_table=right,
        key_pairs=[ColumnPair(left_key, right_key)],
        origin=RelationshipOrigin.AUTOMATIC,
        cardinality=RelationshipCardinality.ONE_TO_MANY,
        score=1.0,
        key_schema_fingerprint=f"schema-{left.table_name}-{right.table_name}",
    )
    if relationship_id is not None:
        result.relationship_id = relationship_id
    return result


def make_config(left, right, rel, left_key="id", right_key="id", join_type=JoinType.LEFT, **options):
    step = JoinStep(
        left_table=left,
        right_table=right,
        key_pairs=(ColumnPair(left_key, right_key),),
        join_type=join_type,
        relationship_id=rel.relationship_id,
    )
    return IntegrationConfig(base_table=left, steps=(step,), **options)


def test_left_join_returns_expected_rows_and_report():
    left = TableRef("sales", "customers")
    right = TableRef("sales", "orders")
    rel = relationship(left, right, "customer_id", "customer_id")
    engine = setup_engine([rel])
    config = make_config(left, right, rel, "customer_id", "customer_id")
    tables = {
        left: pd.DataFrame({"customer_id": [1, 2, 3], "name": ["A", "B", "C"]}),
        right: pd.DataFrame({"customer_id": [1, 1, 2], "amount": [10, 20, 30]}),
    }

    result = engine.integrate(config, tables)

    assert len(result.dataframe) == 4
    assert result.report.final_rows == 4
    assert result.report.steps[0].matched_key_count == 2
    assert result.report.steps[0].left_unmatched_rows == 1
    assert result.report.steps[0].expansion_factor == pytest.approx(4 / 3)


def test_inner_join_drops_unmatched_rows():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    config = make_config(left, right, rel, join_type=JoinType.INNER)

    result = engine.integrate(
        config,
        {
            left: pd.DataFrame({"id": [1, 2]}),
            right: pd.DataFrame({"id": [2, 3], "value": [20, 30]}),
        },
    )

    assert result.dataframe["id"].tolist() == [2]
    assert result.report.steps[0].left_unmatched_rows == 1
    assert result.report.steps[0].right_unmatched_rows == 1


def test_right_and_outer_join_preserve_unmatched_right_rows():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    tables = {
        left: pd.DataFrame({"id": [1, 2]}),
        right: pd.DataFrame({"id": [2, 3], "value": [20, 30]}),
    }

    right_result = engine.integrate(make_config(left, right, rel, join_type=JoinType.RIGHT), tables)
    outer_result = engine.integrate(make_config(left, right, rel, join_type=JoinType.OUTER), tables)

    assert len(right_result.dataframe) == 2
    assert len(outer_result.dataframe) == 3
    assert outer_result.report.steps[0].right_unmatched_rows == 1


def test_composite_keys_are_supported():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = TableRelationship(
        left_table=left,
        right_table=right,
        key_pairs=[ColumnPair("region", "region_code"), ColumnPair("customer", "customer_code")],
        key_schema_fingerprint="composite-schema",
    )
    engine = setup_engine([rel])
    config = IntegrationConfig(
        base_table=left,
        steps=(JoinStep(left, right, tuple(rel.key_pairs), relationship_id=rel.relationship_id),),
    )

    result = engine.integrate(
        config,
        {
            left: pd.DataFrame({"region": ["N", "S"], "customer": [1, 1]}),
            right: pd.DataFrame({"region_code": ["N"], "customer_code": [1], "amount": [50]}),
        },
    )

    assert result.dataframe["amount"].notna().sum() == 1



def test_join_can_traverse_relationship_in_reverse_direction():
    parent = TableRef("s", "parent")
    child = TableRef("s", "child")

    rel = relationship(parent, child, "id", "parent_id")
    engine = setup_engine([rel])

    config = make_config(
        child,
        parent,
        rel,
        left_key="parent_id",
        right_key="id",
    )

    result = engine.integrate(
        config,
        {
            child: pd.DataFrame({
                "id": [10, 20],
                "parent_id": [1, 2],
            }),
            parent: pd.DataFrame({
                "id": [1, 2],
                "label": ["A", "B"],
            }),
        },
    )

    assert result.dataframe["label"].tolist() == ["A", "B"]
    assert result.report.steps[0].matched_rows == 2



def test_integrate_between_uses_unique_graph_path():
    a = TableRef("s", "a")
    b = TableRef("s", "b")
    c = TableRef("s", "c")
    rel_ab = relationship(a, b)
    rel_bc = relationship(b, c)
    engine = setup_engine([rel_ab, rel_bc])

    result = engine.integrate_between(
        a,
        c,
        {
            a: pd.DataFrame({"id": [1]}),
            b: pd.DataFrame({"id": [1], "b_value": [2]}),
            c: pd.DataFrame({"id": [1], "c_value": [3]}),
        },
    )

    assert result.dataframe["b_value"].tolist() == [2]
    assert result.dataframe["c_value"].tolist() == [3]
    assert len(result.report.steps) == 2


def test_integrate_between_rejects_ambiguous_paths():
    a = TableRef("s", "a")
    b = TableRef("s", "b")
    c = TableRef("s", "c")
    d = TableRef("s", "d")
    relations = [
        relationship(a, b),
        relationship(b, d),
        relationship(a, c),
        relationship(c, d),
    ]
    engine = setup_engine(relations)

    with pytest.raises(AmbiguousJoinPathError):
        engine.integrate_between(a, d, {})


def test_unconfirmed_relationship_cannot_be_used():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    manager = RelationshipManager(MemoryStorage())
    manager.add_relationship(rel)
    graph = RelationshipGraph(manager)
    engine = IntegrationEngine(graph, manager)
    config = make_config(left, right, rel)

    with pytest.raises(RelationshipNotUsableError):
        engine.integrate(
            config,
            {left: pd.DataFrame({"id": [1]}), right: pd.DataFrame({"id": [1]})},
        )


def test_stale_relationship_cannot_be_used():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    engine.relationship_manager.get_relationship(rel.relationship_id).is_stale = True
    config = make_config(left, right, rel)

    with pytest.raises(RelationshipNotUsableError):
        engine.integrate(
            config,
            {left: pd.DataFrame({"id": [1]}), right: pd.DataFrame({"id": [1]})},
        )


def test_null_keys_do_not_match_by_default():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    config = make_config(left, right, rel)

    result = engine.integrate(
        config,
        {
            left: pd.DataFrame({"id": [None, 1]}),
            right: pd.DataFrame({"id": [None, 1], "value": [100, 200]}),
        },
    )

    assert len(result.dataframe) == 2
    assert result.dataframe["value"].notna().sum() == 1
    assert result.report.steps[0].left_null_key_rows == 1
    assert result.report.steps[0].right_null_key_rows == 1


def test_pandas_null_policy_allows_null_to_null_match():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    config = make_config(left, right, rel, null_key_policy=NullKeyPolicy.PANDAS)

    result = engine.integrate(
        config,
        {
            left: pd.DataFrame({"id": [None]}),
            right: pd.DataFrame({"id": [None], "value": [7]}),
        },
    )

    assert result.dataframe["value"].iloc[0] == 7


def test_many_to_many_is_rejected_before_merge_unless_enabled():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    tables = {
        left: pd.DataFrame({"id": [1, 1]}),
        right: pd.DataFrame({"id": [1, 1]}),
    }

    with pytest.raises(JoinExpansionError):
        engine.integrate(make_config(left, right, rel), tables)

    config = make_config(left, right, rel, allow_many_to_many=True)
    result = engine.integrate(config, tables)
    assert len(result.dataframe) == 4


def test_expansion_factor_limit_is_enforced():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    config = make_config(left, right, rel, max_expansion_factor=1.4)

    with pytest.raises(JoinExpansionError):
        engine.integrate(
            config,
            {
                left: pd.DataFrame({"id": [1, 2]}),
                right: pd.DataFrame({"id": [1, 1, 2]}),
            },
        )


def test_max_result_rows_limit_is_enforced():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    config = make_config(left, right, rel, max_result_rows=1)

    with pytest.raises(JoinExpansionError):
        engine.integrate(
            config,
            {
                left: pd.DataFrame({"id": [1, 2]}),
                right: pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
            },
        )


def test_mismatched_key_dtypes_are_rejected():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    config = make_config(left, right, rel)

    with pytest.raises(IntegrationValidationError, match="dtypes differ"):
        engine.integrate(
            config,
            {
                left: pd.DataFrame({"id": [1]}),
                right: pd.DataFrame({"id": ["1"]}),
            },
        )


def test_missing_table_and_missing_key_are_reported():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    config = make_config(left, right, rel)

    with pytest.raises(IntegrationValidationError, match="was not provided"):
        engine.integrate(config, {left: pd.DataFrame({"id": [1]})})

    with pytest.raises(IntegrationValidationError, match="does not exist"):
        engine.integrate(
            config,
            {left: pd.DataFrame({"other": [1]}), right: pd.DataFrame({"id": [1]})},
        )


def test_duplicate_column_names_are_rejected():
    left = TableRef("s", "left")
    right = TableRef("s", "right")
    rel = relationship(left, right)
    engine = setup_engine([rel])
    config = make_config(left, right, rel)
    dataframe = pd.DataFrame([[1, 2]], columns=["id", "id"])

    with pytest.raises(IntegrationValidationError, match="duplicate column"):
        engine.integrate(config, {left: dataframe, right: pd.DataFrame({"id": [1]})})


def test_config_rejects_disconnected_join_chain():
    a = TableRef("s", "a")
    b = TableRef("s", "b")
    c = TableRef("s", "c")
    with pytest.raises(ValueError, match="current integrated table"):
        IntegrationConfig(
            base_table=a,
            steps=(
                JoinStep(a, b, (ColumnPair("id", "id"),), relationship_id="r1"),
                JoinStep(a, c, (ColumnPair("id", "id"),), relationship_id="r2"),
            ),
        )
