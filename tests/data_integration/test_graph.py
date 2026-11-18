import pytest

from quantlyra.data_integration.graph import (
    GraphEdge,
    GraphNode,
    RelationshipGraph,
)
from quantlyra.data_integration.models import (
    ColumnPair,
    RelationshipCardinality,
    RelationshipOrigin,
    RelationshipStatus,
    TableRef,
    TableRelationship,
)


# ---------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------

@pytest.fixture
def customer_node():
    return GraphNode("source_1", "customers")


@pytest.fixture
def orders_node():
    return GraphNode("source_1", "orders")


@pytest.fixture
def products_node():
    return GraphNode("source_1", "products")


def make_relationship(
    left_table="customers",
    right_table="orders",
    left_source="source_1",
    right_source="source_1",
    left_column="customer_id",
    right_column="customer_id",
    relationship_id="rel_1",
    status=RelationshipStatus.CONFIRMED,
    is_stale=False,
    score=0.95,
):
    relationship = TableRelationship(
        left_table=TableRef(left_source, left_table),
        right_table=TableRef(right_source, right_table),
        key_pairs=[
            ColumnPair(left_column, right_column)
        ],
        relationship_id=relationship_id,
        status=status,
        cardinality=RelationshipCardinality.ONE_TO_MANY,
        origin=RelationshipOrigin.AUTOMATIC,
        score=score,
        evidence=["Value overlap"],
    )

    relationship.is_stale = is_stale
    return relationship


class MockRelationshipManager:

    def __init__(self, relationships):
        self.relationships = relationships

    def get_confirmed_relationships(self):
        return [
            relationship
            for relationship in self.relationships
            if relationship.status == RelationshipStatus.CONFIRMED
            and not relationship.is_stale
        ]


# ---------------------------------------------------------------------
# GraphNode tests
# ---------------------------------------------------------------------

def test_graph_node_creation():
    node = GraphNode("source_1", "customers")

    assert node.source_id == "source_1"
    assert node.table_name == "customers"


def test_graph_node_equality():
    first = GraphNode("source_1", "customers")
    second = GraphNode("source_1", "customers")

    assert first == second


def test_graph_node_different_sources_are_not_equal():
    first = GraphNode("source_1", "customers")
    second = GraphNode("source_2", "customers")

    assert first != second


@pytest.mark.parametrize(
    "source_id, table_name",
    [
        ("", "customers"),
        ("source_1", ""),
        ("   ", "customers"),
        ("source_1", "   "),
    ],
)
def test_graph_node_rejects_empty_identifiers(
    source_id,
    table_name,
):
    with pytest.raises(ValueError):
        GraphNode(source_id, table_name)


# ---------------------------------------------------------------------
# GraphEdge tests
# ---------------------------------------------------------------------

def test_graph_edge_creation():
    relationship = make_relationship()

    edge = GraphEdge.from_relationship(relationship)

    assert edge.relationship_id == "rel_1"
    assert edge.left_node == GraphNode("source_1", "customers")
    assert edge.right_node == GraphNode("source_1", "orders")
    assert edge.key_pairs == (
        ColumnPair("customer_id", "customer_id"),
    )


def test_graph_edge_rejects_self_connection():
    node = GraphNode("source_1", "customers")

    with pytest.raises(ValueError):
        GraphEdge(
            relationship_id="rel_1",
            left_node=node,
            right_node=node,
            key_pairs=(ColumnPair("id", "id"),),
            cardinality=None,
            origin=RelationshipOrigin.AUTOMATIC,
        )


def test_graph_edge_rejects_empty_key_pairs():
    with pytest.raises(ValueError):
        GraphEdge(
            relationship_id="rel_1",
            left_node=GraphNode("source_1", "customers"),
            right_node=GraphNode("source_1", "orders"),
            key_pairs=(),
            cardinality=None,
            origin=RelationshipOrigin.AUTOMATIC,
        )


def test_graph_edge_rejects_invalid_score():
    with pytest.raises(ValueError):
        GraphEdge(
            relationship_id="rel_1",
            left_node=GraphNode("source_1", "customers"),
            right_node=GraphNode("source_1", "orders"),
            key_pairs=(ColumnPair("id", "id"),),
            cardinality=None,
            origin=RelationshipOrigin.AUTOMATIC,
            score=1.5,
        )


# ---------------------------------------------------------------------
# Node and edge management
# ---------------------------------------------------------------------

def test_add_node(customer_node):
    graph = RelationshipGraph()

    graph.add_node(customer_node)

    assert graph.get_node("source_1", "customers") == customer_node
    assert graph.node_count == 1


def test_add_duplicate_node_does_not_duplicate(customer_node):
    graph = RelationshipGraph()

    graph.add_node(customer_node)
    graph.add_node(customer_node)

    assert graph.node_count == 1


def test_add_relationship():
    graph = RelationshipGraph()
    relationship = make_relationship()

    graph.add_relationship(relationship)

    assert graph.node_count == 2
    assert graph.edge_count == 1


@pytest.mark.parametrize(
    "status, is_stale",
    [
        (RelationshipStatus.SUGGESTED, False),
        (RelationshipStatus.REJECTED, False),
        (RelationshipStatus.CONFIRMED, True),
    ],
)
def test_add_relationship_rejects_unusable_relationship(
    status,
    is_stale,
):
    graph = RelationshipGraph()
    relationship = make_relationship(
        status=status,
        is_stale=is_stale,
    )

    with pytest.raises(ValueError):
        graph.add_relationship(relationship)

    assert graph.edge_count == 0


def test_add_same_relationship_twice():
    graph = RelationshipGraph()
    relationship = make_relationship()

    graph.add_relationship(relationship)
    graph.add_relationship(relationship)

    assert graph.edge_count == 1
    assert graph.node_count == 2


def test_update_existing_relationship():
    graph = RelationshipGraph()

    original = make_relationship(score=0.7)
    updated = make_relationship(score=0.95)

    graph.add_relationship(original)
    graph.add_relationship(updated)

    assert graph.edge_count == 1
    assert graph.get_relationship("rel_1").score == 0.95


def test_remove_relationship():
    graph = RelationshipGraph()
    relationship = make_relationship()

    graph.add_relationship(relationship)

    removed = graph.remove_relationship("rel_1")

    assert removed is True
    assert graph.edge_count == 0


def test_remove_nonexistent_relationship():
    graph = RelationshipGraph()

    removed = graph.remove_relationship("unknown")

    assert removed is False


def test_remove_relationship_keeps_nodes():
    graph = RelationshipGraph()
    relationship = make_relationship()

    graph.add_relationship(relationship)
    graph.remove_relationship("rel_1")

    assert graph.node_count == 2
    assert graph.edge_count == 0


# ---------------------------------------------------------------------
# Neighbor and lookup tests
# ---------------------------------------------------------------------

def test_get_neighbors(customer_node, orders_node):
    graph = RelationshipGraph()
    graph.add_relationship(make_relationship())

    neighbors = graph.get_neighbors(customer_node)

    assert neighbors == [orders_node]


def test_get_neighbors_for_unknown_node():
    graph = RelationshipGraph()
    unknown = GraphNode("source_1", "unknown")

    assert graph.get_neighbors(unknown) == []


def test_get_node_from_another_source():
    graph = RelationshipGraph()
    graph.add_relationship(make_relationship())

    assert graph.get_node("source_1", "customers") is not None
    assert graph.get_node("source_2", "customers") is None


def test_get_edges():
    graph = RelationshipGraph()

    graph.add_relationship(make_relationship())
    graph.add_relationship(
        make_relationship(
            left_table="orders",
            right_table="products",
            relationship_id="rel_2",
            left_column="product_id",
            right_column="product_id",
        )
    )

    assert len(graph.get_edges()) == 2


# ---------------------------------------------------------------------
# Synchronization tests
# ---------------------------------------------------------------------

def test_sync_loads_only_confirmed_relationships():
    relationships = [
        make_relationship(
            relationship_id="confirmed",
            status=RelationshipStatus.CONFIRMED,
        ),
        make_relationship(
            relationship_id="suggested",
            status=RelationshipStatus.SUGGESTED,
        ),
        make_relationship(
            relationship_id="rejected",
            status=RelationshipStatus.REJECTED,
        ),
        make_relationship(
            relationship_id="stale",
            is_stale=True,
        ),
    ]

    manager = MockRelationshipManager(relationships)
    graph = RelationshipGraph(manager)

    graph.sync()

    assert graph.edge_count == 1
    assert graph.get_relationship("confirmed") is not None
    assert graph.get_relationship("suggested") is None
    assert graph.get_relationship("rejected") is None
    assert graph.get_relationship("stale") is None


def test_sync_removes_old_edges():
    manager = MockRelationshipManager([
        make_relationship()
    ])

    graph = RelationshipGraph(manager)
    graph.sync()

    assert graph.edge_count == 1

    manager.relationships = []
    graph.sync()

    assert graph.edge_count == 0
    assert graph.node_count == 0


def test_sync_without_manager_raises_error():
    graph = RelationshipGraph()

    with pytest.raises(ValueError):
        graph.sync()


# ---------------------------------------------------------------------
# Path finding tests
# ---------------------------------------------------------------------

def test_find_path_between_directly_connected_nodes():
    graph = RelationshipGraph()
    graph.add_relationship(make_relationship())

    start = GraphNode("source_1", "customers")
    target = GraphNode("source_1", "orders")

    path = graph.find_path(start, target)

    assert path is not None
    assert path.length == 1
    assert path.nodes == (start, target)


def test_find_path_through_multiple_tables():
    graph = RelationshipGraph()

    graph.add_relationship(make_relationship(
        relationship_id="rel_1",
        left_table="customers",
        right_table="orders",
    ))

    graph.add_relationship(make_relationship(
        relationship_id="rel_2",
        left_table="orders",
        right_table="products",
        left_column="product_id",
        right_column="product_id",
    ))

    start = GraphNode("source_1", "customers")
    target = GraphNode("source_1", "products")

    path = graph.find_path(start, target)

    assert path is not None
    assert path.length == 2
    assert path.nodes == (
        start,
        GraphNode("source_1", "orders"),
        target,
    )


def test_find_path_returns_shortest_path():
    graph = RelationshipGraph()

    # Longer route
    graph.add_relationship(make_relationship(
        relationship_id="rel_1",
        left_table="customers",
        right_table="orders",
    ))
    graph.add_relationship(make_relationship(
        relationship_id="rel_2",
        left_table="orders",
        right_table="products",
    ))

    # Direct route
    graph.add_relationship(make_relationship(
        relationship_id="rel_3",
        left_table="customers",
        right_table="products",
    ))

    path = graph.find_path(
        GraphNode("source_1", "customers"),
        GraphNode("source_1", "products"),
    )

    assert path is not None
    assert path.length == 1
    assert path.edges[0].relationship_id == "rel_3"


def test_find_path_returns_none_for_disconnected_nodes():
    graph = RelationshipGraph()

    graph.add_node(GraphNode("source_1", "customers"))
    graph.add_node(GraphNode("source_1", "products"))

    path = graph.find_path(
        GraphNode("source_1", "customers"),
        GraphNode("source_1", "products"),
    )

    assert path is None


def test_find_path_returns_none_for_unknown_node():
    graph = RelationshipGraph()
    graph.add_node(GraphNode("source_1", "customers"))

    path = graph.find_path(
        GraphNode("source_1", "customers"),
        GraphNode("source_1", "unknown"),
    )

    assert path is None


def test_find_path_from_node_to_itself():
    graph = RelationshipGraph()
    node = GraphNode("source_1", "customers")
    graph.add_node(node)

    path = graph.find_path(node, node)

    assert path is not None
    assert path.length == 0
    assert path.nodes == (node,)


def test_find_path_in_reverse_direction_swaps_keys():
    graph = RelationshipGraph()

    relationship = make_relationship(
        left_table="customers",
        right_table="orders",
        left_column="customer_id",
        right_column="client_id",
    )

    graph.add_relationship(relationship)

    path = graph.find_path(
        GraphNode("source_1", "orders"),
        GraphNode("source_1", "customers"),
    )

    assert path is not None

    step = path.steps[0]

    assert step.key_pairs == (
        ColumnPair("client_id", "customer_id"),
    )


# ---------------------------------------------------------------------
# Alternative paths
# ---------------------------------------------------------------------

def test_find_all_paths_returns_alternative_routes():
    graph = RelationshipGraph()

    graph.add_relationship(make_relationship(
        relationship_id="rel_1",
        left_table="customers",
        right_table="orders",
    ))
    graph.add_relationship(make_relationship(
        relationship_id="rel_2",
        left_table="orders",
        right_table="products",
    ))
    graph.add_relationship(make_relationship(
        relationship_id="rel_3",
        left_table="customers",
        right_table="products",
    ))

    paths = graph.find_all_paths(
        GraphNode("source_1", "customers"),
        GraphNode("source_1", "products"),
    )

    assert len(paths) == 2
    assert {path.length for path in paths} == {1, 2}


def test_find_all_paths_respects_max_depth():
    graph = RelationshipGraph()

    graph.add_relationship(make_relationship(
        relationship_id="rel_1",
        left_table="customers",
        right_table="orders",
    ))
    graph.add_relationship(make_relationship(
        relationship_id="rel_2",
        left_table="orders",
        right_table="products",
    ))

    paths = graph.find_all_paths(
        GraphNode("source_1", "customers"),
        GraphNode("source_1", "products"),
        max_depth=1,
    )

    assert paths == []


def test_find_all_paths_respects_max_paths():
    graph = RelationshipGraph()

    graph.add_relationship(make_relationship(
        relationship_id="rel_1",
        left_table="customers",
        right_table="orders",
    ))
    graph.add_relationship(make_relationship(
        relationship_id="rel_2",
        left_table="orders",
        right_table="products",
    ))
    graph.add_relationship(make_relationship(
        relationship_id="rel_3",
        left_table="customers",
        right_table="products",
    ))

    paths = graph.find_all_paths(
        GraphNode("source_1", "customers"),
        GraphNode("source_1", "products"),
        max_paths=1,
    )

    assert len(paths) == 1


def test_find_all_paths_rejects_invalid_limits():
    graph = RelationshipGraph()
    start = GraphNode("source_1", "customers")
    target = GraphNode("source_1", "orders")

    with pytest.raises(ValueError):
        graph.find_all_paths(start, target, max_depth=0)

    with pytest.raises(ValueError):
        graph.find_all_paths(start, target, max_paths=0)


# ---------------------------------------------------------------------
# Subgraph tests
# ---------------------------------------------------------------------

def test_get_subgraph_contains_only_selected_nodes_and_edges():
    graph = RelationshipGraph()

    graph.add_relationship(make_relationship(
        relationship_id="rel_1",
        left_table="customers",
        right_table="orders",
    ))

    graph.add_relationship(make_relationship(
        relationship_id="rel_2",
        left_table="orders",
        right_table="products",
    ))

    customers = GraphNode("source_1", "customers")
    orders = GraphNode("source_1", "orders")

    subgraph = graph.get_subgraph([customers, orders])

    assert subgraph.node_count == 2
    assert subgraph.edge_count == 1
    assert subgraph.get_relationship("rel_1") is not None
    assert subgraph.get_relationship("rel_2") is None


def test_get_subgraph_keeps_isolated_selected_nodes():
    graph = RelationshipGraph()

    customers = GraphNode("source_1", "customers")
    products = GraphNode("source_1", "products")

    graph.add_node(customers)
    graph.add_node(products)

    subgraph = graph.get_subgraph([customers, products])

    assert subgraph.node_count == 2
    assert subgraph.edge_count == 0


def test_get_subgraph_ignores_unknown_nodes():
    graph = RelationshipGraph()

    customers = GraphNode("source_1", "customers")
    unknown = GraphNode("source_1", "unknown")

    graph.add_node(customers)

    subgraph = graph.get_subgraph([customers, unknown])

    assert subgraph.node_count == 1
    assert subgraph.get_node("source_1", "unknown") is None
