from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .models import (
    ColumnPair,
    RelationshipCardinality,
    RelationshipOrigin,
    RelationshipStatus,
    TableRelationship,
)


@dataclass(frozen=True)
class GraphNode:

    source_id: str
    table_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")

        if not isinstance(self.table_name, str) or not self.table_name.strip():
            raise ValueError("table_name must be a non-empty string")


@dataclass(frozen=True)
class GraphEdge:

    relationship_id: str
    left_node: GraphNode
    right_node: GraphNode
    key_pairs: tuple[ColumnPair, ...]
    cardinality: RelationshipCardinality | None
    origin: RelationshipOrigin
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.relationship_id.strip():
            raise ValueError("relationship_id must be non-empty")

        if self.left_node == self.right_node:
            raise ValueError("An edge cannot connect a node to itself")

        if not self.key_pairs:
            raise ValueError("An edge must contain at least one key pair")

        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")

    @classmethod
    def from_relationship(
        cls,
        relationship: TableRelationship,
    ) -> GraphEdge:

        return cls(
            relationship_id=relationship.relationship_id,
            left_node=GraphNode(
                source_id=relationship.left_table.source_id,
                table_name=relationship.left_table.table_name,
            ),
            right_node=GraphNode(
                source_id=relationship.right_table.source_id,
                table_name=relationship.right_table.table_name,
            ),
            key_pairs=tuple(relationship.key_pairs),
            cardinality=relationship.cardinality,
            origin=relationship.origin,
            score=relationship.score,
        )


@dataclass(frozen=True)
class GraphStep:
    from_node: GraphNode
    to_node: GraphNode
    edge: GraphEdge
    key_pairs: tuple[ColumnPair, ...]


@dataclass(frozen=True)
class GraphPath:
    nodes: tuple[GraphNode, ...]
    steps: tuple[GraphStep, ...]

    @property
    def edges(self) -> tuple[GraphEdge, ...]:
        return tuple(step.edge for step in self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)


class RelationshipGraph:
    def __init__(self, relationship_manager=None) -> None:
        self._relationship_manager = relationship_manager
        self._nodes: dict[tuple[str, str], GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._adjacency: dict[
            GraphNode,
            dict[str, GraphNode],
        ] = {}

    @staticmethod
    def _node_key(node: GraphNode) -> tuple[str, str]:
        return node.source_id, node.table_name

    @staticmethod
    def _is_usable_relationship(
        relationship: TableRelationship,
    ) -> bool:
        return (
            relationship.status == RelationshipStatus.CONFIRMED
            and not relationship.is_stale
        )

    def add_node(self, node: GraphNode) -> None:
        key = self._node_key(node)

        if key in self._nodes:
            return

        self._nodes[key] = node
        self._adjacency[node] = {}

    def add_relationship(
        self,
        relationship: TableRelationship,
    ) -> None:
        if not self._is_usable_relationship(relationship):
            raise ValueError(
                "Only confirmed and non-stale relationships can be added"
            )

        edge = GraphEdge.from_relationship(relationship)

        existing = self._edges.get(edge.relationship_id)
        if existing == edge:
            return

        if existing is not None:
            self.remove_relationship(edge.relationship_id)

        self.add_node(edge.left_node)
        self.add_node(edge.right_node)

        self._edges[edge.relationship_id] = edge

        self._adjacency[edge.left_node][edge.relationship_id] = (
            edge.right_node
        )
        self._adjacency[edge.right_node][edge.relationship_id] = (
            edge.left_node
        )

    def remove_relationship(self, relationship_id: str) -> bool:
        edge = self._edges.pop(relationship_id, None)

        if edge is None:
            return False

        self._adjacency[edge.left_node].pop(relationship_id, None)
        self._adjacency[edge.right_node].pop(relationship_id, None)

        return True

    def sync(self, relationship_manager=None) -> None:
        manager = relationship_manager or self._relationship_manager

        if manager is None:
            raise ValueError("RelationshipManager is not configured")

        relationships = manager.get_confirmed_relationships()

        new_graph = RelationshipGraph(
            relationship_manager=manager
        )

        for relationship in relationships:
            if new_graph._is_usable_relationship(relationship):
                new_graph.add_relationship(relationship)

        self._nodes = new_graph._nodes
        self._edges = new_graph._edges
        self._adjacency = new_graph._adjacency
        self._relationship_manager = manager

    def get_node(
        self,
        source_id: str,
        table_name: str,
    ) -> GraphNode | None:
        return self._nodes.get((source_id, table_name))

    def get_neighbors(self, node: GraphNode) -> list[GraphNode]:
        if node not in self._adjacency:
            return []

        return list(dict.fromkeys(self._adjacency[node].values()))

    def get_edges(self) -> list[GraphEdge]:

        return list(self._edges.values())

    def get_relationship(self, relationship_id: str) -> GraphEdge | None:

        return self._edges.get(relationship_id)

    def find_path(
        self,
        start: GraphNode,
        target: GraphNode,
    ) -> GraphPath | None:

        if start not in self._adjacency:
            return None

        if target not in self._adjacency:
            return None

        if start == target:
            return GraphPath(nodes=(start,), steps=())

        queue = deque([start])
        visited = {start}

        parents: dict[
            GraphNode,
            tuple[GraphNode, GraphEdge, bool],
        ] = {}

        while queue:
            current = queue.popleft()

            for relationship_id, neighbor in self._adjacency[current].items():
                if neighbor in visited:
                    continue

                edge = self._edges[relationship_id]
                forward = current == edge.left_node

                visited.add(neighbor)
                parents[neighbor] = (current, edge, forward)

                if neighbor == target:
                    return self._build_path(start, target, parents)

                queue.append(neighbor)

        return None

    def _build_path(
        self,
        start: GraphNode,
        target: GraphNode,
        parents: dict[GraphNode, tuple[GraphNode, GraphEdge, bool]],
    ) -> GraphPath:

        reversed_steps: list[GraphStep] = []
        current = target

        while current != start:
            previous, edge, forward = parents[current]

            if forward:
                key_pairs = edge.key_pairs
            else:
                key_pairs = tuple(
                    ColumnPair(
                        left_column=pair.right_column,
                        right_column=pair.left_column,
                    )
                    for pair in edge.key_pairs
                )

            reversed_steps.append(
                GraphStep(
                    from_node=previous,
                    to_node=current,
                    edge=edge,
                    key_pairs=key_pairs,
                )
            )

            current = previous

        steps = tuple(reversed(reversed_steps))
        nodes = (start,) + tuple(step.to_node for step in steps)

        return GraphPath(nodes=nodes, steps=steps)

    def find_all_paths(
        self,
        start: GraphNode,
        target: GraphNode,
        max_depth: int = 5,
        max_paths: int = 10,
    ) -> list[GraphPath]:

        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        if max_paths < 1:
            raise ValueError("max_paths must be at least 1")

        if start not in self._adjacency or target not in self._adjacency:
            return []

        if start == target:
            return [GraphPath(nodes=(start,), steps=())]

        results: list[GraphPath] = []
        visited = {start}
        current_steps: list[GraphStep] = []

        def dfs(current: GraphNode) -> None:
            if len(results) >= max_paths:
                return

            if current == target:
                nodes = (start,) + tuple(
                    step.to_node for step in current_steps
                )
                results.append(
                    GraphPath(
                        nodes=nodes,
                        steps=tuple(current_steps),
                    )
                )
                return

            if len(current_steps) >= max_depth:
                return

            for relationship_id, neighbor in self._adjacency[current].items():
                if neighbor in visited:
                    continue

                edge = self._edges[relationship_id]
                forward = current == edge.left_node

                if forward:
                    key_pairs = edge.key_pairs
                else:
                    key_pairs = tuple(
                        ColumnPair(
                            left_column=pair.right_column,
                            right_column=pair.left_column,
                        )
                        for pair in edge.key_pairs
                    )

                step = GraphStep(
                    from_node=current,
                    to_node=neighbor,
                    edge=edge,
                    key_pairs=key_pairs,
                )

                visited.add(neighbor)
                current_steps.append(step)

                dfs(neighbor)

                current_steps.pop()
                visited.remove(neighbor)

                if len(results) >= max_paths:
                    return

        dfs(start)
        return results


    def get_subgraph(
        self,
        nodes: Iterable[GraphNode],
    ) -> RelationshipGraph:

        selected_keys = {
            self._node_key(node)
            for node in nodes
            if self._node_key(node) in self._nodes
        }

        subgraph = RelationshipGraph()

        for key in selected_keys:
            subgraph.add_node(self._nodes[key])

        for relationship_id, edge in self._edges.items():
            left_key = self._node_key(edge.left_node)
            right_key = self._node_key(edge.right_node)

            if left_key not in selected_keys or right_key not in selected_keys:
                continue

            subgraph._edges[relationship_id] = edge
            subgraph._adjacency[edge.left_node][relationship_id] = (
                edge.right_node
            )
            subgraph._adjacency[edge.right_node][relationship_id] = (
                edge.left_node
            )

        return subgraph


    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
