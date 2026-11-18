
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from quantlyra.data_integration.models import (
    ColumnPair,
    RelationshipCardinality,
    RelationshipEvent,
    RelationshipOrigin,
    RelationshipStatus,
    TableRef,
    TableRelationship,
)


class RelationshipStorage:
    """Persists relationships and their event history."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self) -> tuple[list[TableRelationship], list[RelationshipEvent]]:
        if not self.file_path.exists():
            return [], []

        with self.file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        relationships = [
            self._deserialize_relationship(item)
            for item in data.get("relationships", [])
        ]

        events = [
            self._deserialize_event(item)
            for item in data.get("events", [])
        ]

        return relationships, events

    def save(
        self,
        relationships: list[TableRelationship],
        events: list[RelationshipEvent],
    ) -> None:
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "version": 1,
            "relationships": [
                self._serialize_relationship(item)
                for item in relationships
            ],
            "events": [
                self._serialize_event(item)
                for item in events
            ],
        }

        temporary_path = None

        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.file_path.parent,
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)

                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, self.file_path)

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    @staticmethod
    def _serialize_relationship(
        relationship: TableRelationship,
    ) -> dict:
        return {
            "relationship_id": relationship.relationship_id,
            "left_table": {
                "source_id": relationship.left_table.source_id,
                "table_name": relationship.left_table.table_name,
                "fingerprint": relationship.left_table.fingerprint,
            },
            "right_table": {
                "source_id": relationship.right_table.source_id,
                "table_name": relationship.right_table.table_name,
                "fingerprint": relationship.right_table.fingerprint,
            },
            "key_pairs": [
                {
                    "left_column": pair.left_column,
                    "right_column": pair.right_column,
                }
                for pair in relationship.key_pairs
            ],
            "status": relationship.status.value,
            "cardinality": (
                relationship.cardinality.value
                if relationship.cardinality
                else None
            ),
            "origin": relationship.origin.value,
            "score": relationship.score,
            "evidence": relationship.evidence,
            "key_schema_fingerprint": (
                relationship.key_schema_fingerprint
            ),
            "is_stale": relationship.is_stale,
        }

    @staticmethod
    def _deserialize_relationship(data: dict) -> TableRelationship:
        return TableRelationship(
            relationship_id=data["relationship_id"],
            left_table=TableRef(**data["left_table"]),
            right_table=TableRef(**data["right_table"]),
            key_pairs=[
                ColumnPair(**pair)
                for pair in data["key_pairs"]
            ],
            status=RelationshipStatus(data["status"]),
            cardinality=(
                RelationshipCardinality(data["cardinality"])
                if data["cardinality"]
                else None
            ),
            origin=RelationshipOrigin(data["origin"]),
            score=data["score"],
            evidence=data["evidence"],
            key_schema_fingerprint=data.get(
                "key_schema_fingerprint"
            ),
            is_stale=data.get("is_stale", False),
        )

    @staticmethod
    def _serialize_event(event: RelationshipEvent) -> dict:
        return {
            "event_id": event.event_id,
            "relationship_id": event.relationship_id,
            "event_type": event.event_type,
            "previous_status": (
                event.previous_status.value
                if event.previous_status
                else None
            ),
            "new_status": (
                event.new_status.value
                if event.new_status
                else None
            ),
            "origin": event.origin.value,
            "timestamp": event.timestamp,
            "details": event.details,
        }

    @staticmethod
    def _deserialize_event(data: dict) -> RelationshipEvent:
        return RelationshipEvent(
            event_id=data["event_id"],
            relationship_id=data["relationship_id"],
            event_type=data["event_type"],
            previous_status=(
                RelationshipStatus(data["previous_status"])
                if data["previous_status"]
                else None
            ),
            new_status=(
                RelationshipStatus(data["new_status"])
                if data["new_status"]
                else None
            ),
            origin=RelationshipOrigin(data["origin"]),
            timestamp=data["timestamp"],
            details=data.get("details"),
        )
