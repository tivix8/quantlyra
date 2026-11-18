import hashlib
import json
from threading import RLock

import pandas as pd

from quantlyra.data_integration.models import (
    ColumnPair,
    RelationshipCardinality,
    RelationshipEvent,
    RelationshipOrigin,
    RelationshipStatus,
    TableRelationship,
)
from quantlyra.data_integration.storage import RelationshipStorage


class RelationshipManager:

    def __init__(self, storage: RelationshipStorage) -> None:
        self.storage = storage
        self._lock = RLock()

        (
            self._relationships,
            self._events,
        ) = self.storage.load()

    @staticmethod
    def build_key_schema_fingerprint(
        left: pd.DataFrame,
        right: pd.DataFrame,
        key_pairs: list[ColumnPair],
    ) -> str:
        schema = []

        for pair in key_pairs:
            if pair.left_column not in left.columns:
                raise ValueError(
                    f"Column '{pair.left_column}' not found in left DataFrame"
                )

            if pair.right_column not in right.columns:
                raise ValueError(
                    f"Column '{pair.right_column}' not found in right DataFrame"
                )

            schema.append({
                "left_column": pair.left_column,
                "left_dtype": str(left[pair.left_column].dtype),
                "right_column": pair.right_column,
                "right_dtype": str(right[pair.right_column].dtype),
            })

        serialized = json.dumps(
            schema,
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _relationship_identity(
        relationship: TableRelationship,
    ) -> tuple:
        left_table = (
            relationship.left_table.source_id,
            relationship.left_table.table_name,
        )
        right_table = (
            relationship.right_table.source_id,
            relationship.right_table.table_name,
        )

        pairs = tuple(sorted(
            (
                pair.left_column,
                pair.right_column,
            )
            for pair in relationship.key_pairs
        ))

        forward = (left_table, right_table, pairs)

        reverse_pairs = tuple(sorted(
            (
                pair.right_column,
                pair.left_column,
            )
            for pair in relationship.key_pairs
        ))

        reverse = (right_table, left_table, reverse_pairs)

        return min(forward, reverse)

    def _record_event(
        self,
        relationship: TableRelationship,
        event_type: str,
        previous_status: RelationshipStatus | None,
        new_status: RelationshipStatus | None,
        origin: RelationshipOrigin,
        details: str | None = None,
    ) -> None:
        self._events.append(
            RelationshipEvent(
                relationship_id=relationship.relationship_id,
                event_type=event_type,
                previous_status=previous_status,
                new_status=new_status,
                origin=origin,
                details=details,
            )
        )

    def _save(self) -> None:
        self.storage.save(
            self._relationships,
            self._events,
        )

    def add_relationship(
        self,
        relationship: TableRelationship,
    ) -> TableRelationship:
        if not isinstance(relationship, TableRelationship):
            raise TypeError(
                "Relationship must be a TableRelationship instance"
            )

        if relationship.key_schema_fingerprint is None:
            raise ValueError(
                "Key schema fingerprint must be provided"
            )

        with self._lock:
            identity = self._relationship_identity(relationship)

            existing = [
                item
                for item in self._relationships
                if self._relationship_identity(item) == identity
                and not item.is_stale
            ]

            for item in existing:
                if (
                    item.key_schema_fingerprint
                    == relationship.key_schema_fingerprint
                ):
                    previous_status = item.status

                    item.score = relationship.score
                    item.cardinality = relationship.cardinality
                    item.evidence = list(relationship.evidence)

                    self._save()
                    return item

                item.is_stale = True

                self._record_event(
                    relationship=item,
                    event_type="schema_changed",
                    previous_status=item.status,
                    new_status=item.status,
                    origin=RelationshipOrigin.AUTOMATIC,
                    details=(
                        "Relationship key schema changed. "
                        "A new suggestion is required."
                    ),
                )

            relationship.status = RelationshipStatus.SUGGESTED
            relationship.is_stale = False

            self._relationships.append(relationship)

            self._record_event(
                relationship=relationship,
                event_type="created",
                previous_status=None,
                new_status=RelationshipStatus.SUGGESTED,
                origin=relationship.origin,
            )

            self._save()
            return relationship

    def confirm_relationship(
        self,
        relationship_id: str,
    ) -> TableRelationship:
        return self._change_status(
            relationship_id,
            RelationshipStatus.CONFIRMED,
        )

    def reject_relationship(
        self,
        relationship_id: str,
    ) -> TableRelationship:
        return self._change_status(
            relationship_id,
            RelationshipStatus.REJECTED,
        )

    def _change_status(
        self,
        relationship_id: str,
        new_status: RelationshipStatus,
    ) -> TableRelationship:
        with self._lock:
            relationship = self.get_relationship(relationship_id)

            if relationship.is_stale:
                raise ValueError(
                    "Cannot change the status of a stale relationship"
                )

            if relationship.status == new_status:
                return relationship

            if relationship.status != RelationshipStatus.SUGGESTED:
                raise ValueError(
                    "Only suggested relationships can be confirmed or rejected"
                )

            previous_status = relationship.status
            relationship.status = new_status

            self._record_event(
                relationship=relationship,
                event_type=new_status.value,
                previous_status=previous_status,
                new_status=new_status,
                origin=RelationshipOrigin.USER,
            )

            self._save()
            return relationship

    def get_relationship(
        self,
        relationship_id: str,
    ) -> TableRelationship:
        for relationship in self._relationships:
            if relationship.relationship_id == relationship_id:
                return relationship

        raise KeyError(
            f"Relationship '{relationship_id}' not found"
        )

    def get_relationships(
        self,
        status: RelationshipStatus | None = None,
        include_stale: bool = False,
    ) -> list[TableRelationship]:
        with self._lock:
            return [
                relationship
                for relationship in self._relationships
                if (
                    status is None
                    or relationship.status == status
                )
                and (
                    include_stale
                    or not relationship.is_stale
                )
            ]

    def get_confirmed_relationships(
        self,
    ) -> list[TableRelationship]:
        return self.get_relationships(
            status=RelationshipStatus.CONFIRMED,
        )

    def get_history(
        self,
        relationship_id: str,
    ) -> list[RelationshipEvent]:
        with self._lock:
            return [
                event
                for event in self._events
                if event.relationship_id == relationship_id
            ]

    def revalidate_relationship(
        self,
        relationship_id: str,
        new_key_schema_fingerprint: str,
    ) -> TableRelationship:
        if not new_key_schema_fingerprint:
            raise ValueError(
                "Key schema fingerprint cannot be empty"
            )

        with self._lock:
            relationship = self.get_relationship(relationship_id)

            if relationship.is_stale:
                return relationship

            if (
                relationship.key_schema_fingerprint
                == new_key_schema_fingerprint
            ):
                return relationship

            previous_status = relationship.status
            relationship.is_stale = True

            self._record_event(
                relationship=relationship,
                event_type="schema_changed",
                previous_status=previous_status,
                new_status=previous_status,
                origin=RelationshipOrigin.AUTOMATIC,
                details=(
                    "Key schema changed. "
                    "The relationship requires a new detection result."
                ),
            )

            self._save()
            return relationship
