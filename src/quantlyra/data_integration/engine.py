from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import pandas as pd
from pandas.api.types import is_dtype_equal

from .graph import GraphNode, RelationshipGraph
from .models import (
    ColumnPair,
    RelationshipStatus,
    TableRef,
)
from .relationship_manager import RelationshipManager


class JoinType(str, Enum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"


class NullKeyPolicy(str, Enum):
    PANDAS = "pandas"
    NEVER_MATCH = "never_match"


class IntegrationError(Exception):
    pass


class IntegrationValidationError(IntegrationError, ValueError):
    pass


class RelationshipNotUsableError(IntegrationError):
    pass


class AmbiguousJoinPathError(IntegrationError):
    pass


class JoinExpansionError(IntegrationError):
    pass


@dataclass(frozen=True)
class JoinStep:
    left_table: TableRef
    right_table: TableRef
    key_pairs: tuple[ColumnPair, ...]
    join_type: JoinType = JoinType.LEFT
    relationship_id: str | None = None

    def __post_init__(self) -> None:
        if not self.key_pairs:
            raise ValueError("A join step must contain at least one key pair")
        if _table_identity(self.left_table) == _table_identity(self.right_table):
            raise ValueError("A table cannot be joined to itself in one step")
        if not isinstance(self.join_type, JoinType):
            object.__setattr__(self, "join_type", JoinType(self.join_type))
        left_keys = [pair.left_column for pair in self.key_pairs]
        right_keys = [pair.right_column for pair in self.key_pairs]
        if len(left_keys) != len(set(left_keys)):
            raise ValueError("Left join columns must be unique")
        if len(right_keys) != len(set(right_keys)):
            raise ValueError("Right join columns must be unique")


@dataclass(frozen=True)
class IntegrationConfig:
    base_table: TableRef
    steps: tuple[JoinStep, ...]
    suffixes: tuple[str, str] = ("_x", "_y")
    null_key_policy: NullKeyPolicy = NullKeyPolicy.NEVER_MATCH
    max_result_rows: int | None = 5_000_000
    max_expansion_factor: float | None = 100.0
    allow_many_to_many: bool = False

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("At least one join step is required")
        if len(self.suffixes) != 2 or self.suffixes[0] == self.suffixes[1]:
            raise ValueError("suffixes must contain two different suffixes")
        if not isinstance(self.null_key_policy, NullKeyPolicy):
            object.__setattr__(self, "null_key_policy", NullKeyPolicy(self.null_key_policy))
        if self.max_result_rows is not None and self.max_result_rows < 1:
            raise ValueError("max_result_rows must be positive or None")
        if self.max_expansion_factor is not None and self.max_expansion_factor <= 0:
            raise ValueError("max_expansion_factor must be positive or None")
        current = _table_identity(self.base_table)
        seen = {current}
        for step in self.steps:
            if _table_identity(step.left_table) != current:
                raise ValueError("Each join step must start from the current integrated table")
            right = _table_identity(step.right_table)
            if right in seen:
                raise ValueError("Repeated tables and cyclic joins are not supported")
            seen.add(right)
            current = right


@dataclass(frozen=True)
class JoinStepReport:
    relationship_id: str
    join_type: JoinType
    left_rows_before: int
    right_rows: int
    result_rows: int
    matched_rows: int
    left_unmatched_rows: int
    right_unmatched_rows: int
    matched_key_count: int
    expansion_factor: float
    left_null_key_rows: int
    right_null_key_rows: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntegrationReport:
    base_table: TableRef
    final_table: TableRef
    final_rows: int
    final_columns: int
    steps: tuple[JoinStepReport, ...]
    warnings: tuple[str, ...] = ()


@dataclass
class IntegrationResult:
    dataframe: pd.DataFrame
    report: IntegrationReport


def _table_identity(table: TableRef) -> tuple[str, str]:
    return table.source_id, table.table_name


def _key_tuple(values: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(values)


class IntegrationEngine:
    def __init__(
        self,
        graph: RelationshipGraph,
        relationship_manager: RelationshipManager,
    ) -> None:
        self.graph = graph
        self.relationship_manager = relationship_manager

    def integrate(
        self,
        config: IntegrationConfig,
        tables: Mapping[TableRef | tuple[str, str], pd.DataFrame],
    ) -> IntegrationResult:
        resolved_tables = self._normalize_tables(tables)
        base_key = _table_identity(config.base_table)
        if base_key not in resolved_tables:
            raise IntegrationValidationError(f"Base table '{base_key}' was not provided")

        current_df = resolved_tables[base_key]
        self._validate_dataframe(current_df, config.base_table)
        step_reports: list[JoinStepReport] = []
        all_warnings: list[str] = []
        current_table = config.base_table

        for step in config.steps:
            if _table_identity(step.left_table) != _table_identity(current_table):
                raise IntegrationValidationError("Join chain is not continuous")
            right_key = _table_identity(step.right_table)
            if right_key not in resolved_tables:
                raise IntegrationValidationError(f"Table '{right_key}' was not provided")
            right_df = resolved_tables[right_key]
            self._validate_dataframe(right_df, step.right_table)
            relationship_id = self._resolve_relationship(step)
            current_df, report = self._execute_step(
                left_df=current_df,
                right_df=right_df,
                step=step,
                relationship_id=relationship_id,
                config=config,
            )
            step_reports.append(report)
            all_warnings.extend(report.warnings)
            current_table = step.right_table

        report = IntegrationReport(
            base_table=config.base_table,
            final_table=current_table,
            final_rows=len(current_df),
            final_columns=len(current_df.columns),
            steps=tuple(step_reports),
            warnings=tuple(dict.fromkeys(all_warnings)),
        )
        return IntegrationResult(dataframe=current_df, report=report)

    def integrate_between(
        self,
        base_table: TableRef,
        target_table: TableRef,
        tables: Mapping[TableRef | tuple[str, str], pd.DataFrame],
        join_type: JoinType = JoinType.LEFT,
        max_depth: int = 5,
        max_paths: int = 2,
        **config_options,
    ) -> IntegrationResult:
        start = GraphNode(base_table.source_id, base_table.table_name)
        target = GraphNode(target_table.source_id, target_table.table_name)
        paths = self.graph.find_all_paths(
            start=start,
            target=target,
            max_depth=max_depth,
            max_paths=max_paths,
        )
        if not paths:
            raise RelationshipNotUsableError("No confirmed relationship path exists between the tables")
        if len(paths) > 1:
            raise AmbiguousJoinPathError(
                f"More than one relationship path exists between '{base_table.table_name}' and '{target_table.table_name}'"
            )
        path = paths[0]
        steps = tuple(
            JoinStep(
                left_table=TableRef(step.from_node.source_id, step.from_node.table_name),
                right_table=TableRef(step.to_node.source_id, step.to_node.table_name),
                key_pairs=step.key_pairs,
                join_type=join_type,
                relationship_id=step.edge.relationship_id,
            )
            for step in path.steps
        )
        config = IntegrationConfig(
            base_table=base_table,
            steps=steps,
            **config_options,
        )
        return self.integrate(config, tables)

    @staticmethod
    def _normalize_tables(
        tables: Mapping[TableRef | tuple[str, str], pd.DataFrame],
    ) -> dict[tuple[str, str], pd.DataFrame]:
        normalized: dict[tuple[str, str], pd.DataFrame] = {}
        for key, dataframe in tables.items():
            if isinstance(key, TableRef):
                identity = _table_identity(key)
            elif isinstance(key, tuple) and len(key) == 2:
                identity = key
            else:
                raise TypeError("Table mapping keys must be TableRef or (source_id, table_name)")
            if identity in normalized:
                raise IntegrationValidationError(f"Duplicate table mapping for '{identity}'")
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(f"Table '{identity}' must be a pandas DataFrame")
            normalized[identity] = dataframe
        return normalized

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame, table: TableRef) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Table '{_table_identity(table)}' must be a pandas DataFrame")
        if dataframe.columns.has_duplicates:
            raise IntegrationValidationError(
                f"Table '{table.table_name}' contains duplicate column names"
            )

    def _resolve_relationship(self, step: JoinStep) -> str:
        if not step.relationship_id:
            raise RelationshipNotUsableError("A confirmed relationship_id is required for every join step")
        relationship = self.relationship_manager.get_relationship(step.relationship_id)
        if relationship.status != RelationshipStatus.CONFIRMED or relationship.is_stale:
            raise RelationshipNotUsableError("Only confirmed and non-stale relationships can be used")

        left_identity = _table_identity(step.left_table)
        right_identity = _table_identity(step.right_table)
        relationship_left = _table_identity(relationship.left_table)
        relationship_right = _table_identity(relationship.right_table)

        if left_identity == relationship_left and right_identity == relationship_right:
            expected_pairs = tuple(relationship.key_pairs)
        elif left_identity == relationship_right and right_identity == relationship_left:
            expected_pairs = tuple(
                ColumnPair(pair.right_column, pair.left_column)
                for pair in relationship.key_pairs
            )
        else:
            raise RelationshipNotUsableError("Relationship endpoints do not match the requested join")

        if tuple(step.key_pairs) != expected_pairs:
            raise RelationshipNotUsableError("Join keys do not match the confirmed relationship")

        edge = self.graph.get_relationship(step.relationship_id)
        if edge is None:
            raise RelationshipNotUsableError("Relationship is missing from the confirmed relationship graph")
        return step.relationship_id

    @staticmethod
    def _estimate_join_stats(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_on: list[str],
        right_on: list[str],
        join_type: JoinType,
        null_key_policy: NullKeyPolicy,
    ) -> tuple[int, int, int, int]:
        key_names = [f"__key_{index}" for index in range(len(left_on))]
        left_keys = left_df[left_on].copy()
        right_keys = right_df[right_on].copy()
        left_keys.columns = key_names
        right_keys.columns = key_names
        if null_key_policy == NullKeyPolicy.NEVER_MATCH:
            left_keys = left_keys.loc[left_keys.notna().all(axis=1)]
            right_keys = right_keys.loc[right_keys.notna().all(axis=1)]
        left_counts = left_keys.groupby(key_names, dropna=False, sort=False).size().rename("left_count").reset_index()
        right_counts = right_keys.groupby(key_names, dropna=False, sort=False).size().rename("right_count").reset_index()
        matched = left_counts.merge(right_counts, on=key_names, how="inner", sort=False)
        if matched.empty:
            matched_rows = 0
            left_matched_rows = 0
            right_matched_rows = 0
        else:
            products = matched["left_count"] * matched["right_count"]
            matched_rows = int(products.sum())
            left_matched_rows = int(matched["left_count"].sum())
            right_matched_rows = int(matched["right_count"].sum())
        left_unmatched = len(left_df) - left_matched_rows
        right_unmatched = len(right_df) - right_matched_rows
        if join_type == JoinType.INNER:
            result_rows = matched_rows
        elif join_type == JoinType.LEFT:
            result_rows = matched_rows + left_unmatched
        elif join_type == JoinType.RIGHT:
            result_rows = matched_rows + right_unmatched
        else:
            result_rows = matched_rows + left_unmatched + right_unmatched
        return result_rows, left_unmatched, right_unmatched, len(matched)

    def _execute_step(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        step: JoinStep,
        relationship_id: str,
        config: IntegrationConfig,
    ) -> tuple[pd.DataFrame, JoinStepReport]:
        left_on = [pair.left_column for pair in step.key_pairs]
        right_on = [pair.right_column for pair in step.key_pairs]
        for column in left_on:
            if column not in left_df.columns:
                raise IntegrationValidationError(f"Left join column '{column}' does not exist")
        for column in right_on:
            if column not in right_df.columns:
                raise IntegrationValidationError(f"Right join column '{column}' does not exist")
        for left_column, right_column in zip(left_on, right_on):
            if not is_dtype_equal(left_df[left_column].dtype, right_df[right_column].dtype):
                raise IntegrationValidationError(
                    f"Join key dtypes differ: '{left_column}' ({left_df[left_column].dtype}) and '{right_column}' ({right_df[right_column].dtype})"
                )

        warnings: list[str] = []
        left_null_mask = left_df[left_on].isna().any(axis=1)
        right_null_mask = right_df[right_on].isna().any(axis=1)
        if not config.allow_many_to_many:
            cardinality_left = left_df.loc[~left_null_mask] if config.null_key_policy == NullKeyPolicy.NEVER_MATCH else left_df
            cardinality_right = right_df.loc[~right_null_mask] if config.null_key_policy == NullKeyPolicy.NEVER_MATCH else right_df
            left_unique = not cardinality_left.duplicated(subset=left_on).any()
            right_unique = not cardinality_right.duplicated(subset=right_on).any()
            if not left_unique and not right_unique:
                raise JoinExpansionError(
                    "Many-to-many JOIN detected; set allow_many_to_many=True to permit it"
                )
        left_null_count = int(left_null_mask.sum())
        right_null_count = int(right_null_mask.sum())
        estimated_rows, estimated_left_unmatched, estimated_right_unmatched, estimated_matched_keys = self._estimate_join_stats(
            left_df=left_df,
            right_df=right_df,
            left_on=left_on,
            right_on=right_on,
            join_type=step.join_type,
            null_key_policy=config.null_key_policy,
        )
        left_rows_before = len(left_df)
        estimated_expansion = (
            estimated_rows / left_rows_before
            if left_rows_before
            else (1.0 if estimated_rows == 0 else float("inf"))
        )
        if config.max_expansion_factor is not None and estimated_expansion > config.max_expansion_factor:
            raise JoinExpansionError(
                f"Estimated JOIN expansion factor {estimated_expansion:.3f} exceeds limit {config.max_expansion_factor:.3f}"
            )
        if config.max_result_rows is not None and estimated_rows > config.max_result_rows:
            raise JoinExpansionError(
                f"Estimated JOIN result has {estimated_rows} rows, exceeding limit {config.max_result_rows}"
            )
        if left_null_count or right_null_count:
            warnings.append(
                f"Join keys contain nulls: {left_null_count} left rows and {right_null_count} right rows"
            )

        left_work = left_df.copy(deep=False)
        right_work = right_df.copy(deep=False)
        restore_values: list[object] = []
        if config.null_key_policy == NullKeyPolicy.NEVER_MATCH and (left_null_count or right_null_count):
            left_work = left_df.copy()
            right_work = right_df.copy()
            for left_column, right_column in zip(left_on, right_on):
                left_sentinel = object()
                right_sentinel = object()
                left_work[left_column] = left_work[left_column].astype(object)
                right_work[right_column] = right_work[right_column].astype(object)
                left_work.loc[left_null_mask, left_column] = left_sentinel
                right_work.loc[right_null_mask, right_column] = right_sentinel
                restore_values.extend((left_sentinel, right_sentinel))

        indicator = "__quantlyra_merge_status__"
        while indicator in left_work.columns or indicator in right_work.columns:
            indicator += "_"
        merged = left_work.merge(
            right_work,
            how=step.join_type.value,
            left_on=left_on,
            right_on=right_on,
            suffixes=config.suffixes,
            indicator=indicator,
            sort=False,
        )
        status = merged[indicator]
        matched_mask = status == "both"
        matched_rows = int(matched_mask.sum())
        merged = merged.drop(columns=[indicator])

        if restore_values:
            for column in merged.columns:
                if merged[column].dtype == object:
                    merged[column] = merged[column].map(
                        lambda value: pd.NA if any(value is sentinel for sentinel in restore_values) else value
                    )

        left_rows = len(left_df)
        result_rows = len(merged)
        expansion = result_rows / left_rows if left_rows else (1.0 if result_rows == 0 else float("inf"))
        if expansion > 1:
            warnings.append(f"JOIN expanded the left table by a factor of {expansion:.3f}")
        if config.max_expansion_factor is not None and expansion > config.max_expansion_factor:
            raise JoinExpansionError(
                f"JOIN expansion factor {expansion:.3f} exceeds limit {config.max_expansion_factor:.3f}"
            )
        if config.max_result_rows is not None and result_rows > config.max_result_rows:
            raise JoinExpansionError(
                f"JOIN result has {result_rows} rows, exceeding limit {config.max_result_rows}"
            )
        report = JoinStepReport(
            relationship_id=relationship_id,
            join_type=step.join_type,
            left_rows_before=left_rows,
            right_rows=len(right_df),
            result_rows=result_rows,
            matched_rows=matched_rows,
            left_unmatched_rows=estimated_left_unmatched,
            right_unmatched_rows=estimated_right_unmatched,
            matched_key_count=estimated_matched_keys,
            expansion_factor=expansion,
            left_null_key_rows=left_null_count,
            right_null_key_rows=right_null_count,
            warnings=tuple(warnings),
        )
        return merged, report
