
import pytest

from quantlyra.data_quality.models import Severity
from quantlyra.data_quality.rules import (
    MissingValuesRule,
    DuplicateRowsRule,
    NumericRangeRule,
    CategoricalRule,
    DataTypeRule,
    UniqueValuesRule,
    LogicalRule,
)


def test_missing_values_rule():
    rule = MissingValuesRule(
        rule_id="missing_price",
        column="price",
        max_missing_percentage=5.0,
    )

    assert rule.rule_id == "missing_price"
    assert rule.column == "price"
    assert rule.max_missing_percentage == 5.0
    assert rule.severity == Severity.ERROR


def test_duplicate_rows_rule():
    rule = DuplicateRowsRule(
        rule_id="duplicate_orders",
        columns=["order_id", "product_id"],
    )

    assert rule.rule_id == "duplicate_orders"
    assert rule.columns == ["order_id", "product_id"]
    assert rule.severity == Severity.ERROR


def test_numeric_range_rule():
    rule = NumericRangeRule(
        rule_id="price_range",
        column="price",
        min_value=0.0,
        max_value=1000000.0,
    )

    assert rule.column == "price"
    assert rule.min_value == 0.0
    assert rule.max_value == 1000000.0


def test_categorical_rule():
    rule = CategoricalRule(
        rule_id="order_status",
        column="status",
        allowed_values={"new", "processing", "completed"},
    )

    assert rule.column == "status"
    assert rule.allowed_values == {
        "new", "processing", "completed"
    }


def test_data_type_rule():
    rule = DataTypeRule(
        rule_id="quantity_type",
        column="quantity",
        expected_dtype="int64",
    )

    assert rule.column == "quantity"
    assert rule.expected_dtype == "int64"


def test_unique_values_rule():
    rule = UniqueValuesRule(
        rule_id="unique_order_id",
        column="order_id",
    )

    assert rule.column == "order_id"
    assert rule.severity == Severity.ERROR


def test_logical_rule():
    rule = LogicalRule(
        rule_id="date_consistency",
        columns=["start_date", "end_date"],
        condition="start_date <= end_date",
    )

    assert rule.columns == ["start_date", "end_date"]
    assert rule.condition == "start_date <= end_date"


def test_default_severity():
    rule = MissingValuesRule(
        rule_id="missing_check",
        column="price",
    )

    assert rule.severity == Severity.ERROR


def test_custom_severity():
    rule = MissingValuesRule(
        rule_id="missing_check",
        column="price",
        severity=Severity.WARNING,
    )

    assert rule.severity == Severity.WARNING


def test_default_description():
    rule = MissingValuesRule(
        rule_id="missing_check",
        column="price",
    )

    assert rule.description == ""


def test_independent_mutable_defaults():
    first_categorical = CategoricalRule(
        rule_id="category_1",
        column="status",
    )

    second_categorical = CategoricalRule(
        rule_id="category_2",
        column="status",
    )

    first_categorical.allowed_values.add("new")

    assert "new" in first_categorical.allowed_values
    assert "new" not in second_categorical.allowed_values

    first_logical = LogicalRule(
        rule_id="logical_1",
    )

    second_logical = LogicalRule(
        rule_id="logical_2",
    )

    first_logical.columns.append("price")

    assert first_logical.columns == ["price"]
    assert second_logical.columns == []


def test_duplicate_rows_rule_default_columns():
    rule = DuplicateRowsRule(
        rule_id="duplicates",
    )

    assert rule.columns is None


def test_numeric_range_rule_without_limits():
    rule = NumericRangeRule(
        rule_id="numeric_range",
        column="price",
    )

    assert rule.min_value is None
    assert rule.max_value is None
