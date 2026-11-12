
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

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
from quantlyra.data_quality.checker import check_dataframe


def test_check_dataframe_without_issues():
    """DataFrame без нарушений должен возвращать валидный отчёт."""
    dataframe = pd.DataFrame({
        "age": [20, 25, 30],
        "city": ["Moscow", "London", "Paris"],
    })

    rules = [
        MissingValuesRule(rule_id="missing", column="age"),
        NumericRangeRule(
            rule_id="range",
            column="age",
            min_value=18,
            max_value=100,
        ),
        CategoricalRule(
            rule_id="city",
            column="city",
            allowed_values={"Moscow", "London", "Paris"},
        ),
    ]

    report = check_dataframe(dataframe, rules, "users")

    assert report.table_name == "users"
    assert report.row_count == 3
    assert report.checked_rules == 3
    assert report.issues == []
    assert report.is_valid is True


def test_check_dataframe_with_multiple_issues():
    """Несколько правил должны обнаруживать разные нарушения."""
    dataframe = pd.DataFrame({
        "age": [20, None, 150, 25],
        "city": ["Moscow", "London", "Unknown", "Paris"],
    })

    rules = [
        MissingValuesRule(
            rule_id="missing_age",
            column="age",
            max_missing_percentage=0,
        ),
        NumericRangeRule(
            rule_id="age_range",
            column="age",
            min_value=0,
            max_value=100,
        ),
        CategoricalRule(
            rule_id="allowed_cities",
            column="city",
            allowed_values={"Moscow", "London", "Paris"},
        ),
    ]

    report = check_dataframe(dataframe, rules)

    assert report.checked_rules == 3
    assert len(report.issues) == 3
    assert report.is_valid is False

    assert report.error_count == 3
    assert report.warning_count == 0


def test_missing_values_issue_details():
    """Проверяем подробности нарушения пропущенных значений."""
    dataframe = pd.DataFrame({
        "price": [100, None, 200, None],
    })

    rules = [
        MissingValuesRule(
            rule_id="missing_price",
            column="price",
            max_missing_percentage=25,
            severity=Severity.WARNING,
        )
    ]

    report = check_dataframe(dataframe, rules)

    issue = report.issues[0]

    assert issue.rule_id == "missing_price"
    assert issue.severity == Severity.WARNING
    assert issue.affected_rows == 2
    assert issue.row_indices == [1, 3]
    assert issue.column == "price"


def test_duplicate_rows_detection():
    """Все строки из дублирующихся групп должны обнаруживаться."""
    dataframe = pd.DataFrame({
        "id": [1, 2, 2, 3, 3],
        "name": ["A", "B", "B", "C", "C"],
    })

    rules = [
        DuplicateRowsRule(rule_id="duplicates")
    ]

    report = check_dataframe(dataframe, rules)

    issue = report.issues[0]

    assert issue.affected_rows == 4
    assert issue.row_indices == [1, 2, 3, 4]


def test_duplicate_rows_by_selected_columns():
    """Дубликаты должны определяться по выбранным столбцам."""
    dataframe = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "C", "D"],
    })

    rules = [
        DuplicateRowsRule(
            rule_id="duplicate_id",
            columns=["id"],
        )
    ]

    report = check_dataframe(dataframe, rules)

    issue = report.issues[0]

    assert issue.affected_rows == 2
    assert issue.row_indices == [1, 2]


def test_numeric_range_detection():
    """Значения за пределами диапазона должны обнаруживаться."""
    dataframe = pd.DataFrame({
        "temperature": [-10, 0, 25, 100, 120],
    })

    rules = [
        NumericRangeRule(
            rule_id="temperature_range",
            column="temperature",
            min_value=0,
            max_value=100,
        )
    ]

    report = check_dataframe(dataframe, rules)

    issue = report.issues[0]

    assert issue.affected_rows == 2
    assert issue.row_indices == [0, 4]


def test_numeric_range_ignores_missing_values():
    """Пропуски не должны считаться нарушением диапазона."""
    dataframe = pd.DataFrame({
        "price": [100, None, 200],
    })

    rules = [
        NumericRangeRule(
            rule_id="price_range",
            column="price",
            min_value=0,
            max_value=500,
        )
    ]

    report = check_dataframe(dataframe, rules)

    assert report.issues == []


def test_categorical_rule_detection():
    """Неизвестные категории должны обнаруживаться."""
    dataframe = pd.DataFrame({
        "status": ["active", "inactive", "unknown", "active"],
    })

    rules = [
        CategoricalRule(
            rule_id="status_check",
            column="status",
            allowed_values={"active", "inactive"},
        )
    ]

    report = check_dataframe(dataframe, rules)

    issue = report.issues[0]

    assert issue.affected_rows == 1
    assert issue.row_indices == [2]


def test_categorical_rule_ignores_missing_values():
    """Пропуски должны обрабатываться отдельным правилом."""
    dataframe = pd.DataFrame({
        "status": ["active", None, "inactive"],
    })

    rules = [
        CategoricalRule(
            rule_id="status_check",
            column="status",
            allowed_values={"active", "inactive"},
        )
    ]

    report = check_dataframe(dataframe, rules)

    assert report.issues == []


@pytest.mark.parametrize(
    "expected_dtype",
    ["numeric", "integer", "float", "string", "boolean"],
)
def test_data_type_rule_valid_types(expected_dtype):
    """Корректные типы данных не должны вызывать нарушения."""
    dataframes = {
        "numeric": pd.DataFrame({"value": [1, 2, 3]}),
        "integer": pd.DataFrame({"value": [1, 2, 3]}),
        "float": pd.DataFrame({"value": [1.0, 2.0, 3.0]}),
        "string": pd.DataFrame({"value": ["a", "b", "c"]}),
        "boolean": pd.DataFrame({"value": [True, False, True]}),
    }

    dataframe = dataframes[expected_dtype]

    rules = [
        DataTypeRule(
            rule_id="dtype_check",
            column="value",
            expected_dtype=expected_dtype,
        )
    ]

    report = check_dataframe(dataframe, rules)

    assert report.issues == []
    assert report.is_valid is True


def test_data_type_rule_invalid_type():
    """Несоответствие типа должно создавать нарушение."""
    dataframe = pd.DataFrame({
        "age": ["20", "30", "40"],
    })

    rules = [
        DataTypeRule(
            rule_id="age_type",
            column="age",
            expected_dtype="integer",
        )
    ]

    report = check_dataframe(dataframe, rules)

    assert len(report.issues) == 1
    assert report.issues[0].rule_id == "age_type"


def test_unique_values_rule_detection():
    """Повторяющиеся значения должны обнаруживаться."""
    dataframe = pd.DataFrame({
        "email": [
            "a@test.com",
            "b@test.com",
            "a@test.com",
            "c@test.com",
        ],
    })

    rules = [
        UniqueValuesRule(
            rule_id="unique_email",
            column="email",
        )
    ]

    report = check_dataframe(dataframe, rules)

    issue = report.issues[0]

    assert issue.affected_rows == 2
    assert issue.row_indices == [0, 2]


def test_unique_values_ignores_missing_values():
    """Пропущенные значения не должны считаться дубликатами."""
    dataframe = pd.DataFrame({
        "email": ["a@test.com", None, None, "b@test.com"],
    })

    rules = [
        UniqueValuesRule(
            rule_id="unique_email",
            column="email",
        )
    ]

    report = check_dataframe(dataframe, rules)

    assert report.issues == []


def test_dataframe_is_not_modified():
    """Проверки качества не должны изменять исходный DataFrame."""
    dataframe = pd.DataFrame({
        "age": [20, None, 150],
        "city": ["Moscow", "London", "Unknown"],
    })

    original = dataframe.copy(deep=True)

    rules = [
        MissingValuesRule(
            rule_id="missing",
            column="age",
        ),
        NumericRangeRule(
            rule_id="range",
            column="age",
            min_value=0,
            max_value=100,
        ),
        CategoricalRule(
            rule_id="city",
            column="city",
            allowed_values={"Moscow", "London"},
        ),
    ]

    check_dataframe(dataframe, rules)

    assert_frame_equal(dataframe, original)


def test_missing_column_raises_error():
    """Отсутствующий столбец должен вызывать понятную ошибку."""
    dataframe = pd.DataFrame({
        "age": [20, 30],
    })

    rules = [
        MissingValuesRule(
            rule_id="missing",
            column="salary",
        )
    ]

    with pytest.raises(ValueError, match="salary"):
        check_dataframe(dataframe, rules)


def test_invalid_missing_percentage_raises_error():
    """Процент пропусков должен находиться в диапазоне от 0 до 100."""
    dataframe = pd.DataFrame({
        "age": [20, None],
    })

    rules = [
        MissingValuesRule(
            rule_id="missing",
            column="age",
            max_missing_percentage=150,
        )
    ]

    with pytest.raises(ValueError):
        check_dataframe(dataframe, rules)


def test_invalid_numeric_range_raises_error():
    """Минимум не может быть больше максимума."""
    dataframe = pd.DataFrame({
        "age": [20, 30],
    })

    rules = [
        NumericRangeRule(
            rule_id="range",
            column="age",
            min_value=100,
            max_value=0,
        )
    ]

    with pytest.raises(ValueError):
        check_dataframe(dataframe, rules)


def test_numeric_rule_on_string_column_raises_error():
    """Числовое правило нельзя применять к строковому столбцу."""
    dataframe = pd.DataFrame({
        "age": ["20", "30", "40"],
    })

    rules = [
        NumericRangeRule(
            rule_id="range",
            column="age",
            min_value=0,
            max_value=100,
        )
    ]

    with pytest.raises(TypeError):
        check_dataframe(dataframe, rules)


def test_logical_rule_not_supported_yet():
    """Нереализованное логическое правило должно явно отклоняться."""
    dataframe = pd.DataFrame({
        "price": [100, 200],
    })

    rules = [
        LogicalRule(
            rule_id="logical",
            columns=["price"],
            condition="price > 0",
        )
    ]

    with pytest.raises(NotImplementedError):
        check_dataframe(dataframe, rules)


def test_report_severity_counts():
    """Счётчики ошибок и предупреждений должны работать корректно."""
    dataframe = pd.DataFrame({
        "age": [None, 150, 20],
    })

    rules = [
        MissingValuesRule(
            rule_id="missing",
            column="age",
            severity=Severity.ERROR,
        ),
        NumericRangeRule(
            rule_id="range",
            column="age",
            min_value=0,
            max_value=100,
            severity=Severity.WARNING,
        ),
    ]

    report = check_dataframe(dataframe, rules)

    assert report.error_count == 1
    assert report.warning_count == 1
    assert report.info_count == 0
    assert report.is_valid is False


def test_warning_does_not_make_report_invalid():
    """Предупреждения не должны автоматически делать отчёт невалидным."""
    dataframe = pd.DataFrame({
        "age": [150, 20],
    })

    rules = [
        NumericRangeRule(
            rule_id="range",
            column="age",
            min_value=0,
            max_value=100,
            severity=Severity.WARNING,
        )
    ]

    report = check_dataframe(dataframe, rules)

    assert report.warning_count == 1
    assert report.error_count == 0
    assert report.is_valid is True


def test_empty_dataframe():
    """Пустой DataFrame должен обрабатываться без ошибок."""
    dataframe = pd.DataFrame({
        "age": pd.Series(dtype="int64"),
    })

    rules = [
        MissingValuesRule(
            rule_id="missing",
            column="age",
        ),
        NumericRangeRule(
            rule_id="range",
            column="age",
            min_value=0,
            max_value=100,
        ),
    ]

    report = check_dataframe(dataframe, rules)

    assert report.row_count == 0
    assert report.checked_rules == 2
    assert report.issues == []
    assert report.is_valid is True


def test_non_default_dataframe_index():
    """В отчёте должны сохраняться исходные индексы DataFrame."""
    dataframe = pd.DataFrame(
        {"age": [20, 150, 30]},
        index=[10, 20, 30],
    )

    rules = [
        NumericRangeRule(
            rule_id="range",
            column="age",
            min_value=0,
            max_value=100,
        )
    ]

    report = check_dataframe(dataframe, rules)

    assert report.issues[0].row_indices == [20]
