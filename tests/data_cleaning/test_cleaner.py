import pandas as pd
import pytest

from quantlyra.data_cleaning.models import CleaningOperation
from quantlyra.data_cleaning.cleaner import clean_dataframe


def test_clean_dataframe_with_multiple_operations():
    dataframe = pd.DataFrame({
        "name": [" Alex ", "John", "John"],
        "age": [20, None, 30],
    })

    operations = [
        CleaningOperation(
            operation_type="drop_duplicates",
            parameters={"subset": ["name"]},
        ),
        CleaningOperation(
            operation_type="fill_missing_values",
            column="age",
            parameters={"strategy": "median"},
        ),
        CleaningOperation(
            operation_type="normalize_text",
            column="name",
            parameters={"strip": True, "lowercase": True},
        ),
    ]

    result = clean_dataframe(dataframe, operations)

    assert len(result.cleaned_dataframe) == 2
    assert result.cleaned_dataframe["name"].tolist() == [
        "alex",
        "john",
    ]
    assert result.cleaned_dataframe["age"].notna().all()
    assert len(result.history) == 3


def test_clean_dataframe_with_empty_operations():
    dataframe = pd.DataFrame({
        "name": ["Alex", "John"],
        "age": [20, 30],
    })

    result = clean_dataframe(dataframe, [])

    pd.testing.assert_frame_equal(
        result.original_dataframe,
        dataframe,
    )
    pd.testing.assert_frame_equal(
        result.cleaned_dataframe,
        dataframe,
    )
    assert result.history == []


def test_clean_dataframe_preserves_original_dataframe():
    dataframe = pd.DataFrame({
        "name": [" Alex ", "John"],
    })

    original = dataframe.copy(deep=True)

    operations = [
        CleaningOperation(
            operation_type="normalize_text",
            column="name",
            parameters={"strip": True},
        ),
    ]

    clean_dataframe(dataframe, operations)

    pd.testing.assert_frame_equal(dataframe, original)


def test_clean_dataframe_records_history():
    dataframe = pd.DataFrame({
        "name": ["Alex", "Alex", "John"],
    })

    operations = [
        CleaningOperation(
            operation_type="drop_duplicates",
        ),
        CleaningOperation(
            operation_type="normalize_text",
            column="name",
            parameters={"lowercase": True},
        ),
    ]

    result = clean_dataframe(dataframe, operations)

    assert len(result.history) == 2
    assert result.history[0].operation.operation_type == "drop_duplicates"
    assert result.history[0].affected_rows == 1
    assert result.history[1].operation.operation_type == "normalize_text"


def test_clean_dataframe_rejects_unknown_operation():
    dataframe = pd.DataFrame({"age": [20, 30]})

    operations = [
        CleaningOperation(
            operation_type="unknown_operation",
        ),
    ]

    with pytest.raises(ValueError, match="Unsupported cleaning operation"):
        clean_dataframe(dataframe, operations)


def test_clean_dataframe_rejects_non_dataframe():
    with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
        clean_dataframe([1, 2, 3], [])


def test_clean_dataframe_requires_column():
    dataframe = pd.DataFrame({"age": [20, 30]})

    operations = [
        CleaningOperation(
            operation_type="normalize_text",
        ),
    ]

    with pytest.raises(ValueError, match="[Cc]olumn is required"):
        clean_dataframe(dataframe, operations)


def test_clean_dataframe_stops_on_operation_error():
    dataframe = pd.DataFrame({
        "age": [20, 30],
    })

    original = dataframe.copy(deep=True)

    operations = [
        CleaningOperation(
            operation_type="drop_duplicates",
        ),
        CleaningOperation(
            operation_type="normalize_text",
            column="age",
        ),
    ]

    with pytest.raises(TypeError):
        clean_dataframe(dataframe, operations)

    pd.testing.assert_frame_equal(dataframe, original)
