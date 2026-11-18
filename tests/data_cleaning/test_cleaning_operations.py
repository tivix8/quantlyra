
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from quantlyra.data_cleaning.operations import (
    drop_missing_values,
    fill_missing_values,
    drop_duplicates,
    convert_dtype,
    normalize_text,
    clip_numeric_values,
)


# Missing values

def test_drop_missing_values():
    dataframe = pd.DataFrame({
        "name": ["Alex", None, "John"],
        "age": [20, 25, 30],
    })

    result, affected = drop_missing_values(dataframe)

    assert len(result) == 2
    assert affected == 1
    assert result["name"].isna().sum() == 0


def test_drop_missing_values_by_selected_columns():
    dataframe = pd.DataFrame({
        "name": ["Alex", None, "John"],
        "age": [None, 25, 30],
    })

    result, affected = drop_missing_values(
        dataframe,
        columns=["name"],
    )

    assert affected == 1
    assert len(result) == 2


def test_drop_missing_values_how_all():
    dataframe = pd.DataFrame({
        "name": ["Alex", None, "John"],
        "age": [20, None, 30],
    })

    result, affected = drop_missing_values(
        dataframe,
        how="all",
    )

    assert affected == 1
    assert len(result) == 2


def test_drop_missing_values_invalid_how():
    dataframe = pd.DataFrame({"age": [20, None]})

    with pytest.raises(ValueError):
        drop_missing_values(dataframe, how="invalid")


def test_drop_missing_values_missing_column():
    dataframe = pd.DataFrame({"age": [20, 30]})

    with pytest.raises(ValueError, match="salary"):
        drop_missing_values(
            dataframe,
            columns=["salary"],
        )


# Fill missing values

def test_fill_missing_values_constant():
    dataframe = pd.DataFrame({
        "age": [20, None, 30],
    })

    result, affected = fill_missing_values(
        dataframe,
        column="age",
        strategy="constant",
        value=25,
    )

    assert result["age"].tolist() == [20, 25, 30]
    assert affected == 1


def test_fill_missing_values_mean():
    dataframe = pd.DataFrame({
        "price": [10.0, 20.0, None, 30.0],
    })

    result, affected = fill_missing_values(
        dataframe,
        column="price",
        strategy="mean",
    )

    assert result["price"].iloc[2] == 20.0
    assert affected == 1


def test_fill_missing_values_median():
    dataframe = pd.DataFrame({
        "price": [10.0, 20.0, None, 100.0],
    })

    result, affected = fill_missing_values(
        dataframe,
        column="price",
        strategy="median",
    )

    assert result["price"].iloc[2] == 20.0
    assert affected == 1


def test_fill_missing_values_mode():
    dataframe = pd.DataFrame({
        "city": ["Moscow", "London", None, "Moscow"],
    })

    result, affected = fill_missing_values(
        dataframe,
        column="city",
        strategy="mode",
    )

    assert result["city"].iloc[2] == "Moscow"
    assert affected == 1


def test_fill_missing_values_no_missing_values():
    dataframe = pd.DataFrame({
        "age": [20, 30, 40],
    })

    result, affected = fill_missing_values(
        dataframe,
        column="age",
        strategy="mean",
    )

    assert affected == 0
    assert_frame_equal(result, dataframe)


def test_fill_missing_values_invalid_strategy():
    dataframe = pd.DataFrame({
        "age": [20, None],
    })

    with pytest.raises(ValueError):
        fill_missing_values(
            dataframe,
            column="age",
            strategy="unknown",
        )


def test_fill_missing_values_mean_on_string_column():
    dataframe = pd.DataFrame({
        "city": ["Moscow", None, "London"],
    })

    with pytest.raises(TypeError):
        fill_missing_values(
            dataframe,
            column="city",
            strategy="mean",
        )


def test_fill_missing_values_without_constant_value():
    dataframe = pd.DataFrame({
        "age": [20, None],
    })

    with pytest.raises(ValueError):
        fill_missing_values(
            dataframe,
            column="age",
            strategy="constant",
        )


def test_fill_missing_values_all_values_missing():
    dataframe = pd.DataFrame({
        "city": [None, None],
    })

    with pytest.raises(ValueError):
        fill_missing_values(
            dataframe,
            column="city",
            strategy="mode",
        )


# Duplicate rows

def test_drop_duplicates():
    dataframe = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "B", "C"],
    })

    result, affected = drop_duplicates(dataframe)

    assert len(result) == 3
    assert affected == 1


def test_drop_duplicates_by_subset():
    dataframe = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "C", "D"],
    })

    result, affected = drop_duplicates(
        dataframe,
        subset=["id"],
    )

    assert affected == 1
    assert len(result) == 3


def test_drop_duplicates_keep_last():
    dataframe = pd.DataFrame({
        "id": [1, 2, 2],
        "name": ["A", "B", "C"],
    })

    result, affected = drop_duplicates(
        dataframe,
        subset=["id"],
        keep="last",
    )

    assert affected == 1
    assert result.iloc[1]["name"] == "C"


def test_drop_duplicates_keep_false():
    dataframe = pd.DataFrame({
        "id": [1, 2, 2, 3],
    })

    result, affected = drop_duplicates(
        dataframe,
        keep=False,
    )

    assert affected == 2
    assert result["id"].tolist() == [1, 3]


def test_drop_duplicates_invalid_keep():
    dataframe = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(ValueError):
        drop_duplicates(dataframe, keep="invalid")


# Data type conversion

def test_convert_dtype_to_float():
    dataframe = pd.DataFrame({
        "price": ["10.5", "20.5", "30.5"],
    })

    result, affected = convert_dtype(
        dataframe,
        column="price",
        dtype="float",
    )

    assert pd.api.types.is_float_dtype(result["price"])
    assert affected == 3


def test_convert_dtype_to_int():
    dataframe = pd.DataFrame({
        "age": ["20", "30", "40"],
    })

    result, affected = convert_dtype(
        dataframe,
        column="age",
        dtype="int",
    )

    assert pd.api.types.is_integer_dtype(result["age"])
    assert result["age"].tolist() == [20, 30, 40]


def test_convert_dtype_to_string():
    dataframe = pd.DataFrame({
        "id": [100, 200, 300],
    })

    result, affected = convert_dtype(
        dataframe,
        column="id",
        dtype="string",
    )

    assert pd.api.types.is_string_dtype(result["id"])
    assert result["id"].tolist() == ["100", "200", "300"]


def test_convert_dtype_to_datetime():
    dataframe = pd.DataFrame({
        "date": ["2025-01-01", "2025-02-01"],
    })

    result, affected = convert_dtype(
        dataframe,
        column="date",
        dtype="datetime",
    )

    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert affected == 2


def test_convert_dtype_coerce_invalid_values():
    dataframe = pd.DataFrame({
        "price": ["10", "invalid", "30"],
    })

    result, affected = convert_dtype(
        dataframe,
        column="price",
        dtype="float",
        errors="coerce",
    )

    assert pd.isna(result["price"].iloc[1])
    assert affected == 1


def test_convert_dtype_invalid_type():
    dataframe = pd.DataFrame({
        "age": ["20", "30"],
    })

    with pytest.raises(ValueError):
        convert_dtype(
            dataframe,
            column="age",
            dtype="unknown",
        )


def test_convert_dtype_invalid_errors():
    dataframe = pd.DataFrame({
        "age": ["20", "30"],
    })

    with pytest.raises(ValueError):
        convert_dtype(
            dataframe,
            column="age",
            dtype="int",
            errors="ignore",
        )


def test_convert_dtype_int_rejects_fractional_values():
    dataframe = pd.DataFrame({
        "age": ["20.5", "30"],
    })

    with pytest.raises(ValueError):
        convert_dtype(
            dataframe,
            column="age",
            dtype="int",
        )


# Text normalization

def test_normalize_text_strip():
    dataframe = pd.DataFrame({
        "city": [" Moscow ", " London ", " Paris "],
    })

    result, affected = normalize_text(
        dataframe,
        column="city",
    )

    assert result["city"].tolist() == [
        "Moscow",
        "London",
        "Paris",
    ]
    assert affected == 3


def test_normalize_text_lowercase():
    dataframe = pd.DataFrame({
        "city": ["MOSCOW", "LONDON"],
    })

    result, affected = normalize_text(
        dataframe,
        column="city",
        lowercase=True,
    )

    assert result["city"].tolist() == ["moscow", "london"]
    assert affected == 2


def test_normalize_text_uppercase():
    dataframe = pd.DataFrame({
        "city": ["moscow", "london"],
    })

    result, affected = normalize_text(
        dataframe,
        column="city",
        uppercase=True,
    )

    assert result["city"].tolist() == ["MOSCOW", "LONDON"]
    assert affected == 2


def test_normalize_text_collapse_whitespace():
    dataframe = pd.DataFrame({
        "city": ["New     York", "Los  Angeles"],
    })

    result, affected = normalize_text(
        dataframe,
        column="city",
        collapse_whitespace=True,
    )

    assert result["city"].tolist() == [
        "New York",
        "Los Angeles",
    ]
    assert affected == 2


def test_normalize_text_invalid_options():
    dataframe = pd.DataFrame({
        "city": ["Moscow"],
    })

    with pytest.raises(ValueError):
        normalize_text(
            dataframe,
            column="city",
            lowercase=True,
            uppercase=True,
        )


def test_normalize_text_non_string_values():
    dataframe = pd.DataFrame({
        "city": ["Moscow", 123],
    })

    with pytest.raises(TypeError):
        normalize_text(dataframe, column="city")


# Numeric clipping

def test_clip_numeric_values():
    dataframe = pd.DataFrame({
        "price": [-10, 20, 150],
    })

    result, affected = clip_numeric_values(
        dataframe,
        column="price",
        min_value=0,
        max_value=100,
    )

    assert result["price"].tolist() == [0, 20, 100]
    assert affected == 2


def test_clip_numeric_values_min_only():
    dataframe = pd.DataFrame({
        "price": [-10, 20, 30],
    })

    result, affected = clip_numeric_values(
        dataframe,
        column="price",
        min_value=0,
    )

    assert result["price"].tolist() == [0, 20, 30]
    assert affected == 1


def test_clip_numeric_values_max_only():
    dataframe = pd.DataFrame({
        "price": [10, 20, 150],
    })

    result, affected = clip_numeric_values(
        dataframe,
        column="price",
        max_value=100,
    )

    assert result["price"].tolist() == [10, 20, 100]
    assert affected == 1


def test_clip_numeric_values_invalid_range():
    dataframe = pd.DataFrame({
        "price": [10, 20, 30],
    })

    with pytest.raises(ValueError):
        clip_numeric_values(
            dataframe,
            column="price",
            min_value=100,
            max_value=0,
        )


def test_clip_numeric_values_non_numeric_column():
    dataframe = pd.DataFrame({
        "price": ["10", "20"],
    })

    with pytest.raises(TypeError):
        clip_numeric_values(
            dataframe,
            column="price",
            min_value=0,
        )


# Data integrity

def test_operations_do_not_modify_original_dataframe():
    dataframe = pd.DataFrame({
        "name": [" Alex ", None, "John"],
        "age": [20, 30, 150],
    })

    original = dataframe.copy(deep=True)

    drop_missing_values(dataframe)
    fill_missing_values(
        dataframe,
        column="age",
        strategy="mean",
    )
    normalize_text(dataframe, column="name")
    clip_numeric_values(
        dataframe,
        column="age",
        min_value=0,
        max_value=100,
    )

    assert_frame_equal(dataframe, original)


def test_operations_preserve_dataframe_index():
    dataframe = pd.DataFrame(
        {"age": [20, 150, 30]},
        index=[10, 20, 30],
    )

    result, _ = clip_numeric_values(
        dataframe,
        column="age",
        max_value=100,
    )

    assert result.index.tolist() == [10, 20, 30]


def test_drop_missing_values_empty_dataframe():
    dataframe = pd.DataFrame({
        "age": pd.Series(dtype="float64"),
    })

    result, affected = drop_missing_values(dataframe)

    assert len(result) == 0
    assert affected == 0


def test_clip_numeric_values_ignores_missing_values():
    dataframe = pd.DataFrame({
        "price": [10.0, None, 150.0],
    })

    result, affected = clip_numeric_values(
        dataframe,
        column="price",
        max_value=100,
    )

    assert pd.isna(result["price"].iloc[1])
    assert affected == 1
