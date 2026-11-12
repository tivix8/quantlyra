
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from quantlyra.data_processing.profiler import (
    profile_dataframe,
    profile_column,
)


import pandas as pd
import pytest

from quantlyra.data_processing.models import (
    DatasetCollection,
    SourceFile,
)

from quantlyra.data_processing.profiler import profile_collection

from quantlyra.data_processing.profile_models import (
    CollectionProfile,
    SourceProfile,
    TableProfile,
)

# --------------------------------------------------
# Numeric columns
# --------------------------------------------------

def test_profile_numeric_column():
    series = pd.Series([10, 20, 30, None])

    profile = profile_column("price", series)

    assert profile.name == "price"
    assert profile.dtype == "float64"

    assert profile.non_null_count == 3
    assert profile.missing_count == 1
    assert profile.missing_percentage == 25.0

    assert profile.unique_count == 3
    assert profile.unique_percentage == 100.0

    assert profile.numeric_stats is not None
    assert profile.categorical_stats is None
    assert profile.datetime_stats is None

    assert profile.numeric_stats.min_value == 10.0
    assert profile.numeric_stats.max_value == 30.0
    assert profile.numeric_stats.mean == 20.0
    assert profile.numeric_stats.median == 20.0
    assert profile.numeric_stats.std == 10.0

    assert profile.numeric_stats.quantiles == {
        "25%": 15.0,
        "50%": 20.0,
        "75%": 25.0,
    }


def test_profile_empty_numeric_column():
    series = pd.Series([None, None], dtype="float64")

    profile = profile_column("price", series)

    assert profile.non_null_count == 0
    assert profile.missing_count == 2
    assert profile.missing_percentage == 100.0

    assert profile.unique_count == 0
    assert profile.unique_percentage == 0.0

    assert profile.numeric_stats is not None
    assert profile.numeric_stats.min_value is None
    assert profile.numeric_stats.max_value is None
    assert profile.numeric_stats.mean is None
    assert profile.numeric_stats.median is None
    assert profile.numeric_stats.std is None

    assert profile.numeric_stats.quantiles == {
        "25%": None,
        "50%": None,
        "75%": None,
    }


# --------------------------------------------------
# Categorical columns
# --------------------------------------------------

def test_profile_categorical_column():
    series = pd.Series([
        "Laptop",
        "Phone",
        "Laptop",
        "Tablet",
        "Laptop",
        "Phone",
    ])

    profile = profile_column("product", series)

    assert profile.name == "product"
    assert profile.dtype in ("object", "str")

    assert profile.non_null_count == 6
    assert profile.missing_count == 0

    assert profile.unique_count == 3
    assert profile.unique_percentage == 50.0

    assert profile.numeric_stats is None
    assert profile.datetime_stats is None
    assert profile.categorical_stats is not None

    assert profile.categorical_stats.top_values == {
        "Laptop": 3,
        "Phone": 2,
        "Tablet": 1,
    }


def test_categorical_column_with_missing_values():
    series = pd.Series([
        "Laptop",
        None,
        "Phone",
        "Laptop",
    ])

    profile = profile_column("product", series)

    assert profile.non_null_count == 3
    assert profile.missing_count == 1
    assert profile.missing_percentage == 25.0

    assert profile.unique_count == 2
    assert profile.unique_percentage == pytest.approx(66.6667, rel=1e-4)


def test_categorical_column_returns_only_five_top_values():
    series = pd.Series([
        "A", "A",
        "B", "B",
        "C", "C",
        "D", "D",
        "E", "E",
        "F", "F",
    ])

    profile = profile_column("category", series)

    assert profile.categorical_stats is not None
    assert len(profile.categorical_stats.top_values) == 5


# --------------------------------------------------
# Datetime columns
# --------------------------------------------------

def test_profile_datetime_column():
    series = pd.Series(pd.to_datetime([
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    ]))

    profile = profile_column("created_at", series)

    assert profile.name == "created_at"
    assert profile.datetime_stats is not None

    assert profile.numeric_stats is None
    assert profile.categorical_stats is None

    assert profile.datetime_stats.min_date == pd.Timestamp("2026-01-01")
    assert profile.datetime_stats.max_date == pd.Timestamp("2026-01-03")
    assert profile.datetime_stats.date_range == pd.Timedelta(days=2)


def test_datetime_column_with_missing_values():
    series = pd.Series(pd.to_datetime([
        "2026-01-01",
        None,
        "2026-01-03",
    ]))

    profile = profile_column("created_at", series)

    assert profile.missing_count == 1
    assert profile.non_null_count == 2

    assert profile.datetime_stats is not None
    assert profile.datetime_stats.min_date == pd.Timestamp("2026-01-01")
    assert profile.datetime_stats.max_date == pd.Timestamp("2026-01-03")


def test_empty_datetime_column():
    series = pd.Series([], dtype="datetime64[ns]")

    profile = profile_column("created_at", series)

    assert profile.datetime_stats is not None
    assert profile.datetime_stats.min_date is None
    assert profile.datetime_stats.max_date is None
    assert profile.datetime_stats.date_range is None


# --------------------------------------------------
# Boolean columns
# --------------------------------------------------

def test_boolean_column_is_categorical():
    series = pd.Series([True, False, True, True])

    profile = profile_column("is_active", series)

    assert profile.numeric_stats is None
    assert profile.categorical_stats is not None

    assert profile.categorical_stats.top_values == {
        True: 3,
        False: 1,
    }


# --------------------------------------------------
# DataFrame profiling
# --------------------------------------------------

def test_profile_dataframe():
    dataframe = pd.DataFrame({
        "product": ["Laptop", "Phone", "Tablet"],
        "price": [1000, 500, 700],
        "quantity": [2, 5, 3],
    })

    profile = profile_dataframe(dataframe, "sales")

    assert profile.table_name == "sales"
    assert profile.row_count == 3
    assert profile.column_count == 3

    assert profile.duplicate_rows == 0
    assert profile.memory_usage > 0

    assert len(profile.columns) == 3

    assert profile.columns[0].name == "product"
    assert profile.columns[1].name == "price"
    assert profile.columns[2].name == "quantity"


def test_profile_dataframe_with_duplicate_rows():
    dataframe = pd.DataFrame({
        "product": ["Laptop", "Phone", "Laptop"],
        "price": [1000, 500, 1000],
    })

    profile = profile_dataframe(dataframe)

    assert profile.duplicate_rows == 1


def test_profile_empty_dataframe():
    dataframe = pd.DataFrame()

    profile = profile_dataframe(dataframe, "empty")

    assert profile.table_name == "empty"
    assert profile.row_count == 0
    assert profile.column_count == 0
    assert profile.duplicate_rows == 0
    assert profile.columns == []


def test_profile_dataframe_with_zero_rows():
    dataframe = pd.DataFrame({
        "product": pd.Series(dtype="object"),
        "price": pd.Series(dtype="float64"),
    })

    profile = profile_dataframe(dataframe)

    assert profile.row_count == 0
    assert profile.column_count == 2

    assert len(profile.columns) == 2


def test_profile_dataframe_with_duplicate_column_names():
    dataframe = pd.DataFrame(
        [[10, 30], [20, 40]],
        columns=["value", "value"],
    )

    profile = profile_dataframe(dataframe)

    assert len(profile.columns) == 2

    assert profile.columns[0].name == "value"
    assert profile.columns[1].name == "value"

    assert profile.columns[0].numeric_stats.mean == 15.0
    assert profile.columns[1].numeric_stats.mean == 35.0


def test_profile_does_not_modify_original_dataframe():
    dataframe = pd.DataFrame({
        "product": ["Laptop", "Phone", None],
        "price": [1000, 500, None],
    })

    original = dataframe.copy(deep=True)

    profile_dataframe(dataframe)

    assert_frame_equal(dataframe, original)





@pytest.fixture
def sample_collection():
    """Create a collection with multiple files and tables."""

    collection = DatasetCollection()

    # First file: Excel with two sheets
    excel_source = SourceFile(
        name="sales.xlsx",
        format="xlsx",
        tables={
            "Sales": pd.DataFrame({
                "product": ["Laptop", "Phone", "Laptop"],
                "price": [1000, 500, 1000],
            }),
            "Returns": pd.DataFrame({
                "product": ["Phone", "Tablet"],
                "quantity": [1, 2],
            }),
        },
    )

    # Second file: CSV with one table
    csv_source = SourceFile(
        name="products.csv",
        format="csv",
        tables={
            "products": pd.DataFrame({
                "name": ["Laptop", "Phone", "Tablet"],
                "stock": [10, 20, 15],
            }),
        },
    )

    collection.add_source(excel_source)
    collection.add_source(csv_source)

    return collection


def test_profile_empty_collection():
    collection = DatasetCollection()

    profile = profile_collection(collection)

    assert isinstance(profile, CollectionProfile)
    assert profile.collection_id == collection.collection_id
    assert profile.sources == {}


def test_profile_collection_with_multiple_files(sample_collection):
    profile = profile_collection(sample_collection)

    assert isinstance(profile, CollectionProfile)
    assert profile.collection_id == sample_collection.collection_id
    assert len(profile.sources) == 2


def test_profile_collection_preserves_file_ids(sample_collection):
    profile = profile_collection(sample_collection)

    for file_id in sample_collection.sources:
        assert file_id in profile.sources
        assert profile.sources[file_id].file_id == file_id


def test_profile_collection_preserves_file_metadata(sample_collection):
    profile = profile_collection(sample_collection)

    excel_source = next(
        source
        for source in profile.sources.values()
        if source.name == "sales.xlsx"
    )

    assert excel_source.name == "sales.xlsx"
    assert excel_source.format == "xlsx"

    csv_source = next(
        source
        for source in profile.sources.values()
        if source.name == "products.csv"
    )

    assert csv_source.name == "products.csv"
    assert csv_source.format == "csv"


def test_profile_collection_profiles_all_tables(sample_collection):
    profile = profile_collection(sample_collection)

    excel_source = next(
        source
        for source in profile.sources.values()
        if source.name == "sales.xlsx"
    )

    csv_source = next(
        source
        for source in profile.sources.values()
        if source.name == "products.csv"
    )

    assert set(excel_source.tables.keys()) == {"Sales", "Returns"}
    assert set(csv_source.tables.keys()) == {"products"}


def test_profile_collection_returns_table_profiles(sample_collection):
    profile = profile_collection(sample_collection)

    for source in profile.sources.values():
        for table_profile in source.tables.values():
            assert isinstance(table_profile, TableProfile)


def test_profile_collection_calculates_table_statistics(sample_collection):
    profile = profile_collection(sample_collection)

    excel_source = next(
        source
        for source in profile.sources.values()
        if source.name == "sales.xlsx"
    )

    sales_profile = excel_source.tables["Sales"]

    assert sales_profile.row_count == 3
    assert sales_profile.column_count == 2
    assert sales_profile.duplicate_rows == 1

    price_profile = next(
        column
        for column in sales_profile.columns
        if column.name == "price"
    )

    assert price_profile.numeric_stats is not None
    assert price_profile.numeric_stats.mean == pytest.approx(833.333333)


def test_same_table_names_in_different_files(sample_collection):
    first_source = next(
        iter(sample_collection.sources.values())
    )

    second_source = SourceFile(
        name="another.xlsx",
        format="xlsx",
        tables={
            "Sales": pd.DataFrame({
                "amount": [100, 200],
            }),
        },
    )

    sample_collection.add_source(second_source)

    profile = profile_collection(sample_collection)

    first_profile = profile.sources[first_source.file_id]
    second_profile = profile.sources[second_source.file_id]

    assert "Sales" in first_profile.tables
    assert "Sales" in second_profile.tables

    assert (
        first_profile.tables["Sales"].row_count
        == 3
    )

    assert (
        second_profile.tables["Sales"].row_count
        == 2
    )


def test_profile_collection_with_empty_file():
    collection = DatasetCollection()

    source = SourceFile(
        name="empty.xlsx",
        format="xlsx",
        tables={},
    )

    collection.add_source(source)

    profile = profile_collection(collection)

    source_profile = profile.sources[source.file_id]

    assert isinstance(source_profile, SourceProfile)
    assert source_profile.tables == {}


def test_profile_collection_does_not_modify_dataframes(sample_collection):
    originals = {
        file_id: {
            table_name: dataframe.copy(deep=True)
            for table_name, dataframe in source.tables.items()
        }
        for file_id, source in sample_collection.sources.items()
    }

    profile_collection(sample_collection)

    for file_id, source in sample_collection.sources.items():
        for table_name, dataframe in source.tables.items():
            pd.testing.assert_frame_equal(
                dataframe,
                originals[file_id][table_name],
            )
