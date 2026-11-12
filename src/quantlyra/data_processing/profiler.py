import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_complex_dtype,
)

from .profile_models import (
    ColumnProfile,
    TableProfile,
    NumericStats,
    CategoricalStats,
    DatetimeStats,
)

from .models import DatasetCollection

from .profile_models import (
    CollectionProfile,
    SourceProfile,
)

def _to_float(value) -> float | None:
    """Convert a pandas/numpy numeric value to float."""
    if pd.isna(value):
        return None

    return float(value)


def _profile_numeric(series: pd.Series) -> NumericStats:
    """Calculate statistics for a numeric column."""
    values = series.dropna()

    if values.empty:
        return NumericStats(
            quantiles={
                "25%": None,
                "50%": None,
                "75%": None,
            }
        )

    quantiles = values.quantile([0.25, 0.50, 0.75])

    return NumericStats(
        min_value=_to_float(values.min()),
        max_value=_to_float(values.max()),
        mean=_to_float(values.mean()),
        median=_to_float(values.median()),
        std=_to_float(values.std()),
        quantiles={
            "25%": _to_float(quantiles.loc[0.25]),
            "50%": _to_float(quantiles.loc[0.50]),
            "75%": _to_float(quantiles.loc[0.75]),
        },
    )


def _profile_categorical(series: pd.Series) -> CategoricalStats:
    """Calculate statistics for a categorical column."""
    top_values = series.value_counts(dropna=True).head(5)

    return CategoricalStats(
        top_values={
            value: int(count)
            for value, count in top_values.items()
        }
    )


def _profile_datetime(series: pd.Series) -> DatetimeStats:
    """Calculate statistics for a datetime column."""
    values = series.dropna()

    if values.empty:
        return DatetimeStats()

    min_date = values.min()
    max_date = values.max()

    return DatetimeStats(
        min_date=min_date,
        max_date=max_date,
        date_range=max_date - min_date,
    )


def profile_column(name: str, series: pd.Series) -> ColumnProfile:
    """Create a profile for a single DataFrame column."""
    total_count = len(series)

    non_null_count = int(series.count())
    missing_count = int(series.isna().sum())

    missing_percentage = (
        missing_count / total_count * 100
        if total_count > 0
        else 0.0
    )

    unique_count = int(series.nunique(dropna=True))

    unique_percentage = (
        unique_count / non_null_count * 100
        if non_null_count > 0
        else 0.0
    )

    numeric_stats = None
    categorical_stats = None
    datetime_stats = None

    if is_datetime64_any_dtype(series.dtype):
        datetime_stats = _profile_datetime(series)

    elif (
        is_numeric_dtype(series.dtype)
        and not is_bool_dtype(series.dtype)
        and not is_complex_dtype(series.dtype)
    ):
        numeric_stats = _profile_numeric(series)

    else:
        categorical_stats = _profile_categorical(series)

    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        datetime_stats=datetime_stats,
    )


def profile_dataframe(
    dataframe: pd.DataFrame,
    table_name: str = "unnamed",
) -> TableProfile:
    """Create a complete profile for a DataFrame."""

    columns = [
        profile_column(str(column_name), dataframe.iloc[:, index])
        for index, column_name in enumerate(dataframe.columns)
    ]

    return TableProfile(
        table_name=table_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        duplicate_rows=int(dataframe.duplicated().sum()),
        memory_usage=int(
            dataframe.memory_usage(index=True, deep=True).sum()
        ),
        columns=columns,
    )


def profile_collection(
    collection: DatasetCollection,
) -> CollectionProfile:
    """Create profiles for all files and tables in a collection."""

    source_profiles = {}

    for file_id, source in collection.sources.items():

        table_profiles = {}

        for table_name, dataframe in source.tables.items():

            table_profile = profile_dataframe(
                dataframe=dataframe,
                table_name=table_name,
            )

            table_profiles[table_name] = table_profile

        source_profile = SourceProfile(
            file_id=file_id,
            name=source.name,
            format=source.format,
            tables=table_profiles,
        )

        source_profiles[file_id] = source_profile

    return CollectionProfile(
        collection_id=collection.collection_id,
        sources=source_profiles,
    )
