
import pandas as pd
import pytest

from pathlib import Path

from quantlyra.data_processing import (
    DatasetCollection,
    SourceFile,
    load_file,
)


# --------------------------------------------------
# CSV LOADING
# --------------------------------------------------

@pytest.mark.parametrize(
    "separator",
    [",", ";", "\t", "|"]
)
def test_load_csv_with_different_separators(
    tmp_path,
    separator
):
    file_path = tmp_path / "products.csv"

    content = (
        f"name{separator}price\n"
        f"Apple{separator}100\n"
        f"Banana{separator}200\n"
    )

    file_path.write_text(
        content,
        encoding="utf-8"
    )

    source = load_file(file_path)

    assert source.format == "csv"
    assert source.name == "products.csv"
    assert "products" in source.tables

    df = source.tables["products"]

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["name", "price"]


def test_load_csv_cp1251_encoding(tmp_path):
    file_path = tmp_path / "products.csv"

    content = (
        "название;цена\n"
        "Яблоко;100\n"
        "Банан;200\n"
    )

    file_path.write_bytes(
        content.encode("cp1251")
    )

    source = load_file(file_path)

    df = source.tables["products"]

    assert df.iloc[0]["название"] == "Яблоко"
    assert df.iloc[1]["название"] == "Банан"


def test_load_empty_csv(tmp_path):
    file_path = tmp_path / "empty.csv"

    file_path.write_text(
        "name,price\n",
        encoding="utf-8"
    )

    source = load_file(file_path)

    df = source.tables["empty"]

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["name", "price"]


# --------------------------------------------------
# EXCEL LOADING
# --------------------------------------------------

def test_load_excel_multiple_sheets(tmp_path):
    file_path = tmp_path / "sales.xlsx"

    with pd.ExcelWriter(file_path) as writer:
        pd.DataFrame({
            "product": ["Apple", "Banana"],
            "price": [100, 200]
        }).to_excel(
            writer,
            sheet_name="Продажи",
            index=False
        )

        pd.DataFrame({
            "product": ["Orange"],
            "quantity": [50]
        }).to_excel(
            writer,
            sheet_name="Остатки",
            index=False
        )

    source = load_file(file_path)

    assert source.format == "xlsx"
    assert source.name == "sales.xlsx"

    assert set(source.tables.keys()) == {
        "Продажи",
        "Остатки"
    }

    assert source.tables["Продажи"].shape == (2, 2)
    assert source.tables["Остатки"].shape == (1, 2)


def test_load_excel_preserves_sheet_names(tmp_path):
    file_path = tmp_path / "test.xlsx"

    with pd.ExcelWriter(file_path) as writer:
        pd.DataFrame({"value": [1]}).to_excel(
            writer,
            sheet_name="My Sheet",
            index=False
        )

    source = load_file(file_path)

    assert "My Sheet" in source.tables


# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_file("nonexistent.csv")


def test_unsupported_file_format(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError):
        load_file(file_path)


def test_path_is_directory(tmp_path):
    with pytest.raises(ValueError):
        load_file(tmp_path)


# --------------------------------------------------
# DATASET COLLECTION
# --------------------------------------------------

def test_create_dataset_collection():
    collection = DatasetCollection()

    assert isinstance(collection.collection_id, str)
    assert collection.sources == {}


def test_add_source_to_collection():
    collection = DatasetCollection()

    source = SourceFile(
        name="products.csv",
        format="csv",
        tables={
            "products": pd.DataFrame({
                "name": ["Apple"],
                "price": [100]
            })
        }
    )

    collection.add_source(source)

    assert source.file_id in collection.sources
    assert collection.sources[source.file_id] is source


def test_add_duplicate_source_id():
    collection = DatasetCollection()

    source = SourceFile(
        name="products.csv",
        format="csv",
        tables={}
    )

    collection.add_source(source)

    with pytest.raises(ValueError):
        collection.add_source(source)


def test_remove_source():
    collection = DatasetCollection()

    source = SourceFile(
        name="products.csv",
        format="csv",
        tables={}
    )

    collection.add_source(source)
    collection.remove_source(source.file_id)

    assert source.file_id not in collection.sources


def test_remove_nonexistent_source():
    collection = DatasetCollection()

    with pytest.raises(KeyError):
        collection.remove_source("unknown-id")


def test_get_source():
    collection = DatasetCollection()

    source = SourceFile(
        name="products.csv",
        format="csv",
        tables={}
    )

    collection.add_source(source)

    result = collection.get_source(source.file_id)

    assert result is source


def test_get_nonexistent_source():
    collection = DatasetCollection()

    with pytest.raises(KeyError):
        collection.get_source("unknown-id")


def test_get_table():
    collection = DatasetCollection()

    dataframe = pd.DataFrame({
        "name": ["Apple"],
        "price": [100]
    })

    source = SourceFile(
        name="products.xlsx",
        format="xlsx",
        tables={
            "Products": dataframe
        }
    )

    collection.add_source(source)

    result = collection.get_table(
        source.file_id,
        "Products"
    )

    pd.testing.assert_frame_equal(result, dataframe)


def test_get_nonexistent_table():
    collection = DatasetCollection()

    source = SourceFile(
        name="products.xlsx",
        format="xlsx",
        tables={}
    )

    collection.add_source(source)

    with pytest.raises(KeyError):
        collection.get_table(
            source.file_id,
            "Unknown"
        )


# --------------------------------------------------
# MULTIPLE FILES
# --------------------------------------------------

def test_load_multiple_files_with_same_name(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_dir.mkdir()
    second_dir.mkdir()

    first_file = first_dir / "products.csv"
    second_file = second_dir / "products.csv"

    first_file.write_text(
        "name,price\nApple,100\n",
        encoding="utf-8"
    )

    second_file.write_text(
        "name,price\nBanana,200\n",
        encoding="utf-8"
    )

    collection = DatasetCollection()

    first_source = load_file(first_file)
    second_source = load_file(second_file)

    collection.add_source(first_source)
    collection.add_source(second_source)

    assert len(collection.sources) == 2

    assert first_source.file_id != second_source.file_id

    assert (
        first_source.tables["products"].iloc[0]["name"]
        == "Apple"
    )

    assert (
        second_source.tables["products"].iloc[0]["name"]
        == "Banana"
    )
