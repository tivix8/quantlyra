
import csv
from pathlib import Path

import pandas as pd

from .models import SourceFile


def detect_csv_separator(file_path: Path) -> str:
    with file_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as file:
        sample = file.read(8192)

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|"
        )
        return dialect.delimiter

    except csv.Error:
        return ","


def read_csv_file(file_path: Path) -> pd.DataFrame:
    separator = detect_csv_separator(file_path)

    encodings = ["utf-8-sig", "utf-8", "cp1251"]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                sep=separator,
                encoding=encoding
            )

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        f"Could not determine CSV encoding: {last_error}"
    )


def load_csv(file_path: Path) -> dict[str, pd.DataFrame]:
    dataframe = read_csv_file(file_path)

    table_name = file_path.stem

    return {
        table_name: dataframe
    }


def load_excel(file_path: Path) -> dict[str, pd.DataFrame]:
    try:
        tables = pd.read_excel(
            file_path,
            sheet_name=None,
            engine="openpyxl"
        )

        return tables

    except Exception as error:
        raise ValueError(
            f"Failed to read Excel file {file_path.name}: {error}"
        ) from error


def load_file(file_path: str | Path) -> SourceFile:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".csv":
        tables = load_csv(file_path)
        file_format = "csv"

    elif extension in ['.xlsx', '.xls', '.xlsm']:
        tables = load_excel(file_path)
        file_format = "xlsx"

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    return SourceFile(
        name=file_path.name,
        format=file_format,
        tables=tables,
        original_path=file_path.resolve()
    )
