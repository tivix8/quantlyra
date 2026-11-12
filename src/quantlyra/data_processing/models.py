
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pandas as pd


@dataclass
class SourceFile:
    name: str
    format: str
    tables: dict[str, pd.DataFrame]
    file_id: str = field(default_factory=lambda: str(uuid4()))
    original_path: Path | None = None


@dataclass
class DatasetCollection:
    collection_id: str = field(default_factory=lambda: str(uuid4()))
    sources: dict[str, SourceFile] = field(default_factory=dict)

    def add_source(self, source: SourceFile) -> None:
        if source.file_id in self.sources:
            raise ValueError(f"Source {source.file_id} already exists")

        self.sources[source.file_id] = source

    def remove_source(self, file_id: str) -> None:
        if file_id not in self.sources:
            raise KeyError(f"Source {file_id} not found")

        del self.sources[file_id]

    def get_source(self, file_id: str) -> SourceFile:
        if file_id not in self.sources:
            raise KeyError(f"Source {file_id} not found")

        return self.sources[file_id]

    def get_table(
        self,
        file_id: str,
        table_name: str
    ) -> pd.DataFrame:
        source = self.get_source(file_id)

        if table_name not in source.tables:
            raise KeyError(f"Table {table_name} not found")

        return source.tables[table_name]
