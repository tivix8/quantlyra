
import pandas as pd
import pytest

from quantlyra.data_cleaning.cleaner import clean_dataframe
from quantlyra.data_cleaning.history import (
    replay_cleaning_history,
    undo_last_operation,
)
from quantlyra.data_cleaning.models import CleaningOperation


def create_operations():
    return [
        CleaningOperation(
            operation_type="drop_duplicates",
        ),
        CleaningOperation(
            operation_type="normalize_text",
            column="name",
            parameters={"lowercase": True, "strip": True},
        ),
    ]


def test_replay_cleaning_history():
    dataframe = pd.DataFrame({
        "name": [" Alex ", "John", "John"],
    })

    initial_result = clean_dataframe(
        dataframe,
        create_operations(),
    )

    replayed_result = replay_cleaning_history(
        dataframe,
        initial_result.history,
    )

    pd.testing.assert_frame_equal(
        replayed_result.cleaned_dataframe,
        initial_result.cleaned_dataframe,
    )

    assert len(replayed_result.history) == 2


def test_undo_last_operation():
    dataframe = pd.DataFrame({
        "name": [" Alex ", "John", "John"],
    })

    result = clean_dataframe(
        dataframe,
        create_operations(),
    )

    undone_result = undo_last_operation(result)

    assert len(undone_result.history) == 1
    assert undone_result.cleaned_dataframe["name"].tolist() == [
        " Alex ",
        "John",
    ]


def test_undo_with_empty_history():
    dataframe = pd.DataFrame({
        "name": ["Alex", "John"],
    })

    result = clean_dataframe(dataframe, [])

    undone_result = undo_last_operation(result)

    pd.testing.assert_frame_equal(
        undone_result.cleaned_dataframe,
        dataframe,
    )

    assert undone_result.history == []


def test_replay_with_empty_history():
    dataframe = pd.DataFrame({
        "age": [20, 30],
    })

    result = replay_cleaning_history(dataframe, [])

    pd.testing.assert_frame_equal(
        result.cleaned_dataframe,
        dataframe,
    )

    assert result.history == []


def test_replay_does_not_modify_original_dataframe():
    dataframe = pd.DataFrame({
        "name": [" Alex ", "John"],
    })

    original = dataframe.copy(deep=True)

    result = clean_dataframe(
        dataframe,
        create_operations()[1:],
    )

    replay_cleaning_history(
        dataframe,
        result.history,
    )

    pd.testing.assert_frame_equal(dataframe, original)


def test_replay_rejects_invalid_dataframe():
    with pytest.raises(
        TypeError,
        match="original_dataframe must be a pandas DataFrame",
    ):
        replay_cleaning_history([1, 2, 3], [])


def test_replay_rejects_invalid_history():
    dataframe = pd.DataFrame({"age": [20, 30]})

    with pytest.raises(TypeError, match="history must be a list"):
        replay_cleaning_history(dataframe, None)


def test_undo_rejects_invalid_result():
    with pytest.raises(
        TypeError,
        match="cleaning_result must be a CleaningResult instance",
    ):
        undo_last_operation(None)


def test_history_preserves_operation_parameters():
    dataframe = pd.DataFrame({
        "name": [" Alex "],
    })

    operation = CleaningOperation(
        operation_type="normalize_text",
        column="name",
        parameters={"strip": True},
    )

    result = clean_dataframe(dataframe, [operation])

    operation.parameters["strip"] = False

    assert result.history[0].operation.parameters["strip"] is True
