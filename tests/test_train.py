from model.train import get_csvs_df
import os
import pytest
from src.model.train import train_model
import numpy as np


def test_csvs_no_files():
    with pytest.raises(RuntimeError) as error:
        get_csvs_df("./")
    assert error.match("No CSV files found in provided data")


def test_csvs_no_files_invalid_path():
    with pytest.raises(RuntimeError) as error:
        get_csvs_df("/invalid/path/does/not/exist/")
    assert error.match("Cannot use non-existent path provided")


def test_csvs_creates_dataframe():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    datasets_directory = os.path.join(current_directory, 'datasets')
    result = get_csvs_df(datasets_directory)
    assert len(result) == 20


def test_train_model():
    X_train = np.array([1,2,3,4,5,6]).reshape(-1,1)
    y_train = np.array(10,9,8,7,6,5)
    data = {"train":{"X":X_train,"y":y_train}}

    reg_model = train_model(data,{"alpha":1.2})
    preds = reg_model.predict([[1],[2]])
    np.testing.assert_almost_equal(preds,[9.93939393,9.0303030303])
