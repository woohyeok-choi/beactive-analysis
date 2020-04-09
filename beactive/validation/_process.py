from ..preprocess import ProcessedData
from typing import Iterable, Union, Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

class Validator(ABC):
    def __init__(self,
                 processed_data: ProcessedData,
                 apply_filtering: bool,
                 subjects: Iterable[str] = None,
                 include_features: Iterable[str] = None,
                 exclude_features: Iterable[str] = None,
                 seed: int = None):
        features = processed_data.feature_
        labels = processed_data.label_
        group = processed_data.group_
        timestamps = processed_data.point_
        category_names = processed_data.category_cols

        if include_features is not None:
            include_mask = features.columns.isin(include_features) if include_features is not None else np.ones(
                features.shape[1], dtype=bool)
            features = features.loc[:, include_mask]
        elif exclude_features is not None:
            exclude_mask = features.columns.isin(exclude_features) if exclude_features is not None else np.zeros(
                features.shape[1], dtype=bool)
            features = features.loc[:, ~exclude_mask]

        self._X = features.fillna(0.0).to_numpy()
        self._Y = labels
        self._group = group
        self._timestamps = timestamps
        self._category_names = category_names


def _subset_features(features: pd.DataFrame,
                     include_features: Iterable[str] = None,
                     exclude_features: Iterable[str] = None) -> pd.DataFrame:
    mask = None
    columns = features.columns

    if include_features is not None:
        mask = columns.isin(include_features)
    elif exclude_features is not None:
        mask = ~columns.isin(exclude_features)

    return features.loc[:, mask] if mask is not None else features


def _subset_rows(features: pd.DataFrame,
                 labels: pd.DataFrame,
                 apply_filtering: bool = False) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    ret = dict()
    for i in range(labels.shape[1]):
        if apply_filtering:
            mask = labels.iloc[:, i - 1].to_numpy() == 1 if i > 0 else np.ones(labels.shape[0], dtype=bool)
        else:
            mask = labels.iloc[:, i]
        ret[str(labels.columns[i])] = (features.loc[mask, :], labels.loc[mask, :])

    return ret


def _encode_categorical_values(features: pd.DataFrame,
                               category_names: Iterable[str],
                               is_ohe: bool = False) -> Tuple[pd.DataFrame, Iterable[str]]:
    encoder = OneHotEncoder(dtype='uint8', sparse=False) if is_ohe else OrdinalEncoder(dtype='uint8')
    mask_category = features.columns.isin(category_names)
    



def preprocess(processed_data: ProcessedData,
               include_features: Iterable[str] = None,
               exclude_features: Iterable[str] = None):
    features = processed_data.feature_
    labels = processed_data.label_
    GROUP = processed_data.group_
    CATEGORY_NAMES = processed_data.category_cols

    # Select subset of features
    if include_features is not None:
        include_mask = features.columns.isin(include_features) if include_features is not None else np.ones(
            features.shape[1], dtype=bool)
        features = features.loc[:, include_mask]
    elif exclude_features is not None:
        exclude_mask = features.columns.isin(exclude_features) if exclude_features is not None else np.zeros(
            features.shape[1], dtype=bool)
        features = features.loc[:, ~exclude_mask]

    COLUMNS = features.columns
    X = features.fillna(0.0).to_numpy()
    Y = labels.to_numpy()
    GROUP = GROUP.to_numpy()

    return X, Y, GROUP, COLUMNS, COLUMNS.isin(CATEGORY_NAMES)

def _ind_k_fold_split(X: np.ndarray,
                      y: np.ndarray,
                      group: np.ndarray,
                      seed: int):
    pass

def _int_time_split(X: np.ndarray,
                    y: np.ndarray,
                    )

def split(X: np/,
          y: Union[pd.Series, np.ndarray],
          seed: int = None):
    # Ind. K-fold
    # Ind. Time
    # LOO
    pass

def resample(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], seed: int):
    pass

def scale(X: Union[pd.DataFrame, np.ndarray], indices_numeric: np.ndarray):
    pass

def encode(X: Union[pd.DataFrame, np.ndarray], indices_category: np.ndarray):
    pass