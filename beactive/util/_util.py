from itertools import combinations
from time import ctime
from typing import Union, Type, Optional, Callable, Iterable, Dict, Any
from warnings import filterwarnings

import numpy as np
import pandas as pd
import ray
from scipy import stats

filterwarnings('ignore')

R = 63710088


def safe_call_for_mixed_data(
        func: Callable,
        X: np.ndarray,
        categorical_indices: np.ndarray = None,
        **kwargs):

    if categorical_indices is None or categorical_indices.shape[0] == 0:
        return func(X)

    if categorical_indices.dtype == bool:
        categorical_indices = np.flatnonzero(categorical_indices)

    indices_numeric = np.setdiff1d(np.arange(X.shape[1]), categorical_indices)

    X_numeric = X[:, indices_numeric]
    X_category = X[:, categorical_indices]

    X_transform = func(X_numeric, **kwargs)

    indices_all = np.argsort(
        np.hstack([indices_numeric, categorical_indices])
    )

    return np.hstack([X_transform, X_category])[:, indices_all]


def correct_label(y: np.ndarray) -> np.ndarray:
    _y = np.copy(y).astype(int)
    for i in range(1, _y.shape[1]):
        _y[:, i] = np.where(_y[:, i - 1] == 0, 0, _y[:, i])
    return _y.astype(int)


def pack_label(y: Union[np.ndarray, Type[np.ndarray]]) -> np.ndarray:
    _y = correct_label(y)
    return np.packbits(_y, axis=1, bitorder='little').ravel()


def unpack_label(y: np.ndarray, n_dim: int) -> np.ndarray:
    _y = np.copy(y).astype('uint8')
    _y = np.unpackbits(_y.reshape(y.shape[0], 1), axis=1, bitorder='little')[:, :n_dim]
    return correct_label(_y)


def check_ray():
    if not ray.is_initialized():
        raise Exception('Ray should be initialized')


def log(msg: str, elapsed: float = None):
    if elapsed is not None:
        print('[{}] [Elapsed: {:3f} s] {}'.format(ctime(), elapsed, msg))
    else:
        print('[{}] {}'.format(ctime(), msg))


def safe_mean(d: Iterable[Union[int, float]], na: Union[int, float] = 0) -> Union[int, float]:
    s = np.mean(list(d))
    return na if np.isnan(s) else s

def safe_std(d: Iterable[Union[int, float]], na: Union[int, float] = 0.0) -> Union[int, float]:
    s = np.std(list(d))
    return na if np.isnan(s) else s


def safe_max(d: Iterable[Union[int, float]], na: Union[int, float] = np.inf) -> Union[int, float]:
    try:
        return np.max(list(d))
    except ValueError:
        return na


def safe_min(d: Iterable[Union[int, float]], na: Union[int, float] = -np.inf) -> Union[int, float]:
    try:
        return np.min(list(d))
    except ValueError:
        return na


def safe_entropy(d: Iterable[Union[int, float]], na: Union[int, float] = 0.0) -> Union[int, float]:
    s = stats.entropy(list(d))
    return na if np.isnan(s) else s


def combination_sum(kvs: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
    keys = np.array(list(kvs.keys()))
    values = np.array(list(kvs.values()))
    size = keys.shape[0]
    combs = [np.array(indices) for i in range(1, size + 1) for indices in combinations(range(0, size), i)]

    return {
        '+'.join(keys[i]): np.sum(values[i])
        for i in combs
    }

def safe_subset(data: pd.DataFrame,
                from_boundary: int,
                to_boundary: int,
                from_col: str,
                to_col: Optional[str] = None,
                duration_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    # For begin - end column, from_time < end_col & to_time > begin_col
    # For single point, from_time <= begin_col < to_time

    from_times = data.loc[:, from_col].values
    from_times = np.where(np.isnan(from_times), -np.inf, from_times)

    if to_col is None:
        cond = (from_boundary <= from_times) & (from_times < to_boundary)
        return data.loc[cond, :]

    to_times = data.loc[:, to_col].values
    to_times = np.where(np.isnan(to_times), np.inf, to_times)

    inner = (from_boundary <= from_times) & (to_boundary >= to_times)
    outer = (from_boundary > from_times) & (to_boundary < to_times)
    left_overlap = (from_boundary < to_times) & (to_boundary > to_times) & (from_boundary > from_times)
    right_overlap = (to_boundary > from_times) & (from_boundary < from_times) & (to_boundary < to_times)

    cond = inner | outer | left_overlap | right_overlap

    subset = data.loc[cond, :]
    subset_bound = subset.assign(**{
        from_col: np.clip(subset.loc[:, from_col].values, a_min=from_boundary, a_max=None),
        to_col: np.clip(subset.loc[:, to_col].values, a_min=None, a_max=to_boundary)
    })
    subset_non_zero_duration = subset_bound.loc[lambda x: x[to_col] - x[from_col] > 0, :]

    if len(subset_non_zero_duration.index) == 0:
        return None

    if duration_col is None:
        return subset_non_zero_duration
    else:
        return subset_non_zero_duration.assign(**{
            duration_col: subset_non_zero_duration.loc[:, to_col] - subset_non_zero_duration.loc[:, from_col]
        })


def safe_immediate_previous(data: pd.DataFrame, data_point: int, col: str) -> Optional[pd.Series]:
    try:
        return data.loc[lambda x: x[col] < data_point, :].iloc[-1]
    except IndexError:
        return None


def safe_item(d: Dict[str, Any], key: str, default: Any) -> Any:
    if key in d:
        return d[key]
    else:
        return default