from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat

from ..util import safe_call_for_mixed_data


def _check_transformer(transformer: Any) -> bool:
    return hasattr(transformer, 'fit_transform') and hasattr(transformer, 'transform')


def _random_curve(x: np.ndarray, sigma: float, knot: int):
    l, d = x.shape
    xx = np.ones((d, 1)) * np.arange(0, l, (l - 1) / (knot + 1))
    xx = xx.transpose()
    yy = np.random.normal(1.0, sigma, (knot + 2, d))

    r = np.arange(l)

    cs = [CubicSpline(xx[:, i], yy[:, i])(r) for i in range(d)]
    return np.vstack(cs).transpose()


class SafeTransformer:
    def __init__(self,
                 transformer: Any,
                 apply_in_numeric: bool = True,
                 categorical_indices: Optional[np.ndarray] = None):

        if not _check_transformer(transformer):
            raise Exception('Transformer have \'fit\' and \'transform\' attribute.')

        self._is_fitted = False
        self._apply_in_numeric = apply_in_numeric
        self._categorical_indices = categorical_indices
        self._transformer = transformer

    @property
    def transformer_(self):
        return self._transformer

    def transform(self, X: Union[pd.DataFrame, np.ndarray], refit: bool = False, **kwargs) -> \
            Union[pd.DataFrame, np.ndarray]:
        if not self._is_fitted or refit:
            func = self._transformer.fit_transform
            self._is_fitted = True
        else:
            func = self._transformer.transform

        if self._categorical_indices is None:
            categorical_indices = np.array([])
        elif self._categorical_indices.dtype == bool:
            categorical_indices = np.flatnonzero(self._categorical_indices)
        else:
            categorical_indices = self._categorical_indices

        numeric_indices = np.setdiff1d(np.arange(X.shape[1]), categorical_indices)
        columns = None

        if isinstance(X, pd.DataFrame):
            columns = X.columns
            if self._apply_in_numeric:
                dtypes = X.dtypes.iloc[categorical_indices].to_dict()
            else:
                dtypes = X.dtypes.iloc[numeric_indices].to_dict()
            X_numeric = X.iloc[:, numeric_indices]
            X_category = X.iloc[:, categorical_indices]
        else:
            dtypes = X.dtype
            X_numeric = X[:, numeric_indices]
            X_category = X[:, categorical_indices]

        if self._apply_in_numeric:
            X_numeric = func(X_numeric, **kwargs)
        else:
            X_category = func(X_category, **kwargs)

        all_indices = np.argsort(np.hstack([numeric_indices, categorical_indices]))
        ret = np.hstack([X_numeric, X_category])[:, all_indices]

        if columns is not None:
            ret = pd.DataFrame(ret, columns=columns).astype(dtypes)
        else:
            ret = ret.astype(dtypes)

        return ret

class Augmenter(ABC):
    @abstractmethod
    def _augment(self, x_numeric: np.ndarray, **params):
        pass

    def resample(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 n_factor: int = 2,
                 categorical_indices: Optional[np.ndarray] = None,
                 **resample_params) -> Tuple[np.ndarray, np.ndarray]:
        xs = [x]
        ys = [y]

        for _ in range(n_factor):
            X_resample = safe_call_for_mixed_data(
                func=self._augment,
                X=x,
                categorical_indices=categorical_indices,
                **resample_params
            )
            xs.append(X_resample)
            ys.append(y)

        xx = np.vstack(xs)
        yy = np.vstack(ys) if len(y.shape) > 1 else np.hstack(ys)

        return xx, yy


class JitterAugmenter(Augmenter):
    def _augment(self, x_numeric: np.ndarray, **params):
        sigma = params['sigma']
        scale = np.random.normal(0, sigma, x_numeric.shape)
        return x_numeric + scale


class ScaleAugmenter(Augmenter):
    def _augment(self, x_numeric: np.ndarray, **params):
        sigma = params['sigma']
        scale = np.random.normal(1.0, sigma, (1, x_numeric.shape[1]))
        scale = np.matmul(np.ones((x_numeric.shape[0], 1)), scale)
        return x_numeric * scale


class MagWarpAugmenter(Augmenter):
    def _augment(self, x_numeric: np.ndarray, **params):
        sigma = params['sigma']
        knot = params['knot']
        scale = _random_curve(x_numeric, sigma, knot)
        return x_numeric * scale


class TimeWarpAugmenter(Augmenter):
    def _augment(self, x_numeric: np.ndarray, **params):
        l, d = x_numeric.shape
        sigma = params['sigma']
        knot = params['knot']

        scale = _random_curve(x_numeric, sigma, knot)
        cumsum = np.cumsum(scale, axis=0)
        t_scale = np.array([(l - 1) / cumsum[-1, i] for i in range(d)])
        warp = cumsum * t_scale
        r = np.arange(l)

        xs = [np.interp(r, warp[:, i], x_numeric[:, i]) for i in range(d)]
        xs = np.column_stack(xs)
        return xs


class RotationAugmenter(Augmenter):
    def _augment(self, x_numeric: np.ndarray, **params):
        axis = np.random.uniform(-1, 1, x_numeric.shape[1])
        angle = np.random.uniform(-np.pi, np.pi)
        return np.matmul(x_numeric, axangle2mat(axis, angle))
