from typing import List, Tuple, Any, Optional

import numpy as np
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.base import BaseEstimator, clone

from ..util import pack_label, unpack_label, correct_label


class Model(BaseEstimator):
    MODE_HIERARCHICAL = 0
    MODE_COMBINED = 1
    MODE_MULTI_CLASS = 2

    def __init__(self,
                 base_estimator: BaseEstimator,
                 name: str,
                 n_labels: int = 1,
                 mode: int = MODE_HIERARCHICAL,
                 **kwargs):
        """
        Parameters
        ----------
        n_labels : int, default = 1
            the number of outputs or labels. Generally, it is same as y.shape[1] if any.
        mode: int, default = 1 (MODE_HIERARCHICAL)
            one of mode hierarchical, combined, multi-class, and multi-label
        **kwargs
            Keyword arguments that are fed into sub-models. See notes to check conventions.

        Notes
        ---------
        Keyword arguments like:
            {
                alg_0__param_name_1: parameter name used in a first estimator,
                .
                .
                .
                alg_3__param_name_1: parameter name used in a third estimator
            }
        """
        self._name = name
        self._mode: int = mode
        self._n_labels: int = n_labels
        self._base_estimator = base_estimator
        self._estimators = []

        self._build(**kwargs)

    @property
    def estimators_(self) -> List[BaseEstimator]:
        return self._estimators

    @property
    def n_labels_(self) -> int:
        return self._n_labels

    @property
    def mode_(self) -> int:
        return self._mode

    @property
    def name_(self) -> str:
        return self._name

    @property
    def base_estimator_(self) -> Any:
        return self._base_estimator

    @property
    def is_linear_(self) -> bool:
        return hasattr(self._base_estimator, 'decision_function')

    def _build(self, **kwargs):
        self._estimators.clear()

        n_models = self._n_labels if self._mode == self.MODE_HIERARCHICAL or self._mode == self.MODE_COMBINED else 1

        for i in range(n_models):
            prefix = 'alg_{}__'.format(i)
            params = {
                k.replace(prefix, ''): v
                for k, v in kwargs.items() if k.startswith(prefix)
            }
            estimator = clone(self._base_estimator)
            estimator.set_params(**params)
            self._estimators.append(estimator)

    def get_params(self, deep=True):
        general_params = {
            'mode': self._mode,
            'n_labels': self._n_labels,
            'klass': self._base_estimator.__class__,
            'name': self.name_
        }
        sub_params = {
            'alg_{}__{}'.format(i, name): value
            for i in range(len(self._estimators))
            for name, value in self._estimators[i].get_params().items()
        }

        return {
            **general_params,
            **sub_params
        }

    def set_params(self, **params):
        if 'mode' in params:
            self._mode = params['mode']
        if 'n_labels' in params:
            self._n_labels = params['n_labels']
        if 'klass' in params:
            self._base_estimator = params['klass']()
        if 'name' in params:
            self._name = params['name']
        self._build(**params)
        return self

    def _prepare(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        prepared = []
        if self._mode == self.MODE_HIERARCHICAL:
            for i in range(y.shape[1]):
                indices = np.ones(y.shape[0], dtype=bool) if i == 0 else y[:, i - 1] != 0
                prepared.append((X[indices], y[indices, i].ravel()))
        elif self._mode == self.MODE_COMBINED:
            for i in range(y.shape[1]):
                prepared.append((X, y[:, i].ravel()))
        elif self._mode == self.MODE_MULTI_CLASS:
            prepared.append((X, pack_label(y)))

        return prepared

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            categorical_feature: Optional[np.ndarray] = None,
            resample: bool = False):
        """
        Parameters
        ----------
        X : ndarray, (n_samples, n_features)
        y : ndarray, (n_samples, n_classes)
        categorical_feature: ndarray, optional
            masked array or indices of categorical feature if any
        resample: bool
            if True, X and y are resampled by SMOTE or SMOTENC (for categorical feature)
        Returns
        -------
            Model
        """
        assert len(y.shape) > 1 and X.shape[0] == y.shape[0], 'y is multi-labeled, but y\'s shape is {}'.format(y.shape)

        for (xx, yy), estimator in zip(self._prepare(X, y), self._estimators):
            if resample:
                if categorical_feature is None or categorical_feature.shape[0] == 0:
                    xx, yy = SMOTE().fit_resample(xx, yy)
                else:
                    xx, yy = SMOTENC(categorical_features=categorical_feature).fit_resample(xx, yy)

            estimator.fit(xx, yy)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._mode == self.MODE_MULTI_LABEL:
            y = self._estimators[0].predict(X)
        elif self._mode == self.MODE_MULTI_CLASS:
            y = unpack_label(self._estimators[0].predict(X), self._n_labels)
        else:
            y = np.column_stack([e.predict(X) for e in self._estimators])
        return correct_label(y)
