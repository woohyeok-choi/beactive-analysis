import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, \
    precision_recall_fscore_support
from sklearn.base import clone, ClassifierMixin, BaseEstimator
from typing import Any, Callable, List, Tuple, NamedTuple, Iterable, Dict, Mapping, Union, Optional, Set
from abc import ABC, abstractmethod

import multiprocessing


def __to_single_label__(y: np.ndarray) -> np.ndarray:
    _y = np.copy(y).astype('uint8')
    for i in range(1, _y.shape[1]):
        _y[:, i] = np.where(_y[:, i - 1] == 0, 0, _y[:, i])

    return np.packbits(_y, axis=1, bitorder='little').ravel()


def __to_multi_label__(y: np.ndarray, col_len: int) -> np.ndarray:
    return np.unpackbits(y.astype('uint8').reshape(y.shape[0], 1), axis=1, bitorder='little')[:, :col_len]


def __get_sampler__(categories: Optional[List[Tuple[int, Union[Set[str], Set[bool]]]]] = None) -> Union[SMOTE, SMOTENC]:
    if categories:
        return SMOTENC(np.array([i for i, _ in categories]),
                       n_jobs=multiprocessing.cpu_count() / 2,
                       k_neighbors=3
                       )
    else:
        return SMOTE()


def __encode_categories__(x: np.ndarray,
                          categories: Optional[List[Tuple[int, Union[Set[str], Set[bool]]]]] = None) -> np.ndarray:
    if categories:
        cat_idx = np.array([i for i, _ in categories])
        x_categories = x[:, cat_idx]
        x_numeric = x[:, np.setdiff1d(np.arange(x.shape[1]), cat_idx)]
        x_encoded = OneHotEncoder(categories=[
            list(cat) for _, cat in categories
        ], sparse=False, drop='first').fit_transform(x_categories)

        return np.column_stack([x_numeric, x_encoded])
    else:
        return x


class Wrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, alg: Union[Iterable, Any]):
        self._alg = alg

    def set_params(self, **params):
        return super().set_params(**params)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._alg.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._alg.predict(x)

    def get_params(self, deep=True):
        return super().get_params(deep)

    def copy(self):
        return Wrapper(clone(self._alg))


class Score(NamedTuple):
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    precision_each: np.ndarray
    recall_each: np.ndarray
    f1_each: np.ndarray
    confusion_matrix: np.ndarray

    @classmethod
    def zeros(cls, n_labels: int) -> 'Score':
        return Score(
            accuracy=0,
            balanced_accuracy=0,
            precision=0,
            recall=0,
            f1=0,
            precision_each=np.zeros(n_labels),
            recall_each=np.zeros(n_labels),
            f1_each=np.zeros(n_labels),
            confusion_matrix=np.zeros((n_labels, n_labels))
        )

    @classmethod
    def build(cls, y_true: np.ndarray, y_predict: np.ndarray, labels: np.ndarray) -> 'Score':
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_predict, average='weighted')
        precision_each, recall_each, f1_each, _ = precision_recall_fscore_support(
            y_true, y_predict, labels=list(labels)
        )

        return Score(
            accuracy=accuracy_score(y_true, y_predict),
            balanced_accuracy=balanced_accuracy_score(y_true, y_predict),
            precision=precision,
            recall=recall,
            f1=f1,
            precision_each=precision_each,
            recall_each=recall_each,
            f1_each=f1_each,
            confusion_matrix=confusion_matrix(y_true, y_predict, labels)
        )

    def __add__(self, other):
        if isinstance(other, Score):
            return Score(
                accuracy=self.accuracy + other.accuracy,
                balanced_accuracy=self.balanced_accuracy + other.balanced_accuracy,
                precision=self.precision + other.precision,
                recall=self.recall + other.recall,
                f1=self.f1 + other.f1,
                precision_each=self.precision_each + other.precision_each,
                recall_each=self.recall_each + other.recall_each,
                f1_each=self.f1_each + other.f1_each,
                confusion_matrix=self.confusion_matrix + other.confusion_matrix
            )
        else:
            raise Exception('Only object Score can be added into other object Score.')

    def __truediv__(self, other):
        if type(other) == int or type(other) == float:
            return Score(
                accuracy=self.accuracy / other,
                balanced_accuracy=self.balanced_accuracy / other,
                precision=self.precision / other,
                recall=self.recall / other,
                f1=self.f1 / other,
                precision_each=self.precision_each / other,
                recall_each=self.recall_each / other,
                f1_each=self.f1_each / other,
                confusion_matrix=self.confusion_matrix
            )
        else:
            raise Exception('Only object Score can be divided by a number.')


class Model(ABC, ClassifierMixin, BaseEstimator):
    MODE_HIERARCHICAL = 0
    MODE_COMBINED = 1
    MODE_MULTI_CLASS = 2
    MODE_MULTI_LABEL = 3

    def __init__(self,
                 mode: int,
                 base_model: Union[List, Any],
                 categories: Optional[List[Tuple[int, Union[Set[str], Set[bool]]]]] = None):
        self._mode = mode
        self._categories = categories
        self._base_model = base_model
        self._sampler = __get_sampler__(categories)
        self._models = None

    def transform_data(self, x: np.ndarray, y: np.ndarray, resample: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        _y = np.copy(y)

        if self._mode == Model.MODE_HIERARCHICAL:
            for i in range(1, y.shape[1]):
                _y[:, i] = np.where(y[:, i - 1] == 0, -1, y[:, i])
        elif self._mode == Model.MODE_MULTI_CLASS or self._mode == Model.MODE_MULTI_LABEL:
            _y = __to_single_label__(y).reshape((y.shape[0], 1))




    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        :param x: ndarray of (n_samples, n_features)
        :param y: ndarray of (n_samples, n_classes)

        There is for cases such as:
        1) Multiple models returning binary labels
        2) Single model returning multi-class labels
        3) Single model returning multi-labels
        """
        _y = None

        if self._mode == Model.MODE_HIERARCHICAL:
            self._alg = [self._wrapper.copy() for _ in range(y.shape[1])]
            _y = np.copy(y)
            for i in range(1, y.shape[1]):
                _y[:, i] = np.where(y[:, i - 1] == 0, -1, y[:, i])
        elif self._mode == Model.MODE_COMBINED:
            self._alg = [self._wrapper.copy() for _ in range(y.shape[1])]
            _y = y
        elif self._mode == Model.MODE_MULTI_CLASS:
            self._alg = [self._wrapper.copy()]
            _y = __to_single_label__(y)
            _y = _y.reshape((_y.shape[0], 1))
        elif self._mode == Model.MODE_MULTI_LABEL:
            self._alg = [self._wrapper.copy()]
            _y = __to_single_label__(y)
            _y = _y.reshape((_y.shape[0], 1))

        for i in range(len(self._alg)):
            filter_y = _y[:, i] == -1
            x_sample, y_sample = self._sampler.fit_sample(x[~filter_y], _y[~filter_y, i].ravel())
            x_encoded = __encode_categories__(x_sample, self._categories)

            if self._mode == Model.MODE_MULTI_LABEL:
                y_sample = __to_multi_label__(y_sample, y.shape[1])
            self._alg[i].fit(x_encoded, y_sample)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_encoded = __encode_categories__(x, self._categories)
        predicts = np.column_stack([model.predict(x_encoded) for model in self._alg])

        if predicts.shape[1] > 1:
            return __to_single_label__(predicts)
        else:
            return predicts.ravel()

    def predict_each(self, x: np.ndarray) -> np.ndarray:
        x_encoded = __encode_categories__(x, self._categories)
        predicts = np.column_stack([model.predict(x_encoded) for model in self._alg])
        y = np.copy(predicts)
        for i in range(predicts.shape[1] - 1, 0, -1):
            y[:, i] = np.where(predicts[:, i - 1] == 0, 0, predicts[:, i])

        return y

    def score(self, x: np.ndarray, y: np.ndarray) -> Score:
        y_true = __to_single_label__(y)
        y_predict = self.predict(x)

        return Score.build(y_true, y_predict, np.array([2 ** i - 1 for i in range(y.shape[1] + 1)]))

    def score_each(self, x: np.ndarray, y: np.ndarray) -> List[Score]:
        if self._mode == Model.MODE_MULTI_CLASS:
            return [self.score(x, y)]
        y_predicts = self.predict_each(x)

        return [
            Score.build(
                y[:, i].ravel(),
                y_predicts[:, i].ravel(),
                np.array([0, 1])
            ) for i in range(y_predicts.shape[1])
        ]

    def get_params(self) -> List[Mapping[str, Any]]:
        return [wrapper.get_params() for wrapper in self._alg]

    def cross_validate(self, x: np.ndarray, y: np.ndarray, n_fold: int = 10, *args) -> Tuple[Score, Tuple[Any]]:
        fold = StratifiedKFold(n_splits=n_fold, shuffle=True)
        _y = __to_single_label__(y)

        score = Score.zeros(y.shape[1] + 1)

        for train, test in fold.split(x, _y):
            x_train = x[train]
            y_train = y[train]

            x_test = x[test]
            y_test = y[test]

            self.fit(x_train, y_train)

            score += self.score(x_test, y_test)

        return score / n_fold, args



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier