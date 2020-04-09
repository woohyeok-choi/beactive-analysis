from abc import ABC, abstractmethod
from time import perf_counter
from typing import Iterable, NamedTuple, Union, Optional, Dict, Tuple, Any
from ..transform import SafeTransformer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import ray
from datetime import datetime

from ..util import log

PERIODS_FORMAT = [
    ('YR', 60 * 60 * 24 * 365),
    ('MONTH', 60 * 60 * 24 * 30),
    ('DAY', 60 * 60 * 24),
    ('HR', 60 * 60),
    ('MIN', 60),
    ('SEC', 1)
]


def _format_window(size_sec: int):
    if size_sec <= 0:
        return 'CURRENT'

    if size_sec == np.inf:
        return 'SOFAR'

    for form, seconds in PERIODS_FORMAT:
        if size_sec >= seconds:
            val, _ = divmod(size_sec, seconds)
            return '{}{}'.format(val, form)

    return '{}{}'.format(size_sec, 'SEC')


def _safe_dict(d: Optional[Dict]) -> Dict:
    return {} if d is None else d


class Feature(NamedTuple):
    point: int
    name: str
    value: Union[float, int, str, bool]
    group: Optional[str] = None

    @property
    def is_category_(self):
        return type(self.value) == str or type(self.value) == bool


class ProcessedData(NamedTuple):
    data: pd.DataFrame
    point_col: str
    group_col: str
    feature_cols: Iterable[str]
    label_cols: Iterable[str]
    category_cols: Iterable[str]

    def _non_feature_cols(self):
        return list(self.label_cols) + [self.point_col, self.group_col]

    @property
    def data_(self) -> pd.DataFrame:
        return self.data

    @property
    def feature_(self) -> pd.DataFrame:
        return self.data.loc[:, lambda x: ~x.columns.isin(self._non_feature_cols())]

    @property
    def label_(self) -> pd.DataFrame:
        return self.data.loc[:, list(self.label_cols)]

    @property
    def group_(self) -> pd.Series:
        return self.data.loc[:, self.group_col]

    @property
    def point_(self) -> pd.Series:
        return self.data.loc[:, self.point_col]



class FeatureProcessor(ABC):
    def __init__(self,
                 window_sizes: Iterable[int],
                 data_points: Iterable[int],
                 include_daily: bool = False,
                 group: str = None,
                 prefix: str = None):
        self._data_points = data_points
        self._window_sizes = window_sizes
        self._group = group
        self._prefix = '' if prefix is None else prefix
        self._include_daily = include_daily

    @property
    def group_(self):
        return self._group

    @abstractmethod
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        pass

    @abstractmethod
    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        pass

    def process(self, data: pd.DataFrame) -> Iterable[Feature]:
        ret = []
        prepared = self._preprocess(data)

        for point in self._data_points:
            prev = {
                '#'.join([self._prefix, _format_window(0), k]): v
                for k, v in _safe_dict(self._previous(pt=point, data=prepared)).items()
            }
            subsets = {}
            for window_size in self._window_sizes:
                for k, v in _safe_dict(self._subset(from_pt=point - window_size, to_pt=point, data=prepared)).items():
                    subsets['#'.join([self._prefix, _format_window(window_size), k])] = v

            daily_subsets = {}
            if self._include_daily:
                from_pt = datetime.fromtimestamp(point).replace(hour=0, minute=0, second=0, microsecond=1).timestamp()
                for k, v in _safe_dict(self._subset(from_pt=int(from_pt), to_pt=point, data=prepared)).items():
                    daily_subsets['#'.join([self._prefix, 'TODAY', k])] = v

            features = [
                Feature(point=point, name=k, value=v, group=self._group)
                for k, v in {**prev, **subsets, **daily_subsets}.items()
            ]
            ret.extend(features)

        return ret


@ray.remote
def _process_feature(proc_data_pairs: Iterable[Tuple[FeatureProcessor, pd.DataFrame]],
                     group: str,
                     col_point: str,
                     col_group: str) -> Tuple[pd.DataFrame, Iterable[str]]:
    features = []
    for proc, data in proc_data_pairs:
        features.extend(proc.process(data))

    feature_tuples = [(feature.point, feature.name, feature.value) for feature in features]
    categories = set([feature.name for feature in features if feature.is_category_])

    df = pd.DataFrame(
        feature_tuples, columns=[col_point, 'name', 'value']
    ).pivot(
        index=col_point, columns='name', values='value'
    ).reset_index().assign(**{col_group: group})

    return df, categories


def process_feature(proc_data_pairs: Iterable[Tuple[FeatureProcessor, pd.DataFrame]],
                    ref_data: pd.DataFrame,
                    col_labels: Iterable[str],
                    col_point: str,
                    col_group: str) -> ProcessedData:
    _t = perf_counter()
    jobs = {}

    for proc, df in proc_data_pairs:
        if proc.group_ in jobs:
            jobs[proc.group_].append((proc, df))
        else:
            jobs[proc.group_] = [(proc, df)]
    log('Complete to build jobs across groups.', perf_counter() - _t)
    _t = perf_counter()

    job_ids = [
        _process_feature.remote(
            proc_data_pairs=job,
            group=group,
            col_point=col_point,
            col_group=col_group
        )
        for group, job in jobs.items()
    ]
    log('Complete to deliver remote jobs.', perf_counter() - _t)
    _t = perf_counter()

    job_results = ray.get(job_ids)

    df = pd.concat([df for df, _ in job_results], axis=0)
    categories = set([cat for _, categories in job_results for cat in categories])

    log('Complete to extract features.', perf_counter() - _t)
    _t = perf_counter()

    ref_data = ref_data.loc[:, list(col_labels) + [col_point, col_group]]

    joined = pd.merge(
        left=df, right=ref_data,
        on=[col_group, col_point],
        how='inner'
    )
    log('Complete to merge feature and label dataframes', perf_counter() - _t)

    return ProcessedData(
        data=joined,
        feature_cols=df.columns,
        label_cols=col_labels,
        group_col=col_group,
        point_col=col_point,
        category_cols=categories
    )
