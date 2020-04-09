from typing import Dict, Union, Iterable

import numpy as np
import pandas as pd
from poi import PoiCluster

from ._processor import FeatureProcessor
from ..util import safe_subset, safe_entropy, safe_immediate_previous


class LocationProcessor(FeatureProcessor):

    def __init__(self, window_sizes: Iterable[int],
                 data_points: Iterable[int],
                 group: str = None,
                 prefix: str = None,
                 include_daily: bool = False,
                 n_labels: int = 10,
                 d_max: int = 100,
                 r_max: int = 500,
                 t_min: int = 15 * 60,
                 t_max: int = 6 * 60 * 60):
        super().__init__(window_sizes, data_points, include_daily, group, prefix)

        self._n_labels = n_labels
        self._cluster = PoiCluster(d_max=d_max, r_max=r_max, t_max=t_max, t_min=t_min)

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        locs = np.radians(data.loc[:, ['latitude', 'longitude']].to_numpy())
        times = data.loc[:, 'timestamp'].values

        labels = self._cluster.fit_predict(locs, times)
        unique_label, counts = np.unique(labels[labels != 'NONE'], return_counts=True)

        top_labels = unique_label[np.argsort(counts)[-self._n_labels:]]
        top_labels = np.flip(top_labels)

        norm_labels = {
            'label': {
                label: 'TOP-{:02d}-PLACE'.format(idx + 1) for idx, label in enumerate(top_labels)
            }
        }
        clustered_data = data.assign(
            label=np.where(np.isin(labels, top_labels), labels, 'UNDEFINED')
        ).replace(norm_labels)

        diff_data = clustered_data.loc[lambda x: x['label'] != x.shift(1)['label'], :]

        concat_data = pd.concat([
            diff_data,
            diff_data.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)

        return concat_data

    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        prev = safe_immediate_previous(
            data=data,
            data_point=pt,
            col='timestamp'
        )
        if prev is None:
            return {
                'LOC': 'UNDEFINED'
            }
        else:
            return {
                'LOC': str(prev['label']).upper()
            }

    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        labels = [label for label in np.unique(data.loc[:, 'label']) if label != 'UNDEFINED']

        win = safe_subset(
            data=data,
            from_boundary=from_pt,
            to_boundary=to_pt,
            from_col='timestamp',
            to_col='_timestamp',
            duration_col='duration'
        )
        if win is None:
            durations = {l: 0.0 for l in labels}
            frequencies = {l: 0.0 for l in labels}
        else:
            durations = {
                l: np.sum(
                    win.loc[lambda x: x['label'] == l, 'duration'].values
                ) / (to_pt - from_pt) for l in labels
            }
            frequencies = {
                l: len(
                    win.loc[lambda x: x['label'] == l, 'duration'].index
                ) for l in labels
            }

        entropy_dur = safe_entropy(durations.values())
        entropy_frq = safe_entropy(frequencies.values())

        return {
            **{'{}_DUR'.format(k): v for k, v in durations.items()},
            **{'{}_FRQ'.format(k): v for k, v in frequencies.items()},
            'ETRP_DUR': entropy_dur,
            'ETRP_FRQ': entropy_frq
        }
