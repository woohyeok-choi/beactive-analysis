from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, safe_entropy, safe_immediate_previous, safe_item


class ConnectivityProcessor(FeatureProcessor):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        conn_data = data.assign(
            type=lambda x: np.where(
                x['type'].str.contains('WIFI'), 'WIFI', np.where(
                    x['type'].str.contains('MOBILE'), 'MOBILE', 'UNDEFINED'
                )
            )
        ).loc[lambda x: x['type'].isin(['WIFI', 'MOBILE', 'UNDEFINED']), :]

        diff_data = conn_data.loc[lambda x: x['type'] != x.shift(1)['type'], :]
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
        if pt is None:
            return {
                'CONN_TYPE': 'UNDEFINED'
            }
        else:
            return {
                'CONN_TYPE': str(prev['type'])
            }

    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        win = safe_subset(
            data=data,
            from_boundary=from_pt,
            to_boundary=to_pt,
            from_col='timestamp',
            to_col='_timestamp',
            duration_col='duration'
        )
        if win is None:
            durations = {
                'WIFI': 0,
                'MOBILE': 0,
            }
        else:
            conns = ['WIFI', 'MOBILE']
            durations = {
                conn: np.sum([
                    win.loc[lambda x: x['type'] == conn, 'duration'].values
                ]) / (to_pt - from_pt)
                for conn in conns
            }

        entropy_dur = safe_entropy(durations.values())

        return {
            **{'{}_DUR'.format(k): v for k, v in durations.items()},
            'ETRP_DUR': entropy_dur,
        }
