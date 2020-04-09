from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, safe_entropy, combination_sum, safe_immediate_previous


class RingerModeProcessor(FeatureProcessor):


    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        ringer_data = data.loc[lambda x: x['type'].str.startswith('RINGER_MODE'), :].replace('RINGER_MODE_', '', regex=True)
        diff_data = ringer_data.loc[lambda x: x['type'] != x.shift(1)['type'], :]

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
                'RINGER': 'NORMAL'
            }
        else:
            return {
                'RINGER': str(prev['type']).upper()
            }

    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        modes = np.unique(data.loc[:, 'type'])

        win = safe_subset(
            data=data,
            from_boundary=from_pt,
            to_boundary=to_pt,
            from_col='timestamp',
            to_col='_timestamp',
            duration_col='duration'
        )

        if win is None:
            durations = {m: 0.0 for m in modes}
            frequencies = {m: 0.0 for m in modes}
        else:
            durations = {
                m: np.sum(
                    win.loc[lambda x: x['type'] == m, 'duration'].values
                ) / (to_pt - from_pt)
                for m in modes
            }
            frequencies = {
                m: len(
                    win.loc[lambda x: x['type'] == m, 'duration'].index
                )
                for m in modes
            }

        entropy_dur = safe_entropy(durations.values())

        return {
            **{'{}_DUR'.format(k): v for k, v in durations.items()},
            **{'{}_FRQ'.format(k): v for k, v in frequencies.items()},
            'ETRP_DUR': entropy_dur,
        }
