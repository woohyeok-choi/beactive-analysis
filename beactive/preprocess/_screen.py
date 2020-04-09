from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, safe_entropy, safe_immediate_previous


class ScreenProcessor(FeatureProcessor):

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        screen_data = data.loc[lambda x: x['type'].isin(['SCREEN_ON', 'SCREEN_OFF']), :].replace(
            'SCREEN_', '',regex=True
        )
        diff_data = screen_data.loc[lambda x: x['type'] != x.shift(1)['type'], :]

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
                'SCR': 'OFF'
            }
        else:
            return {
                'SCR': str(prev['type']).upper()
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
            return {
                'OFF_DUR': 0.0,
                'OFF_FRQ': 0.0
            }
        else:
            return {
                'OFF_DUR': np.sum(
                    win.loc[lambda x: x['type'] == 'OFF', 'duration'].values
                ) / (to_pt - from_pt),
                'OFF_FRQ': len(
                    win.loc[lambda x: x['type'] == 'OFF', 'duration'].index
                )
            }

