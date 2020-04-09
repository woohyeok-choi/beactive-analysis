from typing import Dict, Union

import numpy as np
import pandas as pd

from ._misc import APP_CATEGORIES
from ._processor import FeatureProcessor
from ..util import safe_subset, safe_entropy, safe_item


class AppUsageProcessor(FeatureProcessor):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        app_data = data.loc[lambda x: x['type'].isin(['MOVE_TO_FOREGROUND', 'MOVE_TO_BACKGROUND']), :]
        diff_data = app_data.loc[
                    lambda x: (x['type'] != x.shift(1)['type']) | (x['package_name'] != x.shift(1)['package_name']), :
                    ]
        concat_data = pd.concat([
            diff_data,
            diff_data.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)

        return concat_data

    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        pass

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
            durations = {category: 0 for category in APP_CATEGORIES.keys()}
            frequencies = {category: 0 for category in APP_CATEGORIES.keys()}
            frq_all = 0
            dur_all = 0
            entropy_dur_all = 0
            entropy_frq_all = 0
        else:
            foreground = win.loc[lambda x: x['type'] == 'MOVE_TO_FOREGROUND', :]
            durations = {
                category: np.sum(
                    foreground.loc[lambda x: x['package_name'].isin(apps), 'duration'].values
                ) / (to_pt - from_pt)
                for category, apps in APP_CATEGORIES.items()
            }
            frequencies = {
                category: len(
                    foreground.loc[lambda x: x['package_name'].isin(apps), 'duration'].index
                )
                for category, apps in APP_CATEGORIES.items()
            }
            frq_all = len(foreground.index)
            dur_all = foreground.loc[:, 'duration'].sum()
            entropy_dur_all = safe_entropy(foreground.groupby('package_name')['duration'].sum())
            entropy_frq_all = safe_entropy(foreground.groupby('package_name')['duration'].count())

        return {
            **{'{}_DUR'.format(k): v for k, v in durations.items()},
            **{'{}_FRQ'.format(k): v for k, v in frequencies.items()},
            'ALL_FRQ': frq_all,
            'ALL_DUR': dur_all,
            'ETRP_DUR_ALL': entropy_dur_all,
            'ETRP_FRQ_ALL': entropy_frq_all
        }
