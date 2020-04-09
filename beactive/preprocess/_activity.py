from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, safe_entropy, safe_item


class ActivityProcessor(FeatureProcessor):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        activity = data.loc[lambda x: x['transition_type'].str.startswith('ENTER'), :]
        diff_data = activity.loc[
                    lambda x: x['transition_type'] != x.shift(1)['transition_type'], :
                    ].replace('ENTER_', '', regex=True)

        concat_data = pd.concat([
            diff_data,
            diff_data.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)

        return concat_data

    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        pass

    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        activities = ['STILL', 'WALKING', 'IN_VEHICLE', 'ON_BICYCLE', 'RUNNING']

        win = safe_subset(
            data=data,
            from_boundary=from_pt,
            to_boundary=to_pt,
            from_col='timestamp',
            to_col='_timestamp',
            duration_col='duration'
        )

        if win is None:
            durations = {activity: 0 for activity in activities}
            frequencies = {activity: 0 for activity in activities}
        else:
            durations = {
                activity: np.sum(
                    win.loc[lambda x: x['transition_type'] == activity, 'duration'].values
                ) / (to_pt - from_pt)
                for activity in activities
            }
            frequencies = {
                activity: len(win.loc[lambda x: x['transition_type'] == activity, 'duration'].index)
                for activity in activities
            }

        entropy_dur = safe_entropy(durations.values())
        entropy_frq = safe_entropy(frequencies.values())

        return {
            **{'{}_DUR'.format(k): v for k, v in durations.items()},
            **{'{}_FRQ'.format(k): v for k, v in frequencies.items()},
            'IDLE_DUR': safe_item(durations, 'IN_VEHICLE', 0.0) + safe_item(durations, 'STILL', 0.0),
            'IDLE_FRQ': safe_item(frequencies, 'IN_VEHICLE', 0.0) + safe_item(frequencies, 'STILL', 0.0),
            'ETRP_DUR': entropy_dur,
            'ETRP_FRQ': entropy_frq
        }
