from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, safe_immediate_previous


class BatteryPluggedProcessor(FeatureProcessor):

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        plug_data = data.assign(plugged=lambda x: x['plugged'] != 'UNDEFINED')
        diff_data = plug_data.loc[lambda x: x['plugged'] != x.shift(1)['plugged'], :]
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
            return {'IS_PLUGGED': 'FALSE'}
        else:
            return {'IS_PLUGGED': str(prev['plugged']).upper()}


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
                'UNPLUG_DUR': 0.0,
                'UNPLUG_FRQ': 0.0
            }
        else:
            return {
                'UNPLUG_DUR': np.sum(
                        win.loc[lambda x: x['plugged'] == False, 'duration'].values
                    ) / (to_pt - from_pt),
                'UNPLUG_FRQ': len(
                        win.loc[lambda x: x['plugged'] == False, 'duration'].index
                    )
            }