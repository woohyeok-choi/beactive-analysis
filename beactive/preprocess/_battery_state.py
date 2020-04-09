from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, safe_entropy, safe_immediate_previous, safe_mean, safe_std, safe_min, safe_max


class BatteryStateProcessor(FeatureProcessor):

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        prepared = data.assign(level=lambda x: x['level'] / 100.0,
                               temperature=lambda x: np.clip(x['temperature'], a_min=0, a_max=500) / 500)
        return prepared

    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        prev = safe_immediate_previous(
            data=data,
            data_point=pt,
            col='timestamp'
        )
        if prev is None:
            return {
                'LEV': 100.0,
                'TEMP': 0.5
            }
        else:
            return {
                'LEV': prev['level'],
                'TEMP': prev['temperature']
            }


    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        win = safe_subset(
            data=data,
            from_boundary=from_pt,
            to_boundary=to_pt,
            from_col='timestamp'
        )
        if win is None:
            return {
                'MEAN_LEV': 100.0,
                'STD_LEV': 0.0,
                'MEAN_TEMP': 0.5,
                'STD_TEMP': 0.0,
                'SLOPE_LEV': 0.0,
                'SLOPE_TEMP': 0.0
            }
        else:

            levels = win.loc[:, 'level'].dropna().to_numpy()
            temperatures = win.loc[:, 'temperature'].dropna().to_numpy()

            try:
                slope_lev = np.polyfit(x=np.arange(levels.shape[0]), y=levels - np.mean(levels), deg=1)[0]
            except:
                slope_lev = 0.0

            try:
                slope_temp = np.polyfit(x=np.arange(temperatures.shape[0]), y=temperatures - np.mean(temperatures), deg=1)[0]
            except:
                slope_temp = 0.0

            return {
                'LEV_MEAN': safe_mean(levels),
                'LEV_STD': safe_std(levels),
                'TEMP_MEAN': safe_mean(temperatures),
                'TEMP_STD': safe_std(temperatures),
                'SLOPE_LEV': slope_lev,
                'SLOPE_TEMP': slope_temp
            }
