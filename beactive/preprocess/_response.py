from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, safe_mean


class ResponseProcessor(FeatureProcessor):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        pass

    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:

        win = safe_subset(
            data=data,
            from_boundary=from_pt,
            to_boundary=to_pt,
            from_col='delivered_time'
        )

        if win is None:
            return {
                'PERC_RATIO': 0.0,
                'AVAL_RATIO': 0.0,
                'ADHR_RATIO': 0.0,
                'AVAL_FILT_RATIO': 0.0,
                'ADHR_FILT_RATIO': 0.0,
            }
        else:
            perc = win.loc[:, 'perception'].values
            aval = win.loc[:, 'availability'].values
            adhr = win.loc[:, 'adherence'].values
            aval_filt = aval[perc == 'Y']
            adhr_filt = adhr[aval == 'Y']

            return {
                'PERC_RATIO': safe_mean(np.where(perc == 1, 1.0, 0.0)),
                'AVAL_RATIO': safe_mean(np.where(aval == 1, 1.0, 0.0)),
                'ADHR_RATIO': safe_mean(np.where(adhr == 1, 1.0, 0.0)),
                'AVAL_FILT_RATIO': safe_mean(np.where(aval_filt == 1, 1.0, 0.0)),
                'ADHR_FILT_RATIO': safe_mean(np.where(adhr_filt == 1, 1.0, 0.0)),
            }
