from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, safe_mean, safe_std


class DataTrafficProcessor(FeatureProcessor):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        pass

    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        win = safe_subset(
            data=data,
            from_boundary=from_pt,
            to_boundary=to_pt,
            from_col='timestamp'
        )

        if win is None:
            return {
                'SUM_RX': 0.0,
                'SUM_TX': 0.0,
                'MEAN_RX': 0.0,
                'MEAN_TX': 0.0,
                'STD_RX': 0.0,
                'STD_TX': 0.0
            }
        else:
            rx = win.loc[lambda x: x['rx_kb'] > 0, 'rx_kb'].values
            tx = win.loc[lambda x: x['tx_kb'] > 0, 'tx_kb'].values

            return {
                'SUM_RX': np.sum(rx),
                'SUM_TX': np.sum(tx),
                'MEAN_RX': safe_mean(rx),
                'MEAN_TX': safe_mean(tx),
                'STD_RX': safe_std(rx),
                'STD_TX': safe_std(tx)
            }
