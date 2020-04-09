from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor


class SelfReportProcessor(FeatureProcessor):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        return {
            'LOC': np.asscalar(data.loc[lambda x: x['delivered_time'] == pt, 'code_location']),
            'ACT': np.asscalar(data.loc[lambda x: x['delivered_time'] == pt, 'code_activity']),
            'SOC': np.asscalar(data.loc[lambda x: x['delivered_time'] == pt, 'code_social']),
            'FCS': float(np.asscalar(data.loc[lambda x: x['delivered_time'] == pt, 'focused'])),
            'ATV': float(np.asscalar(data.loc[lambda x: x['delivered_time'] == pt, 'activeness'])),
        }

    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        pass