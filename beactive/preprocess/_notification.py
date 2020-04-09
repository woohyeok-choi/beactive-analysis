from typing import Dict, Union

import pandas as pd

from ._misc import APP_CATEGORIES
from ._processor import FeatureProcessor
from ..util import safe_subset, safe_entropy


class NotificationProcessor(FeatureProcessor):

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
            frequencies = {category: 0 for category in APP_CATEGORIES.keys()}
        else:
            frequencies = {
                category: len(
                    win.loc[lambda x: x['package_name'].isin(apps), :].index
                )
                for category, apps in APP_CATEGORIES.items()
            }
        entropy_frq_all = safe_entropy(win.groupby('package_name')['package_name'].count())
        frq_all = len(win.index)

        return {
            **{'{}_FRQ'.format(k): v for k, v in frequencies.items()},
            'ALL_FRQ': frq_all,
            'ETRP_FRQ_ALL': entropy_frq_all
        }
