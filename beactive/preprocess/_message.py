from itertools import product
from typing import Dict, Union

import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, combination_sum


class MessageProcessor(FeatureProcessor):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.assign(
            contact=lambda x: x['contact'] != 'UNDEFINED'
        )

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
            frequencies = {
                '{}_{}'.format(box, 'CONTACT' if contact else 'UNKNOWN'): 0
                for box, contact in product(['INCOMING', 'OUTGOING'], [True, False])
            }
        else:
            frequencies = {
                '{}_{}'.format(box, 'CONTACT' if contact else 'UNKNOWN'): len(
                    win.loc[lambda x: (x['message_box'] == box) & x['contact'] == contact, :].index
                )
                for box, contact in product(['INCOMING', 'OUTGOING'], [True, False])
            }

        comb_frq = combination_sum(frequencies)

        return {
            **{'{}_FRQ'.format(k): v for k, v in comb_frq.items()},
        }
