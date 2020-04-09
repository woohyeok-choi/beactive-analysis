from itertools import product
from typing import Dict, Union

import numpy as np
import pandas as pd

from ._processor import FeatureProcessor
from ..util import safe_subset, combination_sum


class CallLogProcessor(FeatureProcessor):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.assign(
            _timestamp=lambda x: x['timestamp'] - x['duration'],
            contact=lambda x: x['contact'] != 'UNDEFINED'
        )

    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        calling = data.loc[lambda x: (x['_timestamp'] <= pt) & (x['timestamp'] >= pt), :]
        if calling is None:
            is_calling = False
        else:
            is_calling = len(calling.index) != 0

        return {'IS_CALLING': str(is_calling).upper()}

    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        miss = safe_subset(
            data=data,
            from_boundary=from_pt,
            to_boundary=to_pt,
            from_col='timestamp'
        )

        if miss is None:
            frequencies_miss = {
                'MISS_CONTACT': 0,
                'MISS_UNKNOWN': 0
            }
        else:
            frequencies_miss = {
                'MISS_CONTACT': len(miss.loc[lambda x: x['contact'] == True, :].index),
                'MISS_UNKNOWN': len(miss.loc[lambda x: x['contact'] == False, :].index),
            }

        win = safe_subset(
            data=data,
            from_boundary=from_pt,
            to_boundary=to_pt,
            from_col='_timestamp',
            to_col='timestamp',
            duration_col='duration'
        )

        if win is None:
            durations_call = {
                '{}_{}'.format(box, 'CONTACT' if contact else 'UNKNOWN'): 0
                for box, contact in product(['INCOMING', 'OUTGOING'], [True, False])
            }
            frequencies_call = {
                '{}_{}'.format(box, 'CONTACT' if contact else 'UNKNOWN'): 0
                for box, contact in product(['INCOMING', 'OUTGOING'], [True, False])
            }
        else:
            durations_call = {
                '{}_{}'.format(box, 'CONTACT' if contact else 'UNKNOWN'): np.sum(
                    win.loc[lambda x: (x['type'] == box) & x['contact'] == contact, 'duration'].values
                ) / (from_pt - to_pt)
                for box, contact in product(['INCOMING', 'OUTGOING'], [True, False])
            }
            frequencies_call = {
                '{}_{}'.format(box, 'CONTACT' if contact else 'UNKNOWN'): len(
                    win.loc[lambda x: (x['type'] == box) & x['contact'] == contact, 'duration'].index
                )
                for box, contact in product(['INCOMING', 'OUTGOING'], [True, False])
            }

        comb_dur = combination_sum(durations_call)
        comb_frq = combination_sum({**frequencies_call, **frequencies_miss})

        return {
            **{'{}_DUR'.format(k): v for k, v in comb_dur.items()},
            **{'{}_FRQ'.format(k): v for k, v in comb_frq.items()},
        }
