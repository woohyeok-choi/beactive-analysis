from datetime import datetime, timedelta
from typing import Dict, Union

import pandas as pd

from ._processor import FeatureProcessor

WEEKDAYS = [
    'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'
]


class TimeProcessor(FeatureProcessor):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def _previous(self, pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        time_obj = datetime.utcfromtimestamp(pt) + timedelta(hours=9)

        if 9 <= time_obj.hour < 12:
            time_slot = 'MORNING'
        elif 12 <= time_obj.hour < 15:
            time_slot = 'LUNCH'
        elif 15 <= time_obj.hour < 18:
            time_slot = 'AFTERNOON'
        elif 18 <= time_obj.hour < 21:
            time_slot = 'DINNER'
        else:
            time_slot = 'NIGHT'

        if 0 <= time_obj.weekday() < 5:
            time_is_weekday = 'TRUE'
        else:
            time_is_weekday = 'FALSE'

        time_weekday = WEEKDAYS[time_obj.weekday()]
        time_norm = (time_obj.hour * 60 + time_obj.minute) / (24 * 60)

        return {
            'SLOT': time_slot,
            'IS_WEEKDAY': time_is_weekday,
            'DAYS_OF_WEEK': time_weekday,
            'DAYTIME': time_norm
        }

    def _subset(self, from_pt: int, to_pt: int, data: pd.DataFrame) -> Dict[str, Union[str, bool, int, float]]:
        pass
