from typing import Dict, Optional, Any, List

class TimeSeriesModel:
    def __init__(self,
                 alg: str,
                 apply_filtering: bool,
                 update_interval_in_days: int,
                 cutoff_prob_neg: float,
                 cutoff_prob_pos: float,
                 category_names: Optional[List[str]] = None,
                 perc_model_params: Optional[Dict[str, Any]] = None,
                 aval_model_params: Optional[Dict[str, Any]] = None,
                 adhr_model_params: Optional[Dict[str, Any]] = None,
                 ):
        self._alg = alg
        self._apply_filtering = apply_filtering
        self._category_names = category_names
        self._update_interval_in_days = update_interval_in_days
        self._cutoff_prob_neg = cutoff_prob_neg
        self._cutoff_prob_pos = cutoff_prob_pos

        self._perc_model_params = perc_model_params
        self._aval_model_params = aval_model_params
        self._adhr_model_params = adhr_model_params

        self._perc_model = None
        self._aval_model = None
        self._adhr_model = None

    @property
    def perc_model_(self):
        return self._perc_model

    @property
    def aval_model_(self):
        return self._aval_model

    @property
    def adhr_model_(self):
        return self._adhr_model