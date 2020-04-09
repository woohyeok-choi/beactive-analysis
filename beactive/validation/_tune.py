from ..preprocess import ProcessedData
from ._validate import _mode_func, prepare, EvalScore
from functools import reduce
from copy import deepcopy
from typing import Dict, Any, Iterable, List, Tuple
from ray.tune import Trainable, run, choice, uniform
from ray.tune.util import get_pinned_object, pin_in_object_store
from ray.tune.schedulers import AsyncHyperBandScheduler
import numpy as np

_NON_PARAMS = [
    'alg', 'id_splits', 'id_feature_names', 'id_category_names'
]


class ParamTune(Trainable):
    def _setup(self, config):
        super()._setup(config)
        self._alg = config['alg']
        self._id_splits = config['id_splits']
        self._id_category_names = config['id_category_names']
        self._params = {k: v for k, v in config.items() if k not in _NON_PARAMS}

    def _train(self):
        func = _mode_func(self._alg)
        splits = get_pinned_object(self._id_splits)
        category_names = get_pinned_object(self._id_category_names)
        scores = []

        for subject, samples in splits.items():
            score = []
            for X_train, y_train, X_test, y_test in samples:
                s = func(X_train=X_train.copy(),
                         y_train=y_train.copy(),
                         X_test=X_test.copy(),
                         y_test=y_test.copy(),
                         n_thread=1,
                         category_names=deepcopy(category_names),
                         params=self._params)
                score.append(s)

            scores.append(EvalScore.average(score))

        avg_score = {}
        for score in scores:
            for k, v in score.items():
                if k in avg_score:
                    avg_score[k].append(v)
                else:
                    avg_score[k] = [v]

        avg_score = {
            k: np.mean(v) for k, v in avg_score.items()
        }

        return dict(
            **avg_score,
            done=True
        )

    def _save(self, tmp_checkpoint_dir):
        return {
            'alg': self._alg,
            'id_splits': self._id_splits,
            'id_category_names': self._id_category_names,
            **self._params
        }

    def _restore(self, checkpoint):
        self._alg = checkpoint.pop('alg')
        self._id_splits = checkpoint.pop('id_splits')
        self._id_category_names = checkpoint.pop('id_category_names')
        self._params = checkpoint


def tune_params(processed_data: ProcessedData,
                label: int,
                alg: str,
                btw_subject: bool,
                n_stop: int,
                logdir: str = None,
                scaler: Any = None,
                subjects: Iterable[str] = None,
                include_features: Iterable[str] = None,
                exclude_features: Iterable[str] = None,
                seed: int = None,
                verbose: int = 2,
                param_choice: Dict[str, List[Any]] = None,
                param_uniform: Dict[str, Tuple[float, float]] = None,
                n_samples: int = 1):
    splits, category_names = prepare(
        processed_data=processed_data,
        dummy_encoded=alg == 'xgb',
        label=label,
        subjects=subjects,
        is_loo=btw_subject,
        include_features=include_features,
        exclude_features=exclude_features,
        seed=seed,
        scaler=scaler
    )
    id_splits = pin_in_object_store(splits)
    id_category_names = pin_in_object_store(category_names)
    param_choice = dict() if param_choice is None else param_choice
    param_uniform = dict() if param_uniform is None else param_uniform

    config = {
        **{k: choice(v) for k, v in param_choice.items()},
        **{k: uniform(v[0], v[1]) for k, v in param_uniform.items()},
        'alg': alg,
        'id_splits': id_splits,
        'id_category_names': id_category_names
    }
    scheduler = AsyncHyperBandScheduler(
        mode='max',
        metric='test_f1_macro',
        time_attr='training_iteration'
    )
    analysis = run(ParamTune,
                   config=config,
                   name='alg-{}_label-{}'.format(alg, label),
                   stop={
                       'training_iteration': n_stop
                   },
                   num_samples=n_samples,
                   scheduler=scheduler,
                   local_dir=logdir,
                   verbose=verbose)
    return analysis
