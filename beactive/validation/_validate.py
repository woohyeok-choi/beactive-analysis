from ..preprocess import ProcessedData
from typing import Dict, Optional, NamedTuple, Union, Any, Iterable, List, Tuple, Callable
from multiprocessing import cpu_count
import lightgbm as gbm
import xgboost as xgb
import catboost as cab
from catboost import EFstrType
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, accuracy_score, roc_curve
import pandas as pd
import shap


class EvalScore(NamedTuple):
    train_true: np.ndarray
    train_pred: np.ndarray
    test_true: np.ndarray
    test_pred: np.ndarray
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_set: Union[gbm.Dataset, xgb.DMatrix, cab.Pool]
    test_set: Union[gbm.Dataset, xgb.DMatrix, cab.Pool]
    booster: Optional[Union[xgb.Booster, gbm.Booster, cab.CatBoostClassifier]]

    @property
    def train_acc_(self) -> float:
        return accuracy_score(y_true=self.train_true, y_pred=np.where(self.train_pred > .5, 1, 0))

    @property
    def test_acc_(self) -> float:
        return accuracy_score(y_true=self.test_true, y_pred=np.where(self.test_pred > .5, 1, 0))

    @property
    def train_bal_acc_(self) -> float:
        return balanced_accuracy_score(y_true=self.train_true, y_pred=np.where(self.train_pred > .5, 1, 0))

    @property
    def test_bal_acc_(self) -> float:
        return balanced_accuracy_score(y_true=self.test_true, y_pred=np.where(self.test_pred > .5, 1, 0))

    @property
    def train_f1_pos_(self):
        return f1_score(y_true=self.train_true, y_pred=np.where(self.train_pred > .5, 1, 0), pos_label=1)

    @property
    def train_f1_neg_(self):
        return f1_score(y_true=self.train_true, y_pred=np.where(self.train_pred > .5, 1, 0), pos_label=0)

    @property
    def test_f1_pos_(self):
        return f1_score(y_true=self.test_true, y_pred=np.where(self.test_pred > .5, 1, 0), pos_label=1)

    @property
    def test_f1_neg_(self):
        return f1_score(y_true=self.test_true, y_pred=np.where(self.test_pred > .5, 1, 0), pos_label=0)

    @property
    def train_f1_macro_(self) -> float:
        return f1_score(y_true=self.train_true, y_pred=np.where(self.train_pred > .5, 1, 0), average='macro')

    @property
    def test_f1_macro_(self) -> float:
        return f1_score(y_true=self.test_true, y_pred=np.where(self.test_pred > .5, 1, 0), average='macro')

    @property
    def train_f1_weighted_(self) -> float:
        return f1_score(y_true=self.train_true, y_pred=np.where(self.train_pred > .5, 1, 0), average='weighted')

    @property
    def test_f1_weighted_(self) -> float:
        return f1_score(y_true=self.test_true, y_pred=np.where(self.test_pred > .5, 1, 0), average='weighted')

    @property
    def train_roc_auc_(self) -> float:
        return roc_auc_score(y_true=self.train_true, y_score=self.train_pred)

    @property
    def test_roc_auc_(self) -> float:
        return roc_auc_score(y_true=self.test_true, y_score=self.test_pred)

    @property
    def train_roc_curve_(self) -> Tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = roc_curve(y_true=self.train_true, y_score=self.train_pred)
        return fpr, tpr

    @property
    def test_roc_curve_(self) -> Tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = roc_curve(y_true=self.test_true, y_score=self.test_pred)
        return fpr, tpr

    def shap_values(self, use_test_set: bool = True) -> Tuple[float, np.ndarray, Iterable[str]]:
        if isinstance(self.booster, xgb.Booster):
            explainer = shap.TreeExplainer(self.booster)
            shap_values = explainer.shap_values(
                self.test_set if use_test_set else self.train_set,
                tree_limit=self._tree_limit()
            )
            expected_value = explainer.expected_value
            feature_names = self.test_set.feature_names if use_test_set else self.train_set.feature_names
        elif isinstance(self.booster, gbm.Booster):
            explainer = shap.TreeExplainer(self.booster)
            _, shap_values = explainer.shap_values(
                self.test_df if use_test_set else self.train_df,
                tree_limit=self._tree_limit()
            )
            expected_value = explainer.expected_value
            feature_names = self.test_set.feature_name if use_test_set else self.train_set.feature_name
        elif isinstance(self.booster, cab.CatBoostClassifier):
            shap_values = self.booster.get_feature_importance(
                data=self.test_set if use_test_set else self.train_set,
                type=EFstrType.ShapValues
            )
            expected_value = shap_values[0, -1]
            shap_values = shap_values[:, :-1]
            feature_names = list(self.test_df.columns)
        else:
            raise Exception(
                'self.booster should be one of \'xgboost.Booster\', '
                '\'lightgbm.Booster\', or \'catboost.CatBoostClassfier\'.'
            )

        return expected_value, shap_values, feature_names

    def _tree_limit(self):
        if hasattr(self.booster, 'best_ntree_limit'):
            return getattr(self.booster, 'best_ntree_limit')
        elif hasattr(self.booster, 'best_iteration'):
            return getattr(self.booster, 'best_iteration')
        else:
            return getattr(self.booster, 'best_iteration_')

    @classmethod
    def average(cls, scores: Iterable['EvalScore']) -> Dict[str, float]:
        ret = dict(
            train_acc=[],
            test_acc=[],
            train_bal_acc=[],
            test_bal_acc=[],
            train_f1_pos=[],
            train_f1_neg=[],
            test_f1_pos=[],
            test_f1_neg=[],
            train_f1_macro=[],
            test_f1_macro=[],
            train_f1_weighted=[],
            test_f1_weighted=[],
            train_roc_auc=[],
            test_roc_auc=[]
        )
        for score in scores:
            ret['train_acc'].append(score.train_acc_)
            ret['test_acc'].append(score.test_acc_)
            ret['train_bal_acc'].append(score.train_bal_acc_)
            ret['test_bal_acc'].append(score.test_bal_acc_)
            ret['train_f1_pos'].append(score.train_f1_pos_)
            ret['train_f1_neg'].append(score.train_f1_neg_)
            ret['test_f1_pos'].append(score.test_f1_pos_)
            ret['test_f1_neg'].append(score.test_f1_neg_)
            ret['train_f1_macro'].append(score.train_f1_macro_)
            ret['test_f1_macro'].append(score.test_f1_macro_)
            ret['train_f1_weighted'].append(score.train_f1_weighted_)
            ret['test_f1_weighted'].append(score.test_f1_weighted_)
            ret['train_roc_auc'].append(score.train_roc_auc_)
            ret['test_roc_auc'].append(score.test_roc_auc_)

        return {
            k: np.mean(v) for k, v in ret.items()
        }

    @classmethod
    def abs_average_shap_values(cls, scores: Iterable['EvalScore'], use_test_set: bool = True) -> Dict[str, float]:
        ret = {}

        for score in scores:
            _, shap_values, feature_names = score.shap_values(use_test_set)
            avg_shap_values = np.mean(abs(shap_values), axis=0)
            for k, v in zip(feature_names, avg_shap_values):
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]

        return {
            k: np.mean(v) for k, v in ret.items()
        }


def prepare(processed_data: ProcessedData,
            dummy_encoded: bool,
            label: int,
            is_loo: bool,
            apply_filtering: bool = True,
            subjects: Iterable[str] = None,
            include_features: Iterable[str] = None,
            exclude_features: Iterable[str] = None,
            scaler: Any = None,
            seed: int = None):
    features = processed_data.feature_
    labels = processed_data.label_
    GROUP = processed_data.group_
    CATEGORY_NAMES = processed_data.category_cols

    # Select subset of features
    if include_features is not None:
        include_mask = features.columns.isin(include_features) if include_features is not None else np.ones(
            features.shape[1], dtype=bool)
        features = features.loc[:, include_mask]
    elif exclude_features is not None:
        exclude_mask = features.columns.isin(exclude_features) if exclude_features is not None else np.zeros(
            features.shape[1], dtype=bool)
        features = features.loc[:, ~exclude_mask]

    COLUMNS = features.columns

    X = features.fillna(0.0).to_numpy()
    Y = labels.to_numpy()
    GROUP = GROUP.to_numpy()

    # Build row-wise mask
    if apply_filtering:
        label_masks = Y[:, label - 1] == 1 if label > 0 else np.ones(Y.shape[0], dtype=bool)
    else:
        label_masks = np.ones(Y.shape[0], dtype=bool)
    y = Y[:, label].ravel()

    # Build numeric and categorical masks
    categorical_masks = COLUMNS.isin(CATEGORY_NAMES)
    numeric_masks = ~categorical_masks

    # Build categorical encoder
    encoder = OneHotEncoder(dtype='uint8', sparse=False) if dummy_encoded else OrdinalEncoder(dtype='uint8')
    categorical_data = X[:, categorical_masks]
    encoder.fit(categorical_data)

    # Build feature names
    categorical_names = encoder.get_feature_names(COLUMNS[categorical_masks]) if dummy_encoded else COLUMNS[
        categorical_masks]
    categorical_names = [col for col in categorical_names]

    numeric_names = COLUMNS[numeric_masks].to_numpy()
    numeric_names = [col for col in numeric_names]

    subjects = np.unique(GROUP) if subjects is None else subjects

    # Build k-fold or loo splits.
    FOLDS = {}
    for subject in subjects:
        FOLDS[subject] = []
        try:
            if is_loo:
                X_train = X[(GROUP != subject) & label_masks]
                y_train = y[(GROUP != subject) & label_masks]
                X_test = X[(GROUP == subject) & label_masks]
                y_test = y[(GROUP == subject) & label_masks]
                FOLDS[subject].append((X_train, y_train, X_test, y_test))
            else:
                X_sub = X[(GROUP == subject) & label_masks]
                y_sub = y[(GROUP == subject) & label_masks]

                for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_sub, y_sub):
                    X_train = X_sub[train]
                    y_train = y_sub[train]
                    X_test = X_sub[test]
                    y_test = y_sub[test]

                    FOLDS[subject].append((X_train, y_train, X_test, y_test))
        except Exception:
            print('[ERROR] Failure to build folds for subject: {}'.format(subject))
            del FOLDS[subject]

    # Resample minority class
    FOLDS_RESAMPLE = {}
    for subject, splits in FOLDS.items():
        FOLDS_RESAMPLE[subject] = []
        try:
            for X_train, y_train, X_test, y_test in splits:
                X_resample, y_resample = _resample(
                    X=X_train,
                    y=y_train,
                    category_masks=categorical_masks,
                    seed=seed
                )
                FOLDS_RESAMPLE[subject].append((X_resample, y_resample, X_test, y_test))
        except Exception:
            print('[ERROR] Failure to over-sample for subject: {}'.format(subject))
            del FOLDS_RESAMPLE[subject]

    # Scaling numeric and categorical data
    FOLDS_PROCESSED = {}
    for subject, splits in FOLDS_RESAMPLE.items():
        FOLDS_PROCESSED[subject] = []

        for X_train, y_train, X_test, y_test in splits:
            X_train_cat = X_train[:, categorical_masks]
            X_test_cat = X_test[:, categorical_masks]

            X_train_num = X_train[:, numeric_masks]
            X_test_num = X_test[:, numeric_masks]

            X_train_cat = encoder.transform(X_train_cat)
            X_test_cat = encoder.transform(X_test_cat)

            if scaler is not None:
                scaler.fit(X_train_num)

                X_train_num = scaler.transform(X_train_num)
                X_test_num = scaler.transform(X_test_num)

            X_train_arr = np.column_stack([X_train_cat, X_train_num])
            X_test_arr = np.column_stack([X_test_cat, X_test_num])

            dtypes = {
                **{k: 'uint8' for k in categorical_names},
                **{k: 'float32' for k in numeric_names}
            }

            X_train = pd.DataFrame(X_train_arr, columns=np.hstack([categorical_names, numeric_names])).astype(dtypes)
            X_test = pd.DataFrame(X_test_arr, columns=np.hstack([categorical_names, numeric_names])).astype(dtypes)

            FOLDS_PROCESSED[subject].append((X_train, y_train, X_test, y_test))

    return FOLDS_PROCESSED, categorical_names


def _mode_func(alg: str) -> Callable:
    if alg == 'xgb':
        func = _xgb
    elif alg == 'gbm':
        func = _gbm
    elif alg == 'cab':
        func = _cab
    else:
        raise Exception('param \'alg\' should be one of [\'xgb\', \'gbm\', \'cab\']')
    return func


def cross_val(processed_data: ProcessedData,
              label: int,
              alg: str,
              is_loo: bool,
              apply_filtering: bool = True,
              scaler: Any = None,
              subjects: Iterable[str] = None,
              include_features: Iterable[str] = None,
              exclude_features: Iterable[str] = None,
              seed: int = None,
              params: Dict[str, Any] = None) -> Dict[str, Iterable[EvalScore]]:
    func = _mode_func(alg)
    folds, category_names = prepare(
        processed_data=processed_data,
        dummy_encoded=alg == 'xgb',
        label=label,
        apply_filtering=apply_filtering,
        subjects=subjects,
        is_loo=is_loo,
        include_features=include_features,
        exclude_features=exclude_features,
        seed=seed,
        scaler=scaler
    )
    scores = {}

    for subject, splits in folds.items():
        eval_score = []
        for X_train, y_train, X_test, y_test in splits:
            s = func(X_train=X_train,
                     y_train=y_train,
                     X_test=X_test,
                     y_test=y_test,
                     category_names=category_names,
                     params=params)
            eval_score.append(s)

        scores[subject] = eval_score

    return scores


def _resample(X: Union[pd.DataFrame, np.ndarray],
              y: Union[pd.Series, np.ndarray],
              category_masks: np.ndarray,
              seed: int) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    if isinstance(X, pd.DataFrame):
        X_dtypes = X.dtypes.to_dict()
        X_columns = X.columns
    else:
        X_dtypes = X.dtype
        X_columns = None

    if isinstance(y, pd.Series):
        y_dtypes = y.dtype
        y_columns = y.name
    else:
        y_dtypes = y.dtype
        y_columns = None

    if np.any(category_masks):
        X_resample, y_resample = SMOTENC(categorical_features=category_masks, random_state=seed).fit_resample(X, y)
    else:
        X_resample, y_resample = SMOTE(random_state=seed).fit_resample(X, y)

    if X_columns is not None:
        X_resample = pd.DataFrame(X_resample, columns=X_columns).astype(X_dtypes)
    else:
        X_resample = X_resample.astype(X_dtypes)

    if y_columns is not None:
        y_resample = pd.Series(y_resample, name=y_columns).astype(y_dtypes)
    else:
        y_resample = y_resample.astype(y_dtypes)

    return X_resample, y_resample


def _xgb(X_train: Union[pd.DataFrame, np.ndarray],
         y_train: Union[pd.Series, np.ndarray],
         X_test: Union[pd.DataFrame, np.ndarray],
         y_test: Union[pd.Series, np.ndarray],
         category_names: Optional[List[str]] = None,
         n_thread: int = None,
         params: Dict[str, Any] = None) -> EvalScore:
    params = dict() if params is None else params
    params = {
        **params,
        'nthread': cpu_count() if n_thread is None else n_thread,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    booster = xgb.train(params=params,
                        dtrain=train,
                        num_boost_round=1000,
                        early_stopping_rounds=100,
                        evals=[(test, 'test')],
                        verbose_eval=False
                        )
    y_pred_train = booster.predict(data=train, ntree_limit=booster.best_ntree_limit)
    y_pred_test = booster.predict(data=test, ntree_limit=booster.best_ntree_limit)

    return EvalScore(
        train_true=y_train,
        train_pred=y_pred_train,
        test_true=y_test,
        test_pred=y_pred_test,
        train_set=train,
        test_set=test,
        train_df=X_train,
        test_df=X_test,
        booster=booster
    )


def _gbm(X_train: Union[pd.DataFrame, np.ndarray],
         y_train: Union[pd.Series, np.ndarray],
         X_test: Union[pd.DataFrame, np.ndarray],
         y_test: Union[pd.Series, np.ndarray],
         category_names: Optional[List[str]] = None,
         n_thread: int = None,
         params: Dict[str, Any] = None) -> EvalScore:
    params = dict() if params is None else params
    params = {
        **params,
        'objective': 'binary',
        'tree_learner': 'feature',
        'nthread': cpu_count() if n_thread is None else n_thread,
        'boosting': 'gbdt',
        'boost_from_average': True,
        'metric': 'auc',
        'verbosity': -1
    }
    train = gbm.Dataset(
        data=X_train,
        label=y_train,
        categorical_feature='auto' if category_names is None else category_names
    )
    test = gbm.Dataset(
        data=X_test,
        label=y_test,
        categorical_feature='auto' if category_names is None else category_names
    )
    booster = gbm.train(params=params,
                        train_set=train,
                        num_boost_round=1000,
                        categorical_feature=category_names,
                        early_stopping_rounds=100,
                        valid_sets=[test],
                        valid_names=['test'],
                        verbose_eval=False)
    y_pred_train = booster.predict(X_train, num_iteration=booster.best_iteration)
    y_pred_test = booster.predict(X_test, num_iteration=booster.best_iteration)

    return EvalScore(
        train_true=y_train,
        train_pred=y_pred_train,
        test_true=y_test,
        test_pred=y_pred_test,
        train_set=train,
        test_set=test,
        train_df=X_train,
        test_df=X_test,
        booster=booster
    )


def _cab(X_train: Union[pd.DataFrame, np.ndarray],
         y_train: Union[pd.Series, np.ndarray],
         X_test: Union[pd.DataFrame, np.ndarray],
         y_test: Union[pd.Series, np.ndarray],
         n_thread: int = None,
         category_names: Optional[List[str]] = None,
         params: Dict[str, Any] = None) -> EvalScore:
    params = dict() if params is None else params
    params = {
        **params,
        'objective': 'Logloss',
        'thread_count': cpu_count() if n_thread is None else n_thread,
        'eval_metric': 'AUC',
        'use_best_model': True,
        'num_boost_round': 1000,
        'verbose': False,
    }
    train = cab.Pool(
        data=X_train,
        label=y_train,
        cat_features=category_names,
    )
    test = cab.Pool(
        data=X_test,
        label=y_test,
        cat_features=category_names,
    )
    booster = cab.CatBoostClassifier(**params)
    booster.fit(X=train,
                eval_set=test,
                early_stopping_rounds=100,
                verbose_eval=False)
    y_pred_train = booster.predict_proba(train)[:, 1]
    y_pred_test = booster.predict_proba(test)[:, 1]

    return EvalScore(
        train_true=y_train,
        train_pred=y_pred_train,
        test_true=y_test,
        test_pred=y_pred_test,
        train_set=train,
        test_set=test,
        train_df=X_train,
        test_df=X_test,
        booster=booster
    )

