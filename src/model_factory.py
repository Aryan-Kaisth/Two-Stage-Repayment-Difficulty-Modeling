from collections.abc import Callable
from typing import Any
from configs.model_params import histgbm_params, lightgbm_params, xgboost_params, catboost_params

from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

MODEL_FACTORY: dict[str, Callable[[], Any]] = {
    "histgbm": lambda: HistGradientBoostingClassifier(**histgbm_params),
    "lightgbm": lambda: LGBMClassifier(**lightgbm_params),
    "xgboost": lambda: XGBClassifier(**xgboost_params),
    "catboost": lambda: CatBoostClassifier(**catboost_params),
}