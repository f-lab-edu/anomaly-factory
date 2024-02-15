"""모델 튜닝에 필요한 파라미터와 메트릭을 구성합니다."""

from abc import ABCMeta, abstractmethod

import mlflow
import optuna
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error


class OptimizingTracker(metaclass=ABCMeta):
    """최적화 모델에 필요한 항목을 구성한 추상 클래스입니다."""

    @abstractmethod
    def __init__(self):
        """모델 구성에 필요한 변수를 초기화 시 선언합니다."""
        self.model
        self.mlflow_module
        self.metric
        self.metric_name
        self.best_metric_name

    @abstractmethod
    def get_suggest_parameters(self, trial: optuna.trial.Trial) -> dict:
        """Optuna의 study에 포함할 튜닝 대상 파라미터를 설정합니다.

        Args:
            trial (optuna.trial.Trial): objective 함수가 다음 파라미터를 정할 수 있도록 인터페이스를 제공하는 객체입니다.

        Returns:
            dict: 튜닝 대상 파라미터로 구성된 딕셔너리 객체입니다.

        """

    @abstractmethod
    def log(self, params: dict, metric_name: str, value: float) -> None:
        """튜닝 과정에서 메트릭을 기록합니다.

        Args:
            params (dict): mlflow에 기록할 파라미터 조합입니다.
            metric_name (str): mlflow에 기록할 metric의 이름입니다.
            value (float): 계산된 metric 값입니다.

        """


class XGBRegressorOptimizingTracker(OptimizingTracker):
    """XGBoost 회귀 모델을 튜닝하기 위해 필요한 변수를 구성한 클래스입니다."""

    def __init__(self):
        """모델 구성에 필요한 변수를 초기화 시 선언합니다."""
        self.model = xgb.XGBRegressor
        self.mlflow_module = mlflow.xgboost
        self.metric = root_mean_squared_error
        self.metric_name = "rmse"
        self.best_metric_name = "best_rmse"

    def get_suggest_parameters(self, trial: optuna.trial.Trial) -> dict:
        """Optuna의 study에 포함할 튜닝 대상 파라미터를 설정합니다.

        Args:
            trial (optuna.trial.Trial): objective 함수가 다음 파라미터를 정할 수 있도록 인터페이스를 제공하는 객체입니다.

        Returns:
            dict: 튜닝 대상 파라미터로 구성된 딕셔너리 객체입니다.

        """
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
        }
        return params

    def log(self, params: dict, metric_name: str, value: float) -> None:
        """튜닝 과정에서 메트릭을 기록합니다.

        Args:
            params (dict): mlflow에 기록할 파라미터 조합입니다.
            metric_name (str): mlflow에 기록할 metric의 이름입니다.
            value (float): 계산된 metric 값입니다.

        """
        mlflow.log_params(params)
        mlflow.log_metric(metric_name, value)
