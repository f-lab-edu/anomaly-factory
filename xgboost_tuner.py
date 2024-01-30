"""Mlflow 환경에서 xgboost 기반 experiment tracking을 수행합니다."""
# %%
import datetime

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

optuna.logging.set_verbosity(optuna.logging.ERROR)


# %%
class XgboostTuner:
    """Xgboost 모델 기반의 Experiment를 수행하여 하이퍼파라미터 튜닝, 검증, 모델 저장을 수행하는 클래스입니다."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "XGBoostExperiment",
        n_trials: int = 100,
        split_ratio: float = 0.25,
    ):
        """Mlflow Experiment를 수행하기 위한 초기값을 설정합니다.

        Args:
            tracking_uri (str, optional): Mlflow 서버 주소입니다. 기본값은 "http://localhost:5000"입니다.
            experiment_name (str, optional): Experiment 이름입니다. 기본값은 "XGBoostExperiment"입니다.
            n_trials (int, optional): Optuna에서 수행할 최적화 횟수입니다. 기본값은 100입니다.
            split_ratio (float, optional): 검증용 데이터의 비율입니다. 기본값은 0.25입니다.

        """
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_id = self.get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_id=self.experiment_id)

        self.n_trials = n_trials
        self.split_ratio = split_ratio

    def get_or_create_experiment(self, experiment_name: str) -> str:
        """Mlflow experiment의 ID를 반환합니다. 없다면 새로 생성합니다.

        Args:
            experiment_name (str): Experiment의 이름입니다.

        Returns:
            str: Experiment ID입니다.

        """
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)

    def tune_model(self, X: pd.DataFrame, y: pd.Series):
        """Xgboost 모델 기반의 experiment를 수행합니다.

        Args:
            X (pd.DataFrame): 예측을 위한 특성이 포함된 DataFrame입니다.
            y (pd.Series): 예측하고자 하는 데이터의 배열입니다.

        """
        self.setup_data(X, y)
        self.experiment_xgboost_model()

    def setup_data(self, X: pd.DataFrame, y: pd.Series):
        """훈련용 및 검증용 데이터셋으로 분리합니다.

        Args:
            X (pd.DataFrame): 예측을 위한 특성이 포함된 DataFrame입니다.
            y (pd.Series): 예측하고자 하는 데이터의 배열입니다.

        """
        self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(X, y, test_size=self.split_ratio)

    def experiment_xgboost_model(
        self,
        run_name: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        plot_name: str = "feature_importances.png",
        artifact_path: str = "model",
        model_format: str = "ubj",
    ):
        """준비된 데이터를 기반으로, Xgboost 하이퍼파라미터 튜닝을 진행하고 아티팩트를 저장합니다.

        Args:
            run_name (str, optional): 진행할 단위 Experiment의 이름입니다. 기본값은 현재 시간에 대한 연월일_시분초입니다.
            plot_name (str, optional): 아티팩트의 파일 이름입니다. 기본값은 "feature_importances.png"입니다.
            artifact_path (str, optional): 아티팩트를 저장할 경로입니다. 기본값은 "model"입니다.
            model_format (str, optional): 저장할 Xgboost 모델의 포맷입니다. 기본값은 "ubj"입니다.

        """
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name, nested=True):
            study = optuna.create_study(direction="minimize")
            study.optimize(self.objective, n_trials=self.n_trials, callbacks=[self.model_callback])
            model = xgb.XGBRegressor(**study.best_params)
            model.fit(self.train_x, self.train_y)
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_rmse", study.best_value)
            mlflow.log_figure(figure=self.plot_feature_importance(model), artifact_file=plot_name)
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=artifact_path,
                input_example=self.train_x.iloc[[0]],
                model_format=model_format,
            )

            model_uri = mlflow.get_artifact_uri(artifact_path)

        mlflow.xgboost.load_model(model_uri)

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Optuna를 통해 파라미터 튜닝에 대한 metric을 기록하는 experiment를 수행합니다.

        Args:
            trial (optuna.trial.Trial): objective 함수가 다음 파라미터를 정할 수 있도록 인터페이스를 제공하는 객체입니다.

        Returns:
            float: 현재 모델의 검증 데이터 오차입니다.

        """
        with mlflow.start_run(nested=True):
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
            model = xgb.XGBRegressor(**params)
            model.fit(self.train_x, self.train_y)
            error = mean_squared_error(self.valid_y, model.predict(self.valid_x), squared=False)
            mlflow.log_params(params)
            mlflow.log_metric("rmse", error)

        return error

    def plot_feature_importance(self, model: xgb.XGBRegressor) -> matplotlib.figure.Figure:
        """Xgboost 모델에서 계산된 특성의 중요도를 차트로 반환합니다.

        Args:
            model (xgb.XGBRegressor): Xgboost 모델입니다.

        Returns:
            matplotlib.figure.Figure: matplotlib Figure 객체입니다.

        """
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(model, importance_type="gain", ax=ax, title="Feature Importance")
        plt.tight_layout()
        plt.close(fig)

        return fig

    def model_callback(self, study: optuna.study.study.Study, frozen_trial: optuna.trial.FrozenTrial):
        """모델의 개선 상황을 출력합니다.

        Args:
            study (optuna.study.study.Study): Trial 객체의 기록을 관리하고 생성하는 관리 객체입니다. 콜백의 인자로, optimize 함수에서 제공됩니다.
            frozen_trial (optuna.trial.FrozenTrial): optuna.trial.Trial 객체의 상태와 결과값입니다. 콜백의 인자로, optimize 함수에서 제공됩니다.

        """
        best = study.user_attrs.get("best", None)
        if study.best_value and best != study.best_value:
            study.set_user_attr("best", study.best_value)
            if best:
                improvement_percent = (abs(best - study.best_value) / study.best_value) * 100
                print(
                    f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                    f"{improvement_percent: .4f}% improvement"
                )
            else:
                print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
