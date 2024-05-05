"""Mlflow 환경에서 Tuning experiment를 통해 최적의 하이퍼파라미터를 찾습니다.

다음 코드로 사용할 수 있습니다.
# 명령 프롬프트에 mlflow ui를 입력해, 서버가 가동중이어야 합니다.

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(as_frame=True, return_X_y=True)
dm = DataModule(X, y)
tuner = TuningExperiment("model_run", XGBRegressorOptimizingTracker(), dm)
tuner.tune_model()
"""

# %%
import json

import mlflow
import optuna
import redis

from experiment_tracking.datamodule import DataModule
from experiment_tracking.optimizing_tracker import OptimizingTracker
from settings import settings

optuna.logging.set_verbosity(optuna.logging.ERROR)
redis_client = redis.Redis(decode_responses=True)


# %%
class TuningExperiment:
    """Experiment를 수행하여 하이퍼파라미터 튜닝, 검증, 모델 저장을 수행하는 클래스입니다."""

    def __init__(self, experiment_name: str, tracker: OptimizingTracker, dm: DataModule):
        """Mlflow Experiment를 수행하기 위한 초기값을 설정합니다.

        Args:
            experiment_name (str): Experiment 이름입니다.
            tracker (OptimizingTracker): 튜닝할 모델의 후보 파라미터와 계산할 메트릭이 포함된 객체입니다.
            dm (DataModule): 튜닝에 필요한 데이터를 포함하는 객체입니다.

        """
        self.tracker = tracker
        self.dm = dm
        self.experiment_name = experiment_name
        self.experiment_id = self.setup_experiment()

    def setup_experiment(self) -> str:
        """실행 중인 서버에 새로운 experiment를 등록합니다.

        Returns:
            str: experiment 이름에 대응하는 id입니다.

        """
        mlflow.set_tracking_uri(settings.tracking_uri)
        experiment_id = self.get_or_create_experiment(self.experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)
        return experiment_id

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

    def tune_model(self):
        """최적의 하이퍼파라미터를 찾습니다."""
        # 고유한 run_name을 생성합니다.

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=settings.n_trials, callbacks=[self.model_callback])
        model = self.tracker.model(**study.best_params)
        model.fit(self.dm.X_train, self.dm.y_train)
        with mlflow.start_run(experiment_id=self.experiment_id, nested=True):
            self.tracker.log(
                params=study.best_params, metric_name=self.tracker.best_metric_name, value=study.best_value
            )
            self.tracker.mlflow_module.log_model(
                model, artifact_path=settings.artifact_path, input_example=self.dm.X_train.iloc[[0]]
            )

    def objective(self, trial: optuna.trial.Trial) -> None:
        """하이퍼파라미터 최적화 experiment를 수행하며 metric을 기록합니다.

        Args:
            trial (optuna.trial.Trial): objective 함수가 다음 파라미터를 정할 수 있도록 인터페이스를 제공하는 객체입니다.

        """
        with mlflow.start_run(nested=True):
            params = self.tracker.get_suggest_parameters(trial)
            model = self.tracker.model(**params)
            model.fit(self.dm.X_train, self.dm.y_train)
            error = self.tracker.metric(self.dm.y_valid, model.predict(self.dm.X_valid))
            self.tracker.log(params=params, metric_name=self.tracker.metric_name, value=error)

        return error

    def model_callback(self, study: optuna.study.study.Study, frozen_trial: optuna.trial.FrozenTrial):
        """모델의 개선 상황을 출력합니다.

        Args:
            study (optuna.study.study.Study): Trial 객체의 기록을 관리하고 생성하는 관리 객체입니다. 콜백의 인자로, optimize 함수에서 제공됩니다.
            frozen_trial (optuna.trial.FrozenTrial): optuna.trial.Trial 객체의 상태와 결과값입니다. 콜백의 인자로, optimize 함수에서 제공됩니다.

        """
        data = frozen_trial.params
        data["number"] = frozen_trial.number
        data["values"] = frozen_trial.values
        data["distributions"] = {}
        for param_name, distribution in frozen_trial.distributions.items():
            data["distributions"][param_name] = optuna.distributions.distribution_to_json(distribution)
        redis_client.publish(self.experiment_name, json.dumps(data))
        best = study.user_attrs.get("best", None)
        if study.best_value and best != study.best_value:
            study.set_user_attr("best", study.best_value)

            number = frozen_trial.number
            value = frozen_trial.value
            if best:
                improvement_percent = (abs(best - study.best_value) / study.best_value) * 100
                print(f"Metric of Trial {number:03}: {value: .4f}({improvement_percent:.4f}% improvement)")
            else:
                print(f"Metric of Trial {number:03}: {value: .4f}")
