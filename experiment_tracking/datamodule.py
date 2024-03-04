"""튜닝에 필요한 데이터를 분리합니다."""

from sklearn.model_selection import train_test_split

import config


class DataModule:
    """Experiment tracking에 사용되는 데이터를 나누는 클래스입니다."""

    def __init__(self, X: list, y: list, train_size: float = config.mlflow["train_size"]):
        """훈련 및 검증 데이터셋으로 분리합니다.

        Args:
            X (list): 변수 예측에 필요할 독립변수 리스트 객체입니다.
            y (list): 예측할 변수의 리스트 객체입니다.
            train_size (float, optional): 훈련 데이터 비율입니다. 기본값은 config.mlflow["train_size"]입니다.

        """
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, train_size=train_size)
