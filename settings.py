"""프로젝트에서 사용되는 변수의 기본값을 관리합니다."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """프로젝트에서 활용하는 변수입니다."""

    tracking_uri: str = "http://localhost:5000"
    plot_name: str = "feature_importances.png"
    artifact_path: str = "model"
    model_format: str = "ubj"
    n_trials: int = 100
    train_size: float = 0.8
