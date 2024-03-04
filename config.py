"""프로젝트에서 사용되는 변수의 기본값을 관리합니다."""
mlflow = {
    "tracking_uri": "http://localhost:5000",
    "plot_name": "feature_importances.png",
    "artifact_path": "model",
    "model_format": "ubj",
    "n_trials": 100,
    "train_size": 0.8,
}
