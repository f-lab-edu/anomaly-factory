"""프로젝트의 공통 변수를 설정합니다."""
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_DIRECTORY = Path(ROOT, "model")

# 디렉토리 구성
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

# 변수 설정
path = {"sklearn_model": Path(MODEL_DIRECTORY, "iris_classification_model.joblib")}
