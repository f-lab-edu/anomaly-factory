"""FastAPI을 활용하여 scikit-learn 모델을 배포합니다."""

# %%
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

import config

model = load(config.path["sklearn_model"])


class IrisModel(BaseModel):
    """FastAPI에 사용하는, iris data의 타입을 명시한 pydantic 클래스입니다."""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def get_prediction(iris_model: IrisModel) -> dict:
    """FastAPI로 입력받은 인자에 대한 예측 결과를 반환합니다.

    Args:
        iris_model (IrisModel): 꽃받침 데이터 형식을 검증하는 pydantic 모델입니다.

    Returns:
        dict: 예측 클래스 및 확률을 포함하는 딕셔너리입니다.

    """
    x = [list(dict(iris_model).values())]
    y = model.predict(x)
    prob = model.predict_proba(x)[0].tolist()

    return {"prediction": int(y), "probability": prob}


app = FastAPI()


# uvicorn main:app --reload
# http://127.0.0.1:8000/docs
@app.post("/predict")
async def predict(data: IrisModel) -> dict:
    """사용자의 iris data 입력에 대한 예측 결과를 반환합니다.

    Args:
        data (IrisModel): 꽃받침 데이터 형식을 검증하는 pydantic 모델입니다.

    Returns:
        dict: 예측 클래스 및 확률을 포함하는 딕셔너리입니다.

    """
    return get_prediction(data)
