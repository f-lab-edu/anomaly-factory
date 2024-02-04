"""FastAPI을 활용하여 scikit-learn 모델을 배포합니다."""

# %%
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

import config

model = load(config.path["sklearn_model"])


def get_prediction(sepal_length, sepal_width, petal_length, petal_width) -> dict:
    """FastAPI로 입력받은 인자에 대한 예측 결과를 반환합니다.

    Args:
        sepal_length (float): 꽃받침 길이입니다.
        sepal_width (float): 꽃받침 너비입니다.
        petal_length (float): 꽃잎 길이입니다.
        petal_width (float): 꽃잎 너비입니다.

    Returns:
        dict: 예측 클래스 및 확률을 포함하는 딕셔너리입니다.

    """
    x = [[sepal_length, sepal_width, petal_length, petal_width]]

    y = model.predict(x)
    prob = model.predict_proba(x)[0].tolist()

    return {"prediction": int(y), "probability": prob}


app = FastAPI()


class IrisModel(BaseModel):
    """FastAPI에 사용하는, iris data의 타입을 명시한 pydantic 클래스입니다."""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# uvicorn main:app --reload
# http://127.0.0.1:8000/docs
@app.post("/predict")
async def predict(data: IrisModel) -> dict:
    """사용자의 iris data 입력에 대한 예측 결과를 반환합니다.

    Args:
        data (IrisModel): iris data의 타입을 명시한 pydantic 클래스입니다.

    Returns:
        dict: 예측 클래스 및 확률을 포함하는 딕셔너리입니다.

    """
    return get_prediction(data.sepal_length, data.sepal_width, data.petal_length, data.petal_width)
