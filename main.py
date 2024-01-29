"""
FastAPI을 활용하여 scikit-learn 모델을 배포합니다. 
"""

# %%
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

model = load('iris_classification_model.joblib')


def get_prediction(param1, param2, param3, param4):
    x = [[param1, param2, param3, param4]]

    y = model.predict(x)
    prob = model.predict_proba(x)[0].tolist()

    return {'prediction': int(y), 'probability': prob}


app = FastAPI()


class IrisModel(BaseModel):
    param1: float
    param2: float
    param3: float
    param4: float


# uvicorn main:app --reload
@app.post('/predict')
async def predict(params: IrisModel):
    return get_prediction(params.param1, params.param2, params.param3, params.param4)