"""프로젝트를 GUI로 제공하기 위한 streamlit 기반 UI 코드입니다."""

# %%
import json
import time
from pathlib import Path
from threading import Thread

import streamlit as st
from sklearn.datasets import load_diabetes, load_iris

import config
from experiment_tracking.datamodule import DataModule
from experiment_tracking.optimizing_tracker import XGBRegressorOptimizingTracker
from experiment_tracking.tuning_experiment import TuningExperiment

not_initailized = "stage" not in st.session_state
if not_initailized:
    st.session_state.stage = 0
    st.session_state.data = "iris"
    st.session_state.df = ""
    st.session_state.target = "petal width"
    st.session_state.model = "xgboost.XGBRegressor"
    st.session_state.tuning_done = False


def select_data():
    """1단계 : 사용자가 파일을 업로드합니다."""
    st.title("1 Select data to train")
    st.session_state.data = st.selectbox("Data", ("sklearn.datasets.load_iris", "sklearn.datasets.load_diabetes"))


def select_target():
    """2단계 : 업로드한 파일을 읽고, 예측할 목표 열을 입력받습니다."""
    if st.session_state.data == "sklearn.datasets.load_iris":
        st.session_state.df, y = load_iris(as_frame=True, return_X_y=True)
        st.session_state.df["target"] = y
    elif st.session_state.data == "sklearn.datasets.load_diabetes":
        st.session_state.df, y = load_diabetes(as_frame=True, return_X_y=True)
        st.session_state.df["target"] = y
    st.title("2 Select target from column names")
    st.dataframe(st.session_state.df.head())
    st.session_state.target = st.selectbox("Target", st.session_state.df.columns.tolist())


def select_model():
    """3단계 : 최적화할 모델을 입력받습니다."""
    st.title("3 Select model to train")
    st.session_state.model = st.selectbox("Model", ("xgboost.XGBRegressor",))


def start_experiment():
    """4단계 : MLflow를 호출하여 모델 최적화를 수행합니다."""
    st.title("4 MLflow experiment tracking")
    dm = DataModule(
        st.session_state.df.drop(columns=st.session_state.target), st.session_state.df[st.session_state.target]
    )
    tuner = TuningExperiment("model_run", XGBRegressorOptimizingTracker(), dm)
    th = Thread(target=tuner.tune_model)
    th.start()


def move(step: int):
    """이전, 다음 버튼 클릭 시 페이지 번호를 업데이트합니다.

    Args:
        step (int): 움직일 단계입니다. 1 입력시 다음 페이지, -1 입력 시 이전 페이지로 이동합니다. 1단계는 0입니다.

    """
    if 0 <= st.session_state.stage + step < 5:
        st.session_state.stage += step


def move_component():
    """이전, 다음 버튼을 렌더링합니다."""
    disable_previous = True if st.session_state.stage == 0 else False
    disable_next = True if st.session_state.stage == 4 else False
    col1, spacing, col2 = st.columns([2, 3, 2])
    col1.button("Previous", on_click=move, args=[-1], disabled=disable_previous, use_container_width=True)
    col2.button("Next", on_click=move, args=[1], disabled=disable_next, use_container_width=True)

    align_button_to_undermost = """
    <style>
        .stButton {{
        position: fixed;
        bottom: 3rem;
        }}
    </style>
    """
    st.markdown(align_button_to_undermost, unsafe_allow_html=True)


# AI 컴포넌트
if st.session_state.stage == 0:
    select_data()
elif st.session_state.stage == 1:
    select_target()
elif st.session_state.stage == 2:
    select_model()
elif st.session_state.stage == 3:
    start_experiment()

    placeholder = st.empty()

    while True:
        time.sleep(0.2)
        if Path("output.json").exists():
            with open("output.json", "r") as f:
                data = json.load(f)
                data.pop("distributions", None)
            placeholder.write(data)
            if data["number"] == config.mlflow["n_trials"] - 1:
                break

move_component()
