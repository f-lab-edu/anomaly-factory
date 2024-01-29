# Anomaly-Factory : Production-Ready End-to-End MLOps Project
![using_mlops_tools](assets/using_mlops_tools.png)  

* 이상 탐지 모델의 데이터 수집부터 모델 배포에 이르는 라이프사이클을 구축하고 관리합니다.
* 사용자는 준비된 데이터를 활용해 모델 학습과 비교를 자동화할 수 있습니다.

## 목표 및 진행상황
* CICD
  * [x] (1주차) 작성한 코드의 품질을 검증합니다.(pre-commit)
* Feature Store
  * [ ] 학습에 활용하기 위해, 원시 데이터에서 유의미한 Feature를 추출합니다.
  * [ ] (Feast) 추출한 Feature를 학습에 재현 가능하도록 특성 저장소(Feature Store)를 구성합니다.
* Versioning 
  * [ ] (DVC) 학습에 사용된 데이터의 버전을 관리합니다.
  * [ ] (MLflow) 선정된 최적 모델을 재사용할 수 있도록 Registry를 구성합니다.
* Experiment Tracking
  * [x] (Sklearn) API에 배포할 모델을 학습합니다.
  * [ ] (MLflow) 하이퍼파라미터 튜닝을 통한 최적 모델을 학습을 자동화합니다.
* Serving
  * [x] (1주차) (FastAPI) 모델을 API로 배포합니다.
  * [ ] (BentoML) 모델을 로컬에서 API로 배포합니다.
  * [ ] (BentoML) 모델을 네이버 클라우드 서비스를 활용한 API로 배포합니다.
* Monitoring
  * (EvidentlyAI) 신규 데이터에 대한 모델의 예측 성능을 모니터링합니다.
## 프로젝트 아키텍처 (사용하는 DB 등 외부 컴포넌트 , 외부 API 연동 등)
* DB : Google Drive
  * Naver Cloud Object Storage는 DVC에서 지원하지 않으므로 제외합니다.
## Requirements
```toml
python = "^3.10"
pre-commit = "^3.6.0"
```
## 설치 및 실행 방법
* API 서버를 실행하여 사용자가 배열로 이루어진 쿼리를 입력, 결과값을 받을 수 있게 구성합니다.
