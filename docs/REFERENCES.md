# 참고문헌 (References)

본 프로젝트는 Azure Functions 실제 워크로드 데이터를 이용해 서버리스 오토스케일링 문제를 강화학습으로 모델링했다. 주요 참고 자료는 아래와 같다.

## 데이터셋

### Azure Functions Dataset 2019

* Microsoft Azure Public Dataset
* https://github.com/Azure/AzurePublicDataset

사용 목적:

* 실제 함수 호출 트레이스 수집
* 학습 및 평가용 워크로드 생성
* steady, spiky, bursty 시나리오 구성

### Shahrad et al. (2020)

> Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider

사용 목적:

* 데이터셋 출처 확인
* 서버리스 워크로드 특성 이해
* 콜드스타트와 비용 문제 정의 참고

---

## 강화학습

### DQN

Mnih et al. (2015)

> Human-level control through deep reinforcement learning

사용 목적:

* Q-learning 기반 에이전트 구현

### PPO

Schulman et al. (2017)

> Proximal Policy Optimization Algorithms

사용 목적:

* 정책 기반 에이전트 구현

### 추가 기법

* Double DQN
* Dueling Network Architecture
* Generalized Advantage Estimation (GAE)

위 논문의 아이디어를 참고하여 구현했다.

---

## 환경 모델링

### Little's Law

> L = λW

사용 목적:

* 호출량과 실행시간으로 필요 동시성 계산

---

## Baseline

### Kubernetes HPA

https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/

사용 목적:

* 반응형 오토스케일링 기준선(Baseline)

### Seasonal Naive Forecast

사용 목적:

* 단순 예측 기반 기준선(Baseline)

---

## 사용 라이브러리

* PyTorch
* NumPy
* pandas
* Matplotlib
* Gymnasium

세부 버전은 `requirements.txt`에 명시했다.

---

## AI 도구 사용

과제 지침에 따라 AI 코딩 도구의 도움을 받아 설계 및 구현을 진행했습니다.
다만 모든 코드와 실험 결과는 직접 검토 및 수정했으며, Stable-Baselines3 등의 강화학습 라이브러리 코드를 복사하지 않고 DQN과 PPO를 직접 구현했다.
