# RL 기반 서버리스 오토스케일링

Azure Functions 2019 실제 워크로드를 이용해 서버리스 오토스케일링 문제를 강화학습으로 해결하는 프로젝트이다.

서버리스 환경에서는 warm 인스턴스를 많이 유지하면 콜드스타트는 줄어들지만 비용이 증가하고, 반대로 인스턴스를 적게 유지하면 비용은 절약되지만 콜드스타트가 늘어난다.

본 프로젝트는 이 문제를 MDP(Markov Decision Process)로 정의하고, DQN과 PPO를 직접 구현하여 기존 오토스케일링 방식(HPA, Seasonal, Fixed, Oracle)과 비교한다.

---

## 프로젝트 개요

### 목표

매 분 유지할 warm 인스턴스 수를 결정하여 다음 두 목표를 동시에 달성한다.

* 콜드스타트 최소화
* 자원 비용 최소화

### 환경

* 데이터: Azure Functions Dataset 2019
* 상태(State): 과거 부하, 현재 인스턴스 수, 이용률, 시간 정보
* 행동(Action): warm 인스턴스 수 조정
* 보상(Reward): 비용 + 콜드스타트 패널티

---

## 핵심 결과

실제 Azure Functions 트레이스 기반 실험 결과는 다음과 같다.

| 시나리오 | 결과 |
| --- | --- |
| steady | DQN이 HPA보다 약 24% 낮은 비용으로 유사한 수준의 안정성 달성 |
| spiky | 알고리즘보다 부하 예측 가능성이 성능을 결정 |
| bursty | HPA와 Seasonal이 이미 강력한 성능을 보임 |

### 주요 관찰

* DQN은 모든 시나리오에서 PPO보다 안정적으로 학습되었다.
* 강화학습의 효과는 부하의 예측 가능성에 크게 의존했다.
* reward 가중치는 비용–콜드스타트 균형점을 결정했다.
* spiky 시나리오에서는 알고리즘보다 state 정보의 한계가 더 큰 문제였다.

자세한 결과 및 해석은 `docs/ANALYSIS.md` 참고.

---

## 실행 방법

### 1. 설치

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
python -m src.data_prep
```

실행 후 다음 파일이 생성된다.

```text
data/processed/
├── func_steady.npz
├── func_spiky.npz
└── func_bursty.npz
```

### 3. 학습

DQN 학습 예시:

```bash
python -m src.train \
  --algo dqn \
  --data data/processed/func_steady.npz \
  --steps 60000
```

### 4. 전체 실험 실행

```bash
bash run_experiments.sh
```

### 5. 결과 시각화

```bash
python -m src.make_plots --indir results/steady
```

---

## 프로젝트 구조

```text
rl-autoscaling/
├── docs/
│   ├── DESIGN.md
│   ├── ANALYSIS.md
│   ├── AUDIT.md
│   └── REFERENCES.md
├── src/
├── data/
└── results/
```

---

## 문서 안내

| 문서 | 내용 |
| --- | --- |
| DESIGN.md | 문제 정의 및 환경 설계 |
| ANALYSIS.md | 실험 결과 및 해석 |
| AUDIT.md | 요구사항 충족 여부 및 한계 점검 |
| REFERENCES.md | 데이터·논문 출처 및 참고자료 |

---

## 재현성

* 실험은 고정된 random seed를 사용한다.
* 데이터는 `python -m src.data_prep`으로 재생성 가능하다.
* 주요 실험 결과는 `results/` 디렉터리에 저장된다.

---

## 참고

* DQN, PPO는 PyTorch로 직접 구현한다.
* Stable-Baselines3 등 강화학습 라이브러리는 사용하지 않았다.
* 데이터 출처 및 참고문헌은 `docs/REFERENCES.md`에 정리한다.
* AI 도구 사용 내역 역시 `docs/REFERENCES.md`에 명시한다.
