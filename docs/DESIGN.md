# RL 기반 서버리스 오토스케일링 — 설계 노트

> Azure Functions 2019 트레이스를 환경으로, **매 분 warm 인스턴스 수를 결정**하는 RL 에이전트를 만들어 콜드스타트(응답 지연)와 비용을 함께 줄이는 것이 목표이다.

---

## 1. 문제와 목표

### 1.1 왜 이 문제인가
현업에서 API 트래픽을 다루면서 오토스케일링 설정이 항상 고민이었다. 요청이 몰릴 때를 대비해 인스턴스를 많이 켜두면 평소에 비용이 낭비되고, 적게 켜두면 급증 시 처음 몇 요청이 눈에 띄게 느려진다.
규칙으로 이 균형을 잡으려면 결국 "언제 올지 모르는 수요"를 항상 가정해야 해서 낭비가 생긴다.
강화학습 수업을 들으면서 이 문제를 RL로 풀어낼 수 있지 않을까 생각했다.

서버리스 플랫폼(Azure Functions, AWS Lambda 등)은 요청이 왔을 때 미리 켜둔 컨테이너(인스턴스)가 없으면 **콜드스타트**(컨테이너 새로 띄우는 시간, 수백 ms~수 초)가 발생한다. 미리 켜두면 그 유지 비용이 든다.

> warm 인스턴스 많이 → 콜드스타트↓, 비용↑
> warm 인스턴스 적게 → 비용↓, 콜드스타트↑

이 트레이드오프를 매 분 상황에 맞게 판단하는 걸 RL로 학습시킨다.

### 1.2 목표
실제 Azure Functions 트레이스를 기반으로, 매 분 warm 인스턴스 수를 결정하는 정책을 학습시키고자 한다.
기존 규칙 기반 방법(임계값 오토스케일러 등)보다 나은 비용-콜드스타트 균형을 달성하는 게 목표다.

### 1.3 검증하고 싶은 것
- H1: RL이 같은 비용에서 HPA 방식보다 콜드스타트를 줄일 수 있는가
- H2: RL이 하루 주기 트래픽 패턴을 학습해 선제적으로 스케일할 수 있는가
- H3: reward 가중치를 바꾸면 비용·SLA 균형을 명시적으로 조절할 수 있는가

---

## 2. 데이터

### 2.1 출처
**Azure Public Dataset — Azure Functions Trace 2019**
`https://github.com/Azure/AzurePublicDataset`
(논문: *Serverless in the Wild*, USENIX ATC 2020)

### 2.2 구성
| 파일 | 내용 |
|---|---|
| `invocations_per_function_md.anon.d01~d14.csv` | 함수별 분당 호출 수 (1~1440 컬럼) |
| `function_durations_percentiles.anon.d01~d14.csv` | 함수별 평균 실행시간(ms) |
| `app_memory_percentiles.anon.d01~d12.csv` | 앱별 메모리 사용량 |

함수가 수만 개라 전부 쓰기는 무리가 있다고 한다. 호출 패턴이 다른 대표 함수 몇 개를 골라서 환경으로 쓴다.

### 2.3 전처리 방향
1. 14일치 invocation CSV → 함수별 길이 20160분(14×1440)의 분당 연속 데이터 확보
2. 각 함수의 평균 실행시간 d(ms) 매칭
3. 패턴 다양성 확보를 위해 3종 분류
   - **steady**: 하루 중 시간대별 패턴이 일정 (예측 가능)
   - **spiky**: 드물다가 갑자기 급증 (콜드스타트 핵심)
   - **bursty**: 항상 호출이 많음

---

## 3. MDP 설계

데이터가 분당 호출수로 기록되어 있어서 에이전트도 1분에 한 번 판단하도록 맞췄다. (스텝 단위 = 1분)

### 3.1 동시성 계산 (Little's law)
분 t에 호출이 λ_t번, 평균 실행시간이 d ms면 필요한 동시 처리 수

- `λ_t`는 분 t에 들어온 요청 수(트레이스에서 읽어옴)
- `C_t`는 그 요청을 처리하기 위해 동시에 필요한 인스턴스 수

이 요청들을 1분 안에 다 처리하기 위한 인스턴스 수는 — 요청 수 × 처리시간 ÷ 1분(60,000ms)으로 계산한다.

```
C_t = λ_t × d / 60000
```

인스턴스를 n_t개 켜두고 있으면:
- n_t ≥ C_t : 충분 → 콜드스타트 없음, 이용률 U_t = C_t / n_t
- n_t < C_t : 부족 → 부족분만큼 콜드스타트 발생

```
coldstart_t = max(0, C_t - n_t)
idle_t      = max(0, n_t - C_t)
```

가정 단순화: 분 내 도착 균일 (인스턴스 1개 = 동시성 1)

### 3.2 상태 (State)
에이전트가 매 분 퍈댠 전에 확인하는 정보

```
s_t = [
  최근 k분 호출수 (정규화),    # 트래픽 추세
  현재 warm 인스턴스 수,
  직전 이용률,
  최근 콜드스타트 비율,
  sin/cos(현재 시각),       # 하루 주기성 인코딩
  sin/cos(요일)
]
```
기본 k=10, 미래 λ는 포함하지 않았다 — 실제 운영에서도 미래 수요는 알 수 없다.

### 3.3 행동 (Action)
이산 조정값: `a_t ∈ {-4, -2, -1, 0, +1, +2, +4}`

다음 분 warm 수: `n_{t+1} = clip(n_t + a_t, 0, n_max)`

왜 이산 조정값인가: 절대값이 아닌 상대 변화로 설계하면 현재 상태 기준으로 작은 조정 / 큰 조정을 선택할 수 있어서 더 자연스럽다.

### 3.4 보상 (Reward)
인스턴스를 많이 켜두면 비용 손해, 콜드스타트가 나면 SLA 손해 — 둘 다 페널티로 합산해 음수로 준다.
```
r_t = -(w_cost × n_{t+1} + w_cold × coldstart_{t+1})
```
기본 w_cost=1.0, w_cold=5.0. w_cold를 키울수록 콜드스타트 회피 우선.
이 가중치를 바꿔가며 비용-SLA 트레이드오프를 실험할 예정.

### 3.5 에피소드 및 Transition
1 에피소드 = 1일(1440 스텝), 평가는 test 구간 고정 rollout 예정이다.

Transition은 별도로 설계할 게 없다 — 트레이스를 그대로 재생하는 방식이라, 에이전트가 행동을 정하면 다음 분 λ는 데이터에서 그냥 읽어온다. 덕분에 같은 정책·같은 시드면 결과가 항상 동일하다.

---

## 4. 환경 구현 (Gymnasium)

```python
class ServerlessScalingEnv(gymnasium.Env):
    def __init__(self, invocations, duration_ms, *,
                 k=10, n_max=64, w_cost=1.0, w_cold=5.0,
                 action_mode="discrete", episode_len=1440):
        # observation_space: (k+6,) 연속 벡터
        # action_space: Discrete(7) — {-4,-2,-1,0,+1,+2,+4}

    def reset(self, seed=None, options=None): ...

    def step(self, action):
        # 1) action → n_next
        # 2) 다음 분 λ로 C 계산
        # 3) coldstart / idle / util 계산
        # 4) reward 계산
        # 5) obs 갱신
        # info에 cost/coldstart/util 기록 → 평가에서 집계
```

트레이스 재생 방식이라 같은 정책·같은 시드면 결과가 동일하여, 재현성 확보에 유리하다.

---

## 5. 알고리즘 계획

| 구분 | 방법 | 비고 |
|---|---|---|
| RL | **DQN** (Double + Dueling) | 이산 액션. PyTorch 직접 구현 |
| RL | **PPO** (clip + GAE) | 비교용. 직접 구현 |
| Baseline | 고정 over-provision | 성능 하한 |
| Baseline | **HPA 방식** (이용률 임계값) | 실무에서 가장 많이 쓰는 방식. 핵심 비교 대상 |
| Baseline | Seasonal-naive (지난 시간대 평균) | 예측형 |
| 상한 | **Oracle** (미래 λ를 알고 최소 n 결정) | 이론적 최선 |

외부 RL 라이브러리 없이 PyTorch로 직접 구현.

---

## 6. 실험 계획

### 평가 지표
- 총 비용 (Σ n_t), 콜드스타트율 (Σcoldstart / Σλ)
- 평균 이용률
- w_cold 스윕 시 비용-콜드스타트 트레이드오프 그래프

### 실험 구성
1. **알고리즘 비교**: DQN vs PPO vs HPA vs Oracle
2. **w_cold 스윕**: `{1, 3, 10, 30}` → 비용-SLA 트레이드오프 시각화
3. **시나리오 비교**: steady / spiky / bursty
4. **통계적 신뢰성**: 각 설정 5 seeds, 95% 신뢰구간

---

## 7. 구조 및 일정

```
rl-autoscaling/
├── docs/DESIGN.md
├── data/
│   ├── raw/          # Azure CSV 원본
│   └── processed/    # 함수별 시계열 .npz
├── src/
│   ├── data_prep.py
│   ├── env.py
│   ├── baselines.py
│   ├── agents/
│   │   ├── dqn.py
│   │   └── ppo.py
│   ├── train.py
│   └── evaluate.py
└── results/
```

### 일정
| # | 작업 |
|---|---|
| 1 | 데이터 전처리, 함수 선별 |
| 2 | env.py + baselines 구현, 동작 확인 |
| 3 | DQN 학습, baseline 대비 검증 |
| 4 | PPO 추가, 가중치 스윕, 다중 시드 |
| 5 | 결과 시각화, README, 보고서 마무리 |
