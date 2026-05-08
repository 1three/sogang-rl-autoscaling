"""
서버리스 오토스케일링 환경 (Gymnasium)

매 분 warm 인스턴스 수를 결정하는 MDP
목표: 비용(유지 비용)과 콜드스타트(대기 지연)를 동시에 줄이기
"""
from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

EPS = 1e-6


def required_concurrency(lam, duration_ms):
    """Little's law: C = λ × d / 60000 ― λ는 분당 호출수, d는 평균 실행시간(ms)"""
    return np.asarray(lam, dtype=np.float64) * duration_ms / 60000.0


def step_metrics(lam, n, duration_ms):
    """콜드스타트·유휴·가동률 계산 ― lam, n은 스칼라 또는 배열"""
    C = required_concurrency(lam, duration_ms)
    n = np.asarray(n, dtype=np.float64)
    coldstart = np.maximum(0.0, C - n)           # 부족 동시성 → 콜드스타트
    idle = np.maximum(0.0, n - C)                # 유휴 warm 인스턴스
    util = np.where(n > 0, np.minimum(C / np.maximum(n, EPS), 1.0),
                    np.where(C > 0, 1.0, 0.0))   # n=0 & 수요>0 이면 포화(1.0)
    return {"C": C, "coldstart": coldstart, "idle": idle, "util": util}


def make_synthetic_trace(kind="steady", days=14, seed=0):
    """
    실제 데이터 없을 때 쓰는 합성 트레이스

    kind: 'steady'(규칙적) | 'spiky'(간헐 급증) | 'bursty'(상시 고부하)
    """
    # peak 동시성이 수십 수준이어야 autoscaling 문제가 의미 있음
    rng = np.random.default_rng(seed)
    T = days * 1440
    minute = np.arange(T) % 1440
    daily = 0.5 * (1 + np.sin(2 * np.pi * (minute - 360) / 1440))  # 정오 피크
    if kind == "steady":                       # 뚜렷한 일중 주기, peak C ~ 35
        base = 200 * daily + 100
        inv = rng.poisson(np.maximum(base, 1)).astype(np.float64)
        duration = 800.0
    elif kind == "spiky":                      # 평소 낮고 드물게 급증, spike C ~ 60
        base = 30 * daily + 5
        inv = rng.poisson(np.maximum(base, 0.5)).astype(np.float64)
        spikes = rng.random(T) < 0.01                              # 1% 확률 급증
        inv[spikes] += rng.integers(800, 2000, spikes.sum())
        duration = 2000.0
    elif kind == "bursty":                     # 상시 고부하 + 노이즈, peak C ~ 45
        base = 4000 * daily + 1500
        noise = 1 + 0.4 * rng.standard_normal(T)
        inv = np.maximum(rng.poisson(np.maximum(base * noise, 1)), 0).astype(np.float64)
        duration = 500.0
    else:
        raise ValueError(f"unknown kind: {kind}")
    return inv, duration


class ServerlessScalingEnv(gym.Env):
    """warm 인스턴스 수를 조절해 비용과 콜드스타트를 최소화하는 환경"""

    metadata = {"render_modes": []}

    def __init__(self, invocations, duration_ms, *, k=10, n_max=64,
                 w_cost=1.0, w_cold=5.0, w_idle=0.0, action_mode="discrete",
                 episode_len=1440, random_start=True):
        super().__init__()
        self.inv = np.asarray(invocations, dtype=np.float64)
        self.T = len(self.inv)
        self.duration = float(duration_ms)
        self.k = int(k)
        self.n_max = int(n_max)
        self.w_cost, self.w_cold, self.w_idle = w_cost, w_cold, w_idle
        self.action_mode = action_mode
        self.episode_len = min(int(episode_len), self.T)
        self.random_start = random_start

        # 보상 계산 시 [0,1] 정규화에 사용
        self.lam_max = max(self.inv.max(), 1.0)
        self.C_max = max(required_concurrency(self.lam_max, self.duration), 1.0)

        # 관측 벡터: 과거 k분 부하 + 현재 상태(3) + 시간 인코딩(4)
        self.obs_dim = self.k + 7
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        if action_mode == "discrete":
            self.deltas = np.array([-4, -2, -1, 0, 1, 2, 4])
            self.action_space = spaces.Discrete(len(self.deltas))
        elif action_mode == "continuous":
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,),
                                           dtype=np.float32)
        else:
            raise ValueError(f"unknown action_mode: {action_mode}")

        self._recent_cold = deque(maxlen=10)

    def _action_to_n(self, action):
        if self.action_mode == "discrete":
            delta = int(self.deltas[int(action)])
            return int(np.clip(self.n + delta, 0, self.n_max))
        a = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        return int(round(a * self.n_max))

    def _get_obs(self):
        # 에피소드 시작 전 구간은 0으로 채움
        lo = self.t - self.k
        window = np.zeros(self.k, dtype=np.float64)
        valid = self.inv[max(lo, 0):self.t]
        window[self.k - len(valid):] = valid
        window = window / self.lam_max

        cold_rate = float(np.mean(self._recent_cold)) if self._recent_cold else 0.0
        minute = self.t % 1440
        dow = (self.t // 1440) % 7
        time_feats = [
            0.5 * (1 + np.sin(2 * np.pi * minute / 1440)),
            0.5 * (1 + np.cos(2 * np.pi * minute / 1440)),
            0.5 * (1 + np.sin(2 * np.pi * dow / 7)),
            0.5 * (1 + np.cos(2 * np.pi * dow / 7)),
        ]
        obs = np.concatenate([
            window,
            [self.n / self.n_max, self.last_util, cold_rate],
            time_feats,
        ]).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(0, self.T - self.episode_len)
        if options and "start" in options:
            self.start = int(options["start"])
        elif self.random_start and max_start > 0:
            self.start = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start = 0
        self.t = self.start
        self.steps_done = 0
        self.n = 0
        self.last_util = 0.0
        self._recent_cold.clear()
        return self._get_obs(), {}

    def step(self, action):
        n_new = self._action_to_n(action)
        lam = self.inv[self.t]
        m = step_metrics(lam, n_new, self.duration)
        C, cold, idle, util = (float(m["C"]), float(m["coldstart"]),
                               float(m["idle"]), float(m["util"]))

        reward = -(self.w_cost * (n_new / self.n_max)
                   + self.w_cold * (cold / self.C_max)
                   + self.w_idle * (idle / self.n_max))

        self.last_util = util
        self._recent_cold.append(cold / max(C, EPS))
        info = {"lam": lam, "n": n_new, "C": C, "coldstart": cold,
                "idle": idle, "util": util}

        self.n = n_new
        self.t += 1
        self.steps_done += 1
        terminated = self.steps_done >= self.episode_len or self.t >= self.T
        return self._get_obs(), reward, terminated, False, info
