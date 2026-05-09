"""
비교용 baseline 정책 4종
invocation 시계열 → 분 단위 n 스케줄 반환

fixed          : n 고정 (가장 단순, 하한 기준)
hpa            : 직전 이용률 기반 반응형 조정 (실무 표준)
oracle         : 미래 수요를 알고 n = ceil(C) 설정 (이론적 최적)
seasonal_naive : 전날 같은 시간대 수요로 선제 프로비저닝
"""
from __future__ import annotations

import numpy as np

from .env import required_concurrency


def fixed_schedule(invocations, duration_ms, n_const, n_max=64):
    T = len(invocations)
    return np.clip(np.full(T, n_const), 0, n_max).astype(int)


def hpa_schedule(invocations, duration_ms, *, target_util=0.7, n_max=64, n0=1, scale_down_factor=0.0):
    """직전 분 이용률로 다음 분 n 결정 ― Kubernetes HPA 공식, 1분 지연 포함"""
    T = len(invocations)
    C = required_concurrency(invocations, duration_ms)
    n = np.zeros(T, dtype=int)
    cur = max(int(n0), 1)
    for t in range(T):
        n[t] = cur
        util = C[t] / max(cur, 1e-6)
        desired = int(np.ceil(cur * util / target_util))
        if util < scale_down_factor * target_util:   # scale_down_factor=0이면 항상 건너뜀
            desired = cur
        cur = int(np.clip(desired, 1, n_max))
    return n


def oracle_schedule(invocations, duration_ms, n_max=64):
    """미래 수요를 알고 n = ceil(C)로 설정 ― 콜드스타트 0의 이론적 최적"""
    C = required_concurrency(invocations, duration_ms)
    return np.clip(np.ceil(C).astype(int), 0, n_max)


def seasonal_naive_schedule(invocations, duration_ms, *, period=1440, n_max=64):
    """전날 같은 시간대 수요로 선제 프로비저닝 ― 패턴이 반복된다는 가정"""
    C = required_concurrency(invocations, duration_ms)
    n = np.ceil(C).astype(int)
    shifted = np.empty_like(n)
    shifted[:period] = n[:period]            # 첫날은 참조할 어제가 없으므로 당일 값 사용
    shifted[period:] = n[:-period]
    return np.clip(shifted, 0, n_max)


BASELINES = {
    "fixed": lambda inv, d, **kw: fixed_schedule(inv, d, kw.get("n_const", 8), kw.get("n_max", 64)),
    "hpa": lambda inv, d, **kw: hpa_schedule(inv, d, **kw),
    "oracle": lambda inv, d, **kw: oracle_schedule(inv, d, kw.get("n_max", 64)),
    "seasonal": lambda inv, d, **kw: seasonal_naive_schedule(inv, d, **kw),
}
