"""통계 유틸 — seed 간 95% 신뢰구간과 알고리즘 간 차이 비교.

scipy 없이 t-분포 임계값 직접 내장 (seed 수 적은 실험 특화).
"""
from __future__ import annotations

import numpy as np

# 양측 95% t 임계값 (자유도 df)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 18: 2.101, 20: 2.086,
        25: 2.060, 30: 2.042}


def t95(df):
    # 자유도 → 양측 95% t 임계값, df>30은 1.96으로 근사
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    if df > 30:
        return 1.96
    keys = sorted(_T95)
    lo = max(k for k in keys if k <= df)
    return _T95[lo]


def mean_ci(values, conf=0.95):
    """평균과 95% 신뢰구간. 반환: dict(mean,std,sem,ci_lo,ci_hi,half,n)."""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    if n < 2:
        return {"mean": mean, "std": 0.0, "sem": 0.0,
                "ci_lo": mean, "ci_hi": mean, "half": 0.0, "n": n}
    std = float(x.std(ddof=1))
    sem = std / np.sqrt(n)
    half = t95(n - 1) * sem
    return {"mean": mean, "std": std, "sem": sem,
            "ci_lo": mean - half, "ci_hi": mean + half, "half": half, "n": n}


def compare(a_values, b_values):
    """A - B 차이의 95% CI (Welch's t-test) ― CI가 0을 벗어나면 유의미."""
    a = np.asarray(a_values, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)
    na, nb = len(a), len(b)
    diff = float(a.mean() - b.mean())
    if na < 2 or nb < 2:
        return {"diff": diff, "ci_lo": diff, "ci_hi": diff,
                "significant": False, "note": "n<2"}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return {"diff": diff, "ci_lo": diff, "ci_hi": diff,
                "significant": diff != 0, "note": "zero variance"}
    # Welch-Satterthwaite 자유도
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    half = t95(int(round(df))) * se
    lo, hi = float(diff - half), float(diff + half)
    return {"diff": diff, "ci_lo": lo, "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0), "df": float(df)}


def curve_mean_ci(returns_by_seed):
    """seed별 학습 곡선 → 짧은 쪽 길이로 맞추고 step별 mean/95%CI 반환."""
    L = min(len(r) for r in returns_by_seed)
    arr = np.array([r[:L] for r in returns_by_seed], dtype=np.float64)  # (seeds, L)
    n = arr.shape[0]
    mean = arr.mean(0)
    if n < 2:
        return mean, np.zeros(L), np.zeros(L)
    sem = arr.std(0, ddof=1) / np.sqrt(n)
    half = t95(n - 1) * sem
    return mean, mean - half, mean + half
