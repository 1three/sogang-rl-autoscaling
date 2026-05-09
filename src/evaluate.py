"""
평가 모듈 ― 정책 성능을 수치로 정리하고 그래프로 그림

RL과 baseline을 같은 기준으로 채점해 공정하게 비교
"""
from __future__ import annotations

import numpy as np

from .env import step_metrics


def aggregate_metrics(invocations, n_schedule, duration_ms, *, weights=None):
    """n 스케줄을 받아 비용·콜드스타트율·가동률 등 지표 계산"""
    inv = np.asarray(invocations, dtype=np.float64)
    n = np.asarray(n_schedule, dtype=np.float64)
    m = step_metrics(inv, n, duration_ms)
    C, cold, idle, util = m["C"], m["coldstart"], m["idle"], m["util"]

    total_demand = C.sum()
    out = {
        "total_cost": float(n.sum()),                       # instance-minutes
        "mean_instances": float(n.mean()),
        "total_coldstart": float(cold.sum()),
        "coldstart_rate": float(cold.sum() / max(total_demand, 1e-9)),
        "minutes_with_coldstart_frac": float((cold > 1e-9).mean()),
        "mean_util": float(util[n > 0].mean()) if (n > 0).any() else 0.0,
        "total_idle": float(idle.sum()),
    }
    if weights is not None:
        w_cost, w_cold, w_idle = weights
        C_max = max(C.max(), 1.0)
        n_max = max(n.max(), 1.0)
        out["mean_reward"] = float(-(
            w_cost * (n / n_max) + w_cold * (cold / C_max) + w_idle * (idle / n_max)
        ).mean())
    return out


def rollout_schedule(env, act_fn, start=0):
    """학습된 정책을 환경에 한 번 돌려서 분별 호출수·인스턴스 수 기록"""
    obs, _ = env.reset(options={"start": start})
    lams, ns = [], []
    done = False
    while not done:
        action = act_fn(obs)
        obs, _, terminated, truncated, info = env.step(action)
        lams.append(info["lam"])
        ns.append(info["n"])
        done = terminated or truncated
    return np.array(lams), np.array(ns)


def _plt():
    # GUI 없는 환경(서버·터미널)에서도 그림 파일로 저장되게 설정
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_learning_curve(returns_by_seed, path, title="Learning curve"):
    plt = _plt()
    arr = np.array(returns_by_seed)                      # shape: (seed 수, 에피소드 수)
    mean = arr.mean(0)
    se = arr.std(0) / np.sqrt(arr.shape[0])
    x = np.arange(len(mean))
    plt.figure(figsize=(7, 4))
    plt.plot(x, mean, label="mean")
    plt.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, alpha=0.3, label="95% CI")
    plt.xlabel("episode"); plt.ylabel("return"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_schedule(lams, ns, duration_ms, path, title="Load vs instances"):
    from .env import required_concurrency
    plt = _plt()
    C = required_concurrency(lams, duration_ms)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(lams, color="tab:gray", alpha=0.5, label="invocations/min")
    ax1.set_ylabel("invocations/min", color="tab:gray")
    ax2 = ax1.twinx()
    ax2.plot(C, color="tab:orange", label="required concurrency")
    ax2.plot(ns, color="tab:blue", label="warm instances (policy)")
    ax2.set_ylabel("instances / concurrency")
    ax1.set_xlabel("minute"); plt.title(title)
    ax2.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_pareto(points_by_method, path, title="Cost vs cold-start Pareto"):
    """비용-콜드스타트율 트레이드오프 그래프 ― {이름: [(cost, rate), ...]} 형식"""
    plt = _plt()
    plt.figure(figsize=(6, 5))
    for method, pts in points_by_method.items():
        pts = sorted(pts)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("total cost (instance-minutes)")
    plt.ylabel("cold-start rate")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
