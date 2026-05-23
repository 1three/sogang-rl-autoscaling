"""
실험 결과 시각화 (train.py가 outdir에 남긴 metrics.json/curves.npz/schedule.npz를 읽음)

사용:
  python -m src.make_plots --indir results/steady
  python -m src.make_plots --pareto results/wcold_1 results/wcold_3 results/wcold_10 results/wcold_30
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경(서버·터미널)에서도 그림 파일로 저장
import matplotlib.pyplot as plt

from .stats import curve_mean_ci

METHODS = ["dqn", "ppo", "hpa", "oracle", "fixed", "seasonal"]
COLORS = {"dqn": "tab:blue", "ppo": "tab:green", "hpa": "tab:orange",
          "oracle": "k", "fixed": "tab:red", "seasonal": "tab:purple"}


def _smooth(x, w=20):
    """이동 평균으로 학습곡선 노이즈 제거"""
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_learning(indir):
    path = os.path.join(indir, "curves.npz")
    if not os.path.exists(path):
        return
    d = np.load(path, allow_pickle=True)
    plt.figure(figsize=(7, 4))
    for key in d.files:
        algo = key.replace("_returns", "")
        returns = [np.asarray(r, dtype=np.float64) for r in d[key]]
        mean, lo, hi = curve_mean_ci(returns)
        mean, lo, hi = _smooth(mean), _smooth(lo), _smooth(hi)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=f"{algo} (mean)", color=COLORS.get(algo))
        plt.fill_between(x, lo, hi, alpha=0.25, color=COLORS.get(algo))
    plt.xlabel("episode"); plt.ylabel("episode return")
    plt.title("Learning curve (mean ± 95% CI over seeds)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(indir, "learning_curve.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"[plot] {out}")


def plot_comparison(indir):
    m = json.load(open(os.path.join(indir, "metrics.json")))
    present = [x for x in METHODS if x in m or x in m.get("baselines", {})]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [(axes[0], "coldstart_rate", "Cold-start rate (↓)"),
                              (axes[1], "total_cost", "Total cost (instance-min, ↓)")]:
        vals, errs, cols = [], [], []
        for x in present:
            if x in ("dqn", "ppo") and x in m:                  # CI 있는 RL
                ci = m[x]["agg"][metric]
                vals.append(ci["mean"]); errs.append(ci["half"])
            else:
                vals.append(m["baselines"][x][metric]); errs.append(0.0)
            cols.append(COLORS.get(x))
        ax.bar(present, vals, yerr=errs, capsize=4, color=cols)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=30)
    sc = m.get("meta", {}).get("scenario")
    fig.suptitle(f"Method comparison — scenario: {sc}  (error bar = 95% CI)")
    plt.tight_layout()
    out = os.path.join(indir, "comparison.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"[plot] {out}")


def plot_schedule(indir, window=2880):
    path = os.path.join(indir, "schedule.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    C = d["C"][:window]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(C, color="tab:gray", lw=1.2, label="required concurrency (C)")
    for algo in ["dqn", "hpa", "oracle"]:
        if algo in d.files:
            ax.plot(d[algo][:window], color=COLORS.get(algo), lw=1.0,
                    alpha=0.8, label=f"{algo} instances")
    ax.set_xlabel("minute (test set)"); ax.set_ylabel("instances / concurrency")
    ax.set_title("Policy behavior: required concurrency vs provisioned instances")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(indir, "schedule.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"[plot] {out}")


def plot_pareto(dirs, out="results/pareto.png"):
    """w_cold 스윕 결과 → 비용-콜드스타트 Pareto 곡선 생성"""
    series = {"dqn": [], "ppo": [], "hpa": []}
    for dd in dirs:
        m = json.load(open(os.path.join(dd, "metrics.json")))
        for algo in ["dqn", "ppo"]:
            if algo in m:
                a = m[algo]["agg"]
                series[algo].append((a["total_cost"]["mean"], a["coldstart_rate"]["mean"]))
        series["hpa"].append((m["baselines"]["hpa"]["total_cost"],
                              m["baselines"]["hpa"]["coldstart_rate"]))
    plt.figure(figsize=(6, 5))
    for algo, pts in series.items():
        if not pts:
            continue
        pts = sorted(pts)
        plt.plot([p[0] for p in pts], [p[1] for p in pts],
                 marker="o", label=algo, color=COLORS.get(algo))
    plt.xlabel("total cost (instance-minutes)"); plt.ylabel("cold-start rate")
    plt.title("Cost vs cold-start Pareto (w_cold sweep)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=120); plt.close()
    print(f"[plot] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", type=str, default=None)
    p.add_argument("--pareto", type=str, nargs="+", default=None)
    args = p.parse_args()
    if args.indir:
        plot_learning(args.indir)
        plot_comparison(args.indir)
        plot_schedule(args.indir)
    if args.pareto:
        plot_pareto(args.pareto)


if __name__ == "__main__":
    main()
