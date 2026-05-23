"""
학습/평가 엔트리포인트
기본은 합성 트레이스로 즉시 실행 가능. data_prep으로 만든 .npz(키: invocations, duration_ms)를 --data로 넘기면 실제 Azure 데이터 사용

ex)
  python -m src.train --algo dqn --scenario steady --steps 50000 --seed 0
  python -m src.train --algo ppo --data data/processed/func_steady.npz --steps 100000
  python -m src.train --algo all --scenario spiky --steps 50000 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .env import ServerlessScalingEnv, make_synthetic_trace, required_concurrency
from .baselines import BASELINES
from .agents.dqn import train_dqn
from .agents.ppo import train_ppo, ppo_greedy_action
from .evaluate import aggregate_metrics, rollout_schedule
from .stats import mean_ci, compare


def load_trace(args):
    """--data 있으면 전처리된 실제 데이터 로드, 없으면 합성 트레이스 생성"""
    if args.data:
        d = np.load(args.data, allow_pickle=True)
        if args.n_max == 0 and "suggested_n_max" in d:
            args.n_max = int(d["suggested_n_max"])  # 전처리 시 저장한 권장값 사용
        return d["invocations"].astype(np.float64), float(d["duration_ms"])
    return make_synthetic_trace(kind=args.scenario, days=args.days, seed=0)


def make_env(inv, dur, args, *, random_start, action_mode):
    return ServerlessScalingEnv(
        inv, dur, k=args.k, n_max=args.n_max,
        w_cost=args.w_cost, w_cold=args.w_cold, w_idle=args.w_idle,
        action_mode=action_mode, episode_len=args.episode_len,
        random_start=random_start)


def split_train_test(inv, args):
    # 앞 train_days일을 학습, 나머지를 테스트 ― 시간 순서 유지
    cut = min(args.train_days * 1440, len(inv))
    return inv[:cut], inv[cut:] if len(inv) > cut else inv[:cut]


def train_one(algo, inv_tr, inv_te, dur, args, seed):
    """algo/seed 조합 하나 학습 + 테스트 평가 ― (metrics, logs, model, schedule) 반환"""
    action_mode = "discrete"
    train_env = make_env(inv_tr, dur, args, random_start=True, action_mode=action_mode)
    test_env  = make_env(inv_te, dur, args, random_start=False, action_mode=action_mode)
    test_env.episode_len = len(inv_te)  # 테스트 구간 전체를 한 에피소드로 평가

    if algo == "dqn":
        agent, logs = train_dqn(train_env, total_steps=args.steps, seed=seed,
                                gamma=args.gamma)
        act_fn    = lambda o: agent.act(o, epsilon=0.0)  # epsilon=0: 탐험 없이 학습된 정책만 사용
        model_obj = agent.q
    elif algo == "ppo":
        model, logs = train_ppo(train_env, total_steps=args.steps, seed=seed,
                                gamma=args.gamma)
        act_fn    = lambda o: ppo_greedy_action(model, o)
        model_obj = model
    else:
        raise ValueError(algo)

    lams, ns = rollout_schedule(test_env, act_fn, start=0)
    metrics  = aggregate_metrics(lams, ns, dur,
                                 weights=(args.w_cost, args.w_cold, args.w_idle))
    return metrics, logs, model_obj, (lams, ns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo",       choices=["dqn", "ppo", "all"], default="dqn")
    p.add_argument("--data",       type=str,   default=None, help="processed .npz path")
    p.add_argument("--scenario",   choices=["steady", "spiky", "bursty"], default="steady")
    p.add_argument("--days",       type=int,   default=14)
    p.add_argument("--train_days", type=int,   default=10)
    p.add_argument("--seeds",      type=int,   nargs="+", default=[0])
    p.add_argument("--steps",      type=int,   default=50_000)
    p.add_argument("--k",          type=int,   default=10)
    p.add_argument("--n_max",      type=int,   default=0, help="0이면 --data의 suggested_n_max(없으면 64) 사용")
    p.add_argument("--episode_len",type=int,   default=1440)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--w_cost",     type=float, default=1.0)
    p.add_argument("--w_cold",     type=float, default=5.0)
    p.add_argument("--w_idle",     type=float, default=0.0)
    p.add_argument("--outdir",     type=str,   default="results")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    inv, dur = load_trace(args)
    if args.n_max == 0:
        args.n_max = 64
    inv_tr, inv_te = split_train_test(inv, args)
    print(f"[data] T={len(inv)} train={len(inv_tr)} test={len(inv_te)} "
          f"duration={dur}ms n_max={args.n_max}")

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)  # 같은 seed면 항상 같은 결과 (재현성)
    torch.set_num_threads(max(1, os.cpu_count() // 2))        # 멀티 seed 학습 시 CPU 과부하 방지

    algos   = ["dqn", "ppo"] if args.algo == "all" else [args.algo]
    weights = (args.w_cost, args.w_cold, args.w_idle)
    results = {"meta": {"scenario": args.scenario, "data": args.data,
                        "steps": args.steps, "seeds": args.seeds, "k": args.k,
                        "n_max": args.n_max, "gamma": args.gamma,
                        "duration_ms": dur, "weights": list(weights)}}

    results["baselines"] = {}
    schedules = {}
    for name, fn in BASELINES.items():
        ns = fn(inv_te, dur, n_max=args.n_max)
        results["baselines"][name] = aggregate_metrics(inv_te, ns, dur, weights=weights)
        schedules[name] = np.asarray(ns)

    curves = {}
    per_seed_by_algo = {}
    for algo in algos:
        per_seed = []
        returns_by_seed = []
        for seed in args.seeds:
            metrics, logs, model_obj, sched = train_one(algo, inv_tr, inv_te, dur, args, seed)
            per_seed.append(metrics)
            returns_by_seed.append(logs["episode_return"])
            if seed == args.seeds[0]:
                schedules[algo] = np.asarray(sched[1])
            ckpt = os.path.join(args.outdir, f"{algo}_seed{seed}.pt")
            torch.save(model_obj.state_dict(), ckpt)
            print(f"[{algo} seed={seed}] cold={metrics['coldstart_rate']:.4f} "
                  f"cost={metrics['total_cost']:.0f} util={metrics['mean_util']:.3f}")
        agg = {k: mean_ci([m[k] for m in per_seed]) for k in per_seed[0].keys()}  # seed별 결과 → 95% CI 집계
        results[algo] = {"per_seed": per_seed, "agg": agg}
        per_seed_by_algo[algo] = per_seed
        curves[algo] = returns_by_seed

    # RL vs HPA ― RL의 CI 상단이 HPA보다 낮으면 유의미한 우위
    results["comparisons"] = {}
    for algo in algos:
        for metric in ["coldstart_rate", "total_cost"]:
            ci   = results[algo]["agg"][metric]
            base = results["baselines"]["hpa"][metric]
            results["comparisons"][f"{algo}_vs_hpa_{metric}"] = {
                "rl_mean": ci["mean"], "rl_ci": [ci["ci_lo"], ci["ci_hi"]],
                "hpa": base, "rl_better_significant": bool(base > ci["ci_hi"])}
    if "dqn" in algos and "ppo" in algos:
        # DQN vs PPO Welch 검정 ― CI가 0을 벗어나면 유의미한 차이
        for metric in ["coldstart_rate", "total_cost", "mean_reward"]:
            results["comparisons"][f"dqn_vs_ppo_{metric}"] = compare(
                [m[metric] for m in per_seed_by_algo["dqn"]],
                [m[metric] for m in per_seed_by_algo["ppo"]])

    def _json_default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    np.savez(os.path.join(args.outdir, "curves.npz"),
             **{f"{a}_returns": np.array([np.array(r) for r in curves[a]], dtype=object)
                for a in algos})
    np.savez(os.path.join(args.outdir, "schedule.npz"),
             inv=inv_te, duration_ms=dur,
             C=required_concurrency(inv_te, dur), **schedules)
    print(f"[done] → {args.outdir}/ (metrics.json, curves.npz, schedule.npz, *.pt)")
    for algo in algos:
        ci = results[algo]["agg"]["coldstart_rate"]
        print(f"  {algo}: coldstart_rate {ci['mean']:.4f} "
              f"[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] (95% CI, n={ci['n']})")
    print(f"  baselines coldstart_rate: " + json.dumps(
        {k: round(v["coldstart_rate"], 4) for k, v in results["baselines"].items()}))


if __name__ == "__main__":
    main()
