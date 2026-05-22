"""
학습/평가 엔트리포인트 ― DQN 학습, baseline 비교, 결과 저장

사용 예:
  python -m src.train --scenario steady --steps 50000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .env import ServerlessScalingEnv, make_synthetic_trace
from .baselines import BASELINES
from .agents.dqn import train_dqn
from .evaluate import aggregate_metrics, rollout_schedule


def load_trace(args):
    """--data 있으면 전처리된 실제 데이터 로드, 없으면 합성 트레이스 생성"""
    if args.data:
        d = np.load(args.data, allow_pickle=True)
        if args.n_max == 0 and "suggested_n_max" in d:
            args.n_max = int(d["suggested_n_max"])  # 전처리 시 저장한 권장값 사용
        return d["invocations"].astype(np.float64), float(d["duration_ms"])
    return make_synthetic_trace(kind=args.scenario, days=args.days, seed=0)


def make_env(inv, dur, args, *, random_start):
    return ServerlessScalingEnv(
        inv, dur, k=args.k, n_max=args.n_max,
        w_cost=args.w_cost, w_cold=args.w_cold, w_idle=0.0,
        action_mode="discrete", episode_len=args.episode_len,
        random_start=random_start)


def split_train_test(inv, args):
    # 앞 train_days 일을 학습, 나머지를 테스트 (시간 순서 유지)
    cut = min(args.train_days * 1440, len(inv))
    return inv[:cut], inv[cut:] if len(inv) > cut else inv[:cut]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str,   default=None)
    p.add_argument("--scenario",   choices=["steady", "spiky", "bursty"], default="steady")
    p.add_argument("--days",       type=int,   default=14)
    p.add_argument("--train_days", type=int,   default=10)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--steps",      type=int,   default=50_000)
    p.add_argument("--k",          type=int,   default=10)
    p.add_argument("--n_max",      type=int,   default=0)
    p.add_argument("--episode_len",type=int,   default=1440)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--w_cost",     type=float, default=1.0)
    p.add_argument("--w_cold",     type=float, default=5.0)
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

    weights = (args.w_cost, args.w_cold, 0.0)
    train_env = make_env(inv_tr, dur, args, random_start=True)
    test_env  = make_env(inv_te, dur, args, random_start=False)
    test_env.episode_len = len(inv_te)  # 테스트 구간 전체를 한 에피소드로 평가

    agent, logs = train_dqn(train_env, total_steps=args.steps, seed=args.seed, gamma=args.gamma)
    act_fn      = lambda o: agent.act(o, epsilon=0.0)  # epsilon=0: 탐험 없이 학습된 정책만 사용
    model_obj   = agent.q

    lams, ns = rollout_schedule(test_env, act_fn, start=0)
    metrics  = aggregate_metrics(lams, ns, dur, weights=weights)

    results = {"meta": {"scenario": args.scenario, "algo": "dqn", "seed": args.seed, "steps": args.steps}, "rl": metrics, "baselines": {}}
    for name, fn in BASELINES.items():
        results["baselines"][name] = aggregate_metrics(inv_te, fn(inv_te, dur, n_max=args.n_max), dur, weights=weights)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    torch.save(model_obj.state_dict(), os.path.join(args.outdir, f"dqn_seed{args.seed}.pt"))
    print(f"[dqn seed={args.seed}] "
          f"cold={metrics['coldstart_rate']:.4f} cost={metrics['total_cost']:.0f}")
    print("baselines coldstart: " +
          str({k: round(v["coldstart_rate"], 4)
               for k, v in results["baselines"].items()}))


if __name__ == "__main__":
    main()
