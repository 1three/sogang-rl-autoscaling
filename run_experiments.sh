#!/bin/bash
# 전체 실험 일괄 실행: 시나리오별 알고리즘 비교 + w_cold 스윕(Pareto)
# 사용: bash run_experiments.sh
set -e
PY=.venv/bin/python
STEPS=${STEPS:-60000}

echo "=== [1/2] 시나리오별 알고리즘 비교 (DQN/PPO, 5 seeds) ==="
for s in steady spiky bursty; do
  $PY -m src.train --algo all --data data/processed/func_$s.npz \
      --steps $STEPS --seeds 0 1 2 3 4 --outdir results/$s
  $PY -m src.make_plots --indir results/$s
done

echo "=== [2/2] w_cold 스윕 → Pareto (spiky, 3 seeds) ==="
for w in 1 3 10 30; do
  $PY -m src.train --algo all --data data/processed/func_spiky.npz \
      --steps $STEPS --seeds 0 1 2 --w_cold $w --outdir results/wcold_$w
done
$PY -m src.make_plots --pareto results/wcold_1 results/wcold_3 results/wcold_10 results/wcold_30

echo "ALL DONE"
