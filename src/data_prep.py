"""
Azure Functions 트레이스 데이터 다운로드·전처리

1) 데이터 파일(~143MB) 다운로드·압축해제 → data/raw/
2) day1 통계로 카테고리별 후보 함수 추출 (각 40개)
3) 후보들의 14일치 분당 호출수를 이어붙이고 평균 실행시간 매칭
4) 카테고리별 대표 함수 1개 결정 → data/processed/func_<kind>.npz

사용:
  python -m src.data_prep            # 전체 14일
  python -m src.data_prep --days 7   # 앞 7일만 (빠름)
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd

DATA_URL = ("https://azurepublicdatasettraces.blob.core.windows.net/"
            "azurepublicdatasetv2/azurefunctions_dataset2019/"
            "azurefunctions-dataset2019.tar.xz")
MINUTE_COLS = [str(i) for i in range(1, 1441)]


def download(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    archive = os.path.join(raw_dir, "azurefunctions-dataset2019.tar.xz")
    if os.path.exists(archive) and os.path.getsize(archive) > 1_000_000:
        print(f"[download] 이미 존재: {archive}")
        return archive

    def _progress(block, bsize, total):
        done = block * bsize
        pct = 100 * done / total if total > 0 else 0
        print(f"\r[download] {done/1e6:6.1f} / {total/1e6:6.1f} MB ({pct:5.1f}%)",
              end="", flush=True)

    print(f"[download] {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, archive, _progress)
    print()
    return archive


def extract(archive, raw_dir):
    sample = os.path.join(raw_dir, "invocations_per_function_md.anon.d01.csv")
    if os.path.exists(sample):
        print("[extract] 이미 압축해제됨")
        return
    print("[extract] 압축해제 중...")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(raw_dir)
    print("[extract] 완료")


def day_paths(raw_dir, days):
    inv, dur = [], []
    for d in range(1, days + 1):
        ip = os.path.join(raw_dir, f"invocations_per_function_md.anon.d{d:02d}.csv")
        dp = os.path.join(raw_dir, f"function_durations_percentiles.anon.d{d:02d}.csv")
        if os.path.exists(ip):
            inv.append((d, ip))
        if os.path.exists(dp):
            dur.append((d, dp))
    return inv, dur


def _row_stats(mat):
    """함수별 호출 통계 계산 (행 단위) ― mat: (함수 수, 1440분)"""
    total = mat.sum(1)
    mean = mat.mean(1)
    peak = mat.max(1)
    std = mat.std(1)
    zero_frac = (mat == 0).mean(1)
    eps = 1e-9
    return {
        "total": total, "mean": mean, "peak": peak,
        "cv": std / (mean + eps),           # 변동계수: 클수록 불규칙
        "peak2mean": peak / (mean + eps),   # 최대/평균 비율: 클수록 급증형
        "zero_frac": zero_frac,             # 호출 없는 분 비율
    }


def pick_candidates(inv_paths, n_per_cat=40):
    """day1 기준 통계로 카테고리별 후보 함수 목록 반환"""
    _, path = inv_paths[0]
    print(f"[select] day1 후보 스캔: {path}")
    df = pd.read_csv(path)
    have = [c for c in MINUTE_COLS if c in df.columns]
    mat = df[have].fillna(0).to_numpy(dtype=np.float64)
    hf = df["HashFunction"].to_numpy()
    s = _row_stats(mat)

    vol_ok = s["total"] >= np.percentile(s["total"], 50)
    mid_high = ((s["total"] >= np.percentile(s["total"], 55)) &
                (s["total"] <= np.percentile(s["total"], 92)))

    def top(score, mask):
        score = np.where(mask, score, -np.inf)
        idx = np.argsort(score)[::-1][:n_per_cat]
        return set(hf[idx[np.isfinite(score[idx])]])

    steady = top(-s["zero_frac"] + 1e-6 * s["total"], vol_ok & (s["zero_frac"] < 0.3))
    spiky  = top(s["peak2mean"],
                 (s["zero_frac"] > 0.3) & (s["zero_frac"] < 0.9) & (s["peak"] > 30))
    bursty = top(s["cv"], mid_high & (s["zero_frac"] < 0.2))

    cands = steady | spiky | bursty
    print(f"[select] 후보 함수 수: {len(cands)} "
          f"(steady{len(steady)} spiky{len(spiky)} bursty{len(bursty)})")
    return cands


def build_series(inv_paths, candidates, days):
    """후보 함수들의 14일치 분당 시계열 구성 ― 전 기간 등장한 함수만 반환"""
    series = {hf: np.zeros(days * 1440, dtype=np.float64) for hf in candidates}
    seen = {hf: np.zeros(days, dtype=bool) for hf in candidates}
    for d, path in inv_paths:
        print(f"\r[series] day {d} 처리 중...", end="", flush=True)
        for chunk in pd.read_csv(path, chunksize=50_000):
            chunk = chunk[chunk["HashFunction"].isin(candidates)]
            if chunk.empty:
                continue
            have = [c for c in MINUTE_COLS if c in chunk.columns]
            vals = chunk[have].fillna(0).to_numpy(dtype=np.float64)
            for i, hf in enumerate(chunk["HashFunction"].to_numpy()):
                series[hf][(d - 1) * 1440:d * 1440] = vals[i]
                seen[hf][d - 1] = True
    print()
    full = {hf: s for hf, s in series.items() if seen[hf].all()}
    print(f"[series] 전체 {days}일 등장 함수: {len(full)}")
    return full


def get_durations(dur_paths, candidates):
    """후보 함수별 평균 실행시간(ms) ― 날짜 평균값 사용"""
    acc, cnt = {}, {}
    for d, path in dur_paths:
        df = pd.read_csv(path)
        df = df[df["HashFunction"].isin(candidates)]
        for hf, avg in zip(df["HashFunction"], df["Average"]):
            if np.isfinite(avg) and avg > 0:
                acc[hf] = acc.get(hf, 0.0) + avg
                cnt[hf] = cnt.get(hf, 0) + 1
    return {hf: acc[hf] / cnt[hf] for hf in acc}


def full_stats(x, days):
    """14일 전체 통계 계산 ― periodicity(하루 패턴 반복성, 높을수록 규칙적) 포함"""
    total, mean, peak = x.sum(), x.mean(), x.max()
    std = x.std()
    zero_frac = float((x == 0).mean())
    eps = 1e-9
    profiles = x.reshape(days, 1440)
    mean_prof = profiles.mean(0)
    if days > 1 and mean_prof.std() > 0:
        corrs = [np.corrcoef(profiles[d], mean_prof)[0, 1]
                 for d in range(days) if profiles[d].std() > 0]
        periodicity = float(np.nanmean(corrs)) if corrs else 0.0
    else:
        periodicity = 0.0
    return {"total": float(total), "mean": float(mean), "peak": float(peak),
            "cv": float(std / (mean + eps)), "peak2mean": float(peak / (mean + eps)),
            "zero_frac": zero_frac, "periodicity": periodicity}


def choose_best(series, durations, days):
    """후보 중 카테고리별 점수 높은 대표 함수 1개씩 반환"""
    rows = []
    for hf, x in series.items():
        if hf not in durations:
            continue
        st = full_stats(x, days)
        st["hash"] = hf
        st["duration_ms"] = durations[hf]
        st["peak_concurrency"] = st["peak"] * st["duration_ms"] / 60000.0
        rows.append(st)

    def in_band(r):
        return 5 <= r["peak_concurrency"] <= 55  # 학습에 적합한 동시성 범위

    def active(r):
        return r["total"] >= 200

    def best(score_fn, filt):
        cand = [r for r in rows if filt(r) and in_band(r) and active(r)]
        if not cand:
            cand = [r for r in rows if filt(r) and 3 <= r["peak_concurrency"] <= 120]
        if not cand:
            cand = rows
        return max(cand, key=score_fn)

    chosen = {
        "steady": best(lambda r: r["periodicity"] - r["zero_frac"],
                       lambda r: r["zero_frac"] < 0.2),
        "spiky":  best(lambda r: r["peak2mean"],
                       lambda r: 0.3 < r["zero_frac"] < 0.95),
        "bursty": best(lambda r: r["cv"],
                       lambda r: r["zero_frac"] < 0.2 and r["periodicity"] < 0.7),
    }
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days",      type=int, default=14)
    p.add_argument("--raw_dir",   default="data/raw")
    p.add_argument("--out_dir",   default="data/processed")
    p.add_argument("--n_per_cat", type=int, default=40)
    args = p.parse_args()

    archive = download(args.raw_dir)
    extract(archive, args.raw_dir)
    inv_paths, dur_paths = day_paths(args.raw_dir, args.days)
    days = len(inv_paths)
    if days == 0:
        raise SystemExit("invocation 파일을 찾지 못했습니다.")
    print(f"[main] 사용 일수: {days}")

    candidates = pick_candidates(inv_paths, args.n_per_cat)
    series     = build_series(inv_paths, candidates, days)
    durations  = get_durations(dur_paths, candidates)
    chosen     = choose_best(series, durations, days)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for kind, st in chosen.items():
        hf = st["hash"]
        x  = series[hf]
        out = os.path.join(args.out_dir, f"func_{kind}.npz")
        suggested_n_max = int(max(16, np.ceil(st["peak_concurrency"] * 1.3)))
        np.savez(out, invocations=x, duration_ms=np.float64(st["duration_ms"]),
                 hash_function=hf, kind=kind,
                 suggested_n_max=np.int64(suggested_n_max))
        summary[kind] = {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in st.items()}
        print(f"[save] {kind:7s} → {out}  "
              f"dur={st['duration_ms']:.0f}ms peakC={st['peak_concurrency']:.1f} "
              f"zero_frac={st['zero_frac']:.2f} period={st['periodicity']:.2f}")

    with open(os.path.join(args.out_dir, "selection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[done] data/processed/*.npz + selection_summary.json")


if __name__ == "__main__":
    main()
