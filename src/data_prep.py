"""
Azure Functions Trace 2019 다운로드·전처리

사용:
  python -m src.data_prep
  python -m src.data_prep --days 7
"""
from __future__ import annotations

import argparse
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


def pick_functions(raw_dir):
    """day1 기준으로 steady/spiky/bursty 대표 함수 각 1개 선정"""
    path = os.path.join(raw_dir, "invocations_per_function_md.anon.d01.csv")
    df = pd.read_csv(path)
    have = [c for c in MINUTE_COLS if c in df.columns]
    mat = df[have].fillna(0).to_numpy(dtype=np.float64)
    hf = df["HashFunction"].to_numpy()

    total = mat.sum(1)
    zero_frac = (mat == 0).mean(1)
    peak = mat.max(1)
    mean = mat.mean(1) + 1e-9
    active = total > np.percentile(total, 50)

    # steady: 거의 항상 호출이 있고 볼륨 높은 것
    steady_score = np.where(active & (zero_frac < 0.2), total, -1)
    steady_fn = hf[np.argmax(steady_score)]

    # spiky: 가끔 호출되지만 peak가 높은 것
    spiky_score = np.where(active & (zero_frac > 0.4) & (zero_frac < 0.9),
                           peak / mean, -1)
    spiky_fn = hf[np.argmax(spiky_score)]

    # bursty: 항상 활성이고 변동이 큰 것
    bursty_score = np.where(active & (zero_frac < 0.15) & (peak / mean > 2),
                            peak / mean, -1)
    bursty_fn = hf[np.argmax(bursty_score)]

    print(f"[select] steady={steady_fn[:8]} spiky={spiky_fn[:8]} bursty={bursty_fn[:8]}")
    return {"steady": steady_fn, "spiky": spiky_fn, "bursty": bursty_fn}


def build_series(raw_dir, chosen, days):
    """선택된 함수들의 days*1440 시계열 구성"""
    all_hf = set(chosen.values())
    series = {hf: np.zeros(days * 1440, dtype=np.float64) for hf in all_hf}

    for d in range(1, days + 1):
        path = os.path.join(raw_dir, f"invocations_per_function_md.anon.d{d:02d}.csv")
        if not os.path.exists(path):
            break
        print(f"\r[series] day {d} 처리 중...", end="", flush=True)
        df = pd.read_csv(path)
        df = df[df["HashFunction"].isin(all_hf)]
        have = [c for c in MINUTE_COLS if c in df.columns]
        for _, row in df.iterrows():
            hf = row["HashFunction"]
            series[hf][(d - 1) * 1440: d * 1440] = row[have].fillna(0).to_numpy()
    print()
    return series


def get_duration(raw_dir, hf, days):
    """함수의 평균 실행시간(ms)"""
    vals = []
    for d in range(1, days + 1):
        path = os.path.join(raw_dir, f"function_durations_percentiles.anon.d{d:02d}.csv")
        if not os.path.exists(path):
            break
        df = pd.read_csv(path)
        row = df[df["HashFunction"] == hf]
        if not row.empty and np.isfinite(row["Average"].values[0]):
            vals.append(float(row["Average"].values[0]))
    return float(np.mean(vals)) if vals else 100.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--out_dir", default="data/processed")
    args = p.parse_args()

    archive = download(args.raw_dir)
    extract(archive, args.raw_dir)

    chosen = pick_functions(args.raw_dir)
    series = build_series(args.raw_dir, chosen, args.days)

    os.makedirs(args.out_dir, exist_ok=True)
    for kind, hf in chosen.items():
        x = series[hf]
        dur = get_duration(args.raw_dir, hf, args.days)
        out = os.path.join(args.out_dir, f"func_{kind}.npz")
        np.savez(out, invocations=x, duration_ms=np.float64(dur),
                 hash_function=hf, kind=kind)
        print(f"[save] {kind:7s} → {out}  dur={dur:.0f}ms peak={x.max():.0f}")
    print("[done]")


if __name__ == "__main__":
    main()
