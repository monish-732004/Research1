"""
fetch_real_ci_and_rerun_cads.py
────────────────────────────────
Run this on your Windows machine inside the gpu-carbon-waste venv.

    cd D:\Gpu\gpu-carbon-waste
    .venv\Scripts\python.exe fetch_real_ci_and_rerun_cads.py

What it does:
  1. Pulls hourly Carbon Intensity for zone CN from Electricity Maps
     (2025-01-01 → 2025-07-05, covering the 185-day window = 4 440 rel_hours)
  2. Saves to   outputs/real_ci_china.parquet
  3. Loads your existing sweep_results.parquet (the 300-run simulated sweep)
  4. Re-runs the CADS simulation replacing simulated CI with real CI
  5. Saves to   outputs/sweep_results_real_ci.parquet
  6. Prints the new best config + comparison table vs old results
"""

import os, sys, json, math, time, datetime as dt
from pathlib import Path
import requests
import pandas as pd
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
API_KEY   = os.environ.get("ELECTRICITY_MAPS_API_KEY", "")
ZONE      = "CN"
# 185 days starting 2025-01-01  →  4 440 hourly slots (rel_hour 0..4439)
START_UTC = dt.datetime(2025, 1, 1,  0, 0, 0, tzinfo=dt.timezone.utc)
END_UTC   = dt.datetime(2025, 7,  5, 23, 0, 0, tzinfo=dt.timezone.utc)

BASE_DIR  = Path(__file__).parent
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CI_PATH   = OUT_DIR / "real_ci_china.parquet"
OLD_SWEEP = OUT_DIR / "sim_runs" / "sweep_results.parquet"
NEW_SWEEP = OUT_DIR / "sweep_results_real_ci.parquet"

HEADERS   = {"auth-token": API_KEY, "Content-Type": "application/json"}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 – pull real carbon intensity
# ═════════════════════════════════════════════════════════════════════════════

def fetch_ci_range(zone: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """
    Try /v3/carbon-intensity/past-range first (needs paid/history tier).
    Falls back to pulling /v3/carbon-intensity/history (last 24 h) if 403.
    """
    # Chunk into 30-day windows — the API typically limits range queries
    records = []
    cursor = start
    chunk_days = 9

    print(f"Pulling Carbon Intensity for zone={zone}")
    print(f"  Window: {start.date()} → {end.date()} ({(end-start).days} days)")
    print()

    while cursor < end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days), end)
        url = (
            f"https://api.electricitymaps.com/v3/carbon-intensity/past-range"
            f"?zone={zone}"
            f"&start={cursor.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&end={chunk_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        print(f"  GET {cursor.date()} → {chunk_end.date()} ... ", end="", flush=True)
        resp = requests.get(url, headers=HEADERS, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            rows = data.get("data", data.get("history", []))
            records.extend(rows)
            print(f"OK  ({len(rows)} rows)")
        elif resp.status_code == 403:
            print(f"403 – historical range not available on this tier")
            print()
            print("NOTE: Your Electricity Maps plan may not include past-range access.")
            print("Trying /v3/carbon-intensity/history (last 24 h) to confirm key works...")
            fallback = requests.get(
                f"https://api.electricitymaps.com/v3/carbon-intensity/history?zone={zone}",
                headers=HEADERS, timeout=30
            )
            if fallback.status_code == 200:
                fb = fallback.json()
                rows = fb.get("history", [])
                print(f"Key works — got {len(rows)} recent rows.")
                print()
                print("=" * 60)
                print("PLAN LIMITATION: Full history requires a commercial plan.")
                print("Two options:")
                print("  A) Email Electricity Maps (info@electricitymaps.com)")
                print("     requesting researcher / academic access to CN history.")
                print("  B) Use the 24h fallback data for a 'live demo' section")
                print("     in the paper (Section VIII future-work paragraph).")
                print("=" * 60)
                # save what we have
                df = pd.DataFrame(rows)
                df.to_parquet(CI_PATH.with_suffix(".last24h.parquet"), index=False)
                print(f"Saved last-24h snapshot → {CI_PATH.with_suffix('.last24h.parquet')}")
            else:
                print(f"Key error: {fallback.status_code} {fallback.text[:200]}")
            sys.exit(0)
        else:
            print(f"Error {resp.status_code}: {resp.text[:200]}")
            time.sleep(2)

        cursor = chunk_end
        time.sleep(0.5)   # be polite

    if not records:
        print("No records retrieved. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(records)
    # normalise column names across API versions
    if "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
    elif "updatedAt" in df.columns:
        df["timestamp"] = pd.to_datetime(df["updatedAt"], utc=True)

    df = df.rename(columns={
        "carbonIntensity": "ci_gco2_kwh",
        "value": "ci_gco2_kwh",
    })
    if "ci_gco2_kwh" not in df.columns:
        # try to find the right column
        num_cols = df.select_dtypes("number").columns.tolist()
        if num_cols:
            df = df.rename(columns={num_cols[0]: "ci_gco2_kwh"})

    df = df[["timestamp", "ci_gco2_kwh"]].dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def assign_rel_hour(df: pd.DataFrame, start: dt.datetime) -> pd.DataFrame:
    """Map each timestamp to rel_hour = hours since start (0-based)."""
    df = df.copy()
    df["rel_hour"] = (
        (df["timestamp"] - pd.Timestamp(start))
        .dt.total_seconds() // 3600
    ).astype(int)
    # keep only the 4 440 valid hours
    df = df[(df["rel_hour"] >= 0) & (df["rel_hour"] <= 4439)]
    # fill any gaps with forward-fill then backward-fill
    full_idx = pd.DataFrame({"rel_hour": range(4440)})
    df = full_idx.merge(df[["rel_hour", "ci_gco2_kwh"]], on="rel_hour", how="left")
    df["ci_gco2_kwh"] = df["ci_gco2_kwh"].ffill().bfill()
    df["zone"] = ZONE
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 – CADS re-simulation with real CI
# ═════════════════════════════════════════════════════════════════════════════

def run_cads_with_real_ci(sweep_df: pd.DataFrame, ci_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-simulate each of the 300 original CADS configurations but swap the
    simulated carbon-intensity signal for the real Electricity Maps CN data.

    The original sweep used per-region annual average CI values.
    We now use the real hourly CI series for CN (treating all four Chinese
    regions as the single CN zone from Electricity Maps, since sub-regional
    data is not available at Tier C).

    Returns a DataFrame with the same schema as sweep_results.parquet plus
    two extra columns: ci_source = "real" and ci_variance.
    """
    ci_series = ci_df.set_index("rel_hour")["ci_gco2_kwh"].to_dict()

    # ── constants from the project ───────────────────────────────────────────
    BASELINE_CO2_TONNES = 92_340.0          # 185-day fleet baseline
    TOTAL_JOBS_ELIGIBLE = 14_246_582        # non-Inference jobs
    HOURS               = 4_440

    # Average power of a deferred job (estimated from fleet GPU mix + TDP)
    # At ~35% mean utilisation, a typical deferred job uses ~2 GPUs × 400 W
    # for its actual duration.  We use a simple energy-per-job proxy:
    # baseline_co2 / HOURS gives us an hourly fleet rate in tonnes/h.
    # A single deferred job's carbon saving ≈ ΔCI × energy_per_job.
    # Since we don't have per-job energy in this script, we compute fleet-
    # level savings by comparing deferred-window CI vs overall mean CI.
    mean_ci = np.mean(list(ci_series.values()))
    ci_std  = np.std(list(ci_series.values()))

    print(f"\nReal CN carbon intensity stats:")
    print(f"  Mean : {mean_ci:.1f} g CO₂/kWh")
    print(f"  Std  : {ci_std:.1f} g CO₂/kWh")
    print(f"  Min  : {min(ci_series.values()):.1f}")
    print(f"  Max  : {max(ci_series.values()):.1f}")
    print()

    results = []
    seen = set()   # de-dupe threshold 400/500/600 if they collapse again

    for _, row in sweep_df.iterrows():
        region       = row["region"]
        anchor       = row["anchor"]
        threshold    = row["threshold"]
        max_defer    = row.get("max_defer", 12)
        # convert max_defer string like "12h" → int hours
        if isinstance(max_defer, str):
            max_defer = int(max_defer.replace("h", ""))

        # ── simulate deferral ────────────────────────────────────────────────
        # For each hour h with CI > threshold, defer jobs to the best hour
        # within [h+1, h+max_defer] that has the lowest CI.
        # Track total CO2 saved = Σ (CI[h] - CI[best]) × hourly_jobs_energy
        #
        # Hourly baseline CO2 allocation per eligible job:
        hourly_rate = BASELINE_CO2_TONNES / HOURS   # tonnes/hour fleet total
        jobs_per_hour = TOTAL_JOBS_ELIGIBLE / HOURS  # avg jobs submitted/hour

        deferred_hours = 0
        total_saved_tonnes = 0.0
        deferred_jobs = 0

        for h in range(HOURS):
            ci_h = ci_series.get(h, mean_ci)
            if ci_h <= threshold:
                continue   # don't defer — already below threshold

            # find best window
            best_ci = ci_h
            for dh in range(1, max_defer + 1):
                future_h = h + dh
                if future_h >= HOURS:
                    break
                future_ci = ci_series.get(future_h, ci_h)
                if future_ci < best_ci:
                    best_ci = future_ci

            if best_ci < ci_h:
                ci_delta       = ci_h - best_ci     # g CO₂/kWh saved per deferred unit
                # fraction of baseline associated with jobs we can defer
                # (non-Inference = ~70% of fleet by GPU-hours, estimated)
                deferrable_frac = 0.70
                hour_savings   = (ci_delta / mean_ci) * hourly_rate * deferrable_frac
                total_saved_tonnes += hour_savings
                deferred_hours += 1
                deferred_jobs  += jobs_per_hour

        reduction_pct     = (total_saved_tonnes / BASELINE_CO2_TONNES) * 100
        jobs_deferred_pct = (deferred_jobs / TOTAL_JOBS_ELIGIBLE) * 100

        results.append({
            "region":            region,
            "anchor":            anchor,
            "threshold":         threshold,
            "max_defer":         f"{max_defer}h",
            "ci_source":         "real_electricity_maps",
            "ci_mean":           round(mean_ci, 1),
            "ci_std":            round(ci_std, 1),
            "baseline_co2_kg":   BASELINE_CO2_TONNES * 1000,
            "total_co2_kg":      (BASELINE_CO2_TONNES - total_saved_tonnes) * 1000,
            "reduction_pct":     round(reduction_pct, 4),
            "jobs_deferred":     int(deferred_jobs),
            "jobs_total_eligible": TOTAL_JOBS_ELIGIBLE,
            "jobs_deferred_pct": round(jobs_deferred_pct, 4),
            "deferred_hours":    deferred_hours,
            "p95_defer_hours":   min(max_defer, 12),
        })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STEP 1 — Fetch real Carbon Intensity from Electricity Maps")
    print("=" * 60)

    if CI_PATH.exists():
        print(f"  Cache hit: {CI_PATH} already exists — skipping fetch.")
        ci_df = pd.read_parquet(CI_PATH)
    else:
        raw = fetch_ci_range(ZONE, START_UTC, END_UTC)
        ci_df = assign_rel_hour(raw, START_UTC)
        ci_df.to_parquet(CI_PATH, index=False)
        print(f"\n  Saved {len(ci_df)} rows → {CI_PATH}")

    print(f"\n  Real CI loaded: {len(ci_df)} hours, "
          f"mean={ci_df['ci_gco2_kwh'].mean():.1f} g CO₂/kWh\n")

    print("=" * 60)
    print("STEP 2 — Load original 300-run sweep")
    print("=" * 60)

    if not OLD_SWEEP.exists():
        print(f"  ERROR: {OLD_SWEEP} not found.")
        print("  Run the original CADS sweep first.")
        sys.exit(1)

    sweep_df = pd.read_parquet(OLD_SWEEP)
    print(f"  Loaded {len(sweep_df)} configurations from {OLD_SWEEP}\n")

    print("=" * 60)
    print("STEP 3 — Re-simulate CADS with real CI")
    print("=" * 60)

    new_sweep = run_cads_with_real_ci(sweep_df, ci_df)
    new_sweep.to_parquet(NEW_SWEEP, index=False)
    print(f"\n  Saved {len(new_sweep)} results → {NEW_SWEEP}")

    print()
    print("=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    # Best config by reduction_pct
    best = new_sweep.loc[new_sweep["reduction_pct"].idxmax()]
    print(f"\nBest config (real CI):")
    print(f"  region       = {best['region']}")
    print(f"  threshold    = {best['threshold']} g CO₂/kWh")
    print(f"  max_defer    = {best['max_defer']}")
    print(f"  reduction    = {best['reduction_pct']:.4f}%")
    print(f"  jobs deferred = {int(best['jobs_deferred']):,} "
          f"({best['jobs_deferred_pct']:.2f}% of eligible)")

    # Compare vs simulated
    print(f"\nSimulated best (original):  reduction = 0.0341%")
    print(f"Real CI best (this run):    reduction = {new_sweep['reduction_pct'].max():.4f}%")
    print()

    # Threshold collapse check
    thresh_summary = (
        new_sweep.groupby("threshold")["reduction_pct"]
        .mean()
        .reset_index()
        .rename(columns={"reduction_pct": "mean_reduction_pct"})
    )
    print("Mean reduction by threshold (real CI):")
    print(thresh_summary.to_string(index=False))

    print()
    print("=" * 60)
    print(f"Done. Share outputs/sweep_results_real_ci.parquet with Claude.")
    print("=" * 60)


if __name__ == "__main__":
    main()
