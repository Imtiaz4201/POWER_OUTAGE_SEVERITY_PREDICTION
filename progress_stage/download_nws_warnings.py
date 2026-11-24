"""
Download NWS watches / warnings / advisories (VTEC events) from the Iowa Environmental Mesonet.

Source: https://mesonet.agron.iastate.edu/json/vtec_events_bystate.py?state=XX&year=YYYY
One request per state and year; only the columns needed are kept. Each event has
  issue        time the event was issued (UTC)
  init_expire  expiry time announced AT ISSUE (UTC)  -> known in real time
  expire       final expiry after later updates (UTC) -> NOT used (known only afterwards)

Output: Data/nws_raw/nws_events_<STATE>_<YEAR>.csv.gz  and  Data/nws_raw/nws_events.parquet (all combined)
Run from the project folder or from progress_stage2:  python download_nws_warnings.py
Already-downloaded files are skipped, so the script can be re-run safely.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "Data", "nws_raw")
YEARS = range(2005, 2025)
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
          "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
          "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
          "VT", "VA", "WA", "WV", "WI", "WY", "PR"]
KEEP = ["phenomena", "significance", "issue", "init_expire", "wfo", "area"]
URL = "https://mesonet.agron.iastate.edu/json/vtec_events_bystate.py?state={s}&year={y}"
WORKERS = 6


def fetch(state, year, tries=4):
    path = os.path.join(OUT, f"nws_events_{state}_{year}.csv.gz")
    if os.path.exists(path):
        return state, year, "cached"
    for k in range(tries):
        try:
            r = requests.get(URL.format(s=state, y=year), timeout=180,
                             headers={"User-Agent": "outage-severity-research (student project)"})
            r.raise_for_status()
            ev = pd.DataFrame(r.json().get("events", []))
            ev = ev.reindex(columns=KEEP)
            ev.insert(0, "state", state)
            ev.to_csv(path, index=False, compression="gzip")
            return state, year, len(ev)
        except Exception as e:  # network hiccup: wait and retry
            if k == tries - 1:
                return state, year, f"FAILED: {e}"
            time.sleep(10 * (k + 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, y) for s in STATES for y in YEARS]
    t0, done = time.time(), 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch, s, y) for s, y in jobs]
        for f in as_completed(futs):
            s, y, res = f.result()
            done += 1
            if done % 50 == 0 or str(res).startswith("FAILED"):
                print(f"{done}/{len(jobs)}  last: {s} {y} -> {res}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    files = [os.path.join(OUT, f) for f in os.listdir(OUT) if f.startswith("nws_events_") and f.endswith(".csv.gz")]
    allev = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    allev.to_parquet(os.path.join(OUT, "nws_events.parquet"), index=False)
    missing = len(jobs) - len(files)
    print(f"Done: {len(allev)} events from {len(files)} state-years ({missing} missing), "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
