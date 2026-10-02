"""Walk-forward model selection on 2015-2022. Never loads 2023-2024."""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from common import *
warnings.filterwarnings("ignore")

only = sys.argv[1:]  # optional subset of candidate names (for splitting long runs)
d = load(include_test=False)
assert not d.season.isin(TEST_SEASONS).any()
cols = feature_cols(d)
cands = candidates(cols)
if only:
    cands = {k: v for k, v in cands.items() if any(k.startswith(o) for o in only)}
rows = []
for name, (kind, make) in cands.items():
    t = time.time()
    oof = []
    for s in VAL_SEASONS:
        tr, va = d[d.season < s], d[d.season == s]
        m = make().fit(tr)
        o = va[["game_id", "season", "home_win", "margin"]].copy()
        o["p"] = m.predict_proba(va)
        if kind == "margin":
            o["pred_margin"] = m.predict_margin(va)
        oof.append(o)
    oof = pd.concat(oof)
    os.makedirs("results/validation_oof", exist_ok=True)
    oof.to_parquet(f"results/validation_oof/oof_{name}.parquet", index=False)
    r = {"model": name, "kind": kind, **win_metrics(oof.home_win.values, oof.p.values)}
    if kind == "margin":
        r.update(margin_metrics(oof.margin.values, oof.pred_margin.values))
    r["secs"] = round(time.time() - t, 1)
    rows.append(r)
    print(json.dumps(r), flush=True)
