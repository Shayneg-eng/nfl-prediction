"""Refit the frozen ensembles on every season (1999-2024) for predicting future games.
Run after 02_final_test.py. Load with joblib and call predict() below."""
import json, warnings
import joblib
import numpy as np
from common import *
warnings.filterwarnings("ignore")

sel = json.load(open("results/selection.json"))
d = load(include_test=True)
cols = feature_cols(d)
C = candidates(cols)
prod = {}
for n in sorted(set(sel["winner_ensemble"]) | set(sel["margin_ensemble"])):
    m = C[n][1]().fit(d)
    if hasattr(m, "make_est"):
        m.make_est = None          # factory lambda isn't picklable, and isn't needed after fitting
    prod[n] = m
joblib.dump({"models": prod, "selection": sel, "trained_through": int(d.season.max())},
            "results/production_models.joblib")


def predict(bundle, features_df):
    """features_df: rows shaped like nfl_model_features (needs neutral_site). Returns
    P(home win) and predicted home margin from the frozen ensembles."""
    f = features_df.copy()
    f["home_adv"] = 1 - f.neutral_site
    ms, sel = bundle["models"], bundle["selection"]
    p = np.mean([ms[n].predict_proba(f) for n in sel["winner_ensemble"]], axis=0)
    pm = np.mean([ms[n].predict_margin(f) for n in sel["margin_ensemble"]], axis=0)
    return p, pm


if __name__ == "__main__":
    b = joblib.load("results/production_models.joblib")
    p, pm = predict(b, d.tail(3))
    print("saved; smoke test on last 3 games:", np.round(p, 3), np.round(pm, 1))
