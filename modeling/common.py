"""Shared data loading, feature prep and model definitions for modeling/.

Protocol (fixed before looking at any results):
  * model selection: walk-forward validation over seasons 2015-2022, where each
    season is predicted by models trained only on earlier seasons
  * final test: seasons 2023-2024, scored once, after selection is frozen
  * TEST_SEASONS rows are filtered out at load time in validation mode, so
    01_validate.py cannot see them even by accident.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROC = "../data/processed"
VAL_SEASONS = list(range(2015, 2023))
TEST_SEASONS = [2023, 2024]
CTX = ["is_playoff", "div_game", "is_indoor", "is_grass", "temp", "wind", "is_primetime"]


def load(include_test=False):
    f = pd.read_parquet(f"{PROC}/nfl_model_features_1999_2024.parquet")
    l = pd.read_parquet(f"{PROC}/nfl_model_labels_1999_2024.parquet")[["game_id", "home_win", "margin", "is_tie"]]
    d = f.merge(l, on="game_id", validate="one_to_one")
    if not include_test:
        d = d[~d.season.isin(TEST_SEASONS)]
    d = d[d.is_tie == 0].reset_index(drop=True)          # 14 ties dropped: no winner to predict
    d["home_adv"] = 1 - d.neutral_site                   # 1 = real home game, 0 = neutral
    return d


def feature_cols(d):
    return [c for c in d.columns if c.startswith("diff_")] + CTX + ["home_adv"]


def flip(X):
    """Same game from the other team's side: diffs negate, home advantage flips sign."""
    Xf = X.copy()
    diff = [c for c in X.columns if c.startswith("diff_")]
    Xf[diff] = -Xf[diff]
    Xf["home_adv"] = -Xf["home_adv"]
    return Xf


def augment(X, y):
    """Train on both orientations of every game so the model can't learn a
    spurious 'home' effect beyond the explicit home_adv feature. y is margin
    (negated) or win (1-y)."""
    Xa = pd.concat([X, flip(X)], ignore_index=True)
    if set(np.unique(y)) <= {0, 1}:
        ya = np.concatenate([y, 1 - y])
    else:
        ya = np.concatenate([y, -y])
    return Xa, ya


# ------------------------------------------------------------------ models
class WinModel:
    """Wraps a classifier; predictions are symmetrized over both orientations."""
    def __init__(self, est, cols):
        self.est, self.cols = est, cols

    def fit(self, d):
        X, y = augment(d[self.cols], d.home_win.values.astype(int))
        self.est.fit(X, y)
        return self

    def predict_proba(self, d):
        X = d[self.cols]
        p = self.est.predict_proba(X)[:, 1]
        pf = self.est.predict_proba(flip(X))[:, 1]
        return (p + (1 - pf)) / 2


class MarginModel:
    """Regressor on margin; win prob = P(N(pred, sigma) > 0). sigma comes from
    residuals on the last 2 training seasons, held out from an inner fit, so it
    reflects out-of-sample error rather than training error."""
    def __init__(self, make_est, cols):
        self.make_est, self.cols = make_est, cols

    def _fit(self, d):
        X, y = augment(d[self.cols], d.margin.values.astype(float))
        return self.make_est().fit(X, y)

    def _pred(self, est, d):
        X = d[self.cols]
        return (est.predict(X) - est.predict(flip(X))) / 2

    def fit(self, d):
        last = sorted(d.season.unique())[-2:]
        inner = self._fit(d[~d.season.isin(last)])
        hold = d[d.season.isin(last)]
        self.sigma = float(np.std(hold.margin - self._pred(inner, hold)))
        self.est = self._fit(d)
        return self

    def predict_margin(self, d):
        return self._pred(self.est, d)

    def predict_proba(self, d):
        return norm.cdf(self.predict_margin(d) / self.sigma)


def logit(C):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000))


def hgb_clf(lr, depth, iters, leaf):
    return HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth, max_iter=iters,
                                          min_samples_leaf=leaf, l2_regularization=1.0, random_state=0)


def ridge(alpha):
    return lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                 Ridge(alpha=alpha))


def hgb_reg(lr, depth, iters, leaf):
    return lambda: HistGradientBoostingRegressor(learning_rate=lr, max_depth=depth, max_iter=iters,
                                                 min_samples_leaf=leaf, l2_regularization=1.0, random_state=0)


# Hand-picked compact set (chosen from football knowledge, before seeing test data):
# team strength, QB quality/availability, rest, schedule strength.
CURATED = ["diff_elo", "diff_ewm_pt_diff", "diff_season_avg_pt_diff", "diff_prev_season_pt_diff_pg",
           "diff_qb_last8_anya", "diff_qb_career_anya", "diff_ewm_off_anya", "diff_ewm_def_anya_allowed",
           "diff_season_off_anya", "diff_season_def_anya_allowed", "diff_rest_days", "diff_off_bye",
           "diff_inj_exp_qb_out_or_doubtful", "diff_qb_changed_last_game", "diff_qb_career_starts",
           "diff_season_to_margin_pg", "diff_coach_new_this_season", "diff_season_avg_opp_elo", "diff_win_pct",
           "diff_div_rank", "home_adv", "is_playoff"]


def candidates(cols):
    """name -> (kind, factory). kind 'win' = classifier, 'margin' = regressor."""
    elo = ["diff_elo", "home_adv"]
    c = {"elo_logit": ("win", lambda: WinModel(logit(1.0), elo)),
         "elo_margin": ("margin", lambda: MarginModel(ridge(1.0), elo))}
    for C in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]:
        c[f"logit_C{C}"] = ("win", lambda C=C: WinModel(logit(C), cols))
    for C in [0.001, 0.003, 0.01]:
        c[f"curated_logit_C{C}"] = ("win", lambda C=C: WinModel(logit(C), CURATED))
    for a in [300, 1000, 3000]:
        c[f"curated_ridge_a{a}"] = ("margin", lambda a=a: MarginModel(ridge(a), CURATED))
    for a in [30, 100, 300, 1000, 3000, 10000]:
        c[f"ridge_a{a}"] = ("margin", lambda a=a: MarginModel(ridge(a), cols))
    for lr, depth, iters in [(0.03, 2, 300), (0.03, 3, 300), (0.02, 3, 600), (0.05, 2, 200)]:
        tag = f"lr{lr}_d{depth}_n{iters}"
        c[f"hgb_clf_{tag}"] = ("win", lambda lr=lr, d=depth, n=iters: WinModel(hgb_clf(lr, d, n, 80), cols))
        c[f"hgb_reg_{tag}"] = ("margin", lambda lr=lr, d=depth, n=iters: MarginModel(hgb_reg(lr, d, n, 80), cols))
    return c


# ------------------------------------------------------------------ metrics
def win_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": len(y), "accuracy": float(np.mean((p > 0.5) == (y == 1))),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2))}


def margin_metrics(m, pred):
    return {"mae": float(np.mean(np.abs(m - pred))), "rmse": float(np.sqrt(np.mean((m - pred) ** 2)))}
