"""Experiment: attention-based models vs the v1 ensemble.
Walk-forward validation 2015-2022 only (same protocol as 01_validate.py). Test seasons are never loaded.

A) FT-Transformer on the v1 feature set: every feature becomes a token and attends
   to every other feature, so the model can learn interactions between features.
B) History attention: for each team, a transformer reads its last K games (results +
   box score + opponent strength + how long ago), strictly before the game date,
   and learns which past games matter instead of relying on fixed "last 3" /
   "season" averages. Combined with a small set of pre-game facts (Elo, QB, rest,
   injuries).

Both are multi-task (win + margin), trained on both home/away orientations, and
symmetrized at prediction time like v1. Each fold holds out its last training season
for early stopping, and results are averaged over 3 seeds.
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm

warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, os.cpu_count()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, feature_cols, win_metrics, margin_metrics, VAL_SEASONS, TEST_SEASONS, PROC

MODE = sys.argv[1] if len(sys.argv) > 1 else "ft"
SEEDS = [0, 1, 2]
K = 16                      # games of history per team
MARGIN_SCALE = 14.0
FR = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

d = load(include_test=False)
assert not d.season.isin(TEST_SEASONS).any()
FEATS = feature_cols(d)
DIFF = [c for c in FEATS if c.startswith("diff_")]


# ============================================================ shared training utils
def train_loop(model, make_batch, n_tr, n_va, epochs=60, lr=1e-3, wd=1e-4, bs=256, patience=8, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n_tr)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            logit, marg, yw, ym = make_batch("train", idx, model)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, yw) + \
                0.5 * nn.functional.mse_loss(marg, ym / MARGIN_SCALE)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            logit, marg, yw, ym = make_batch("val", np.arange(n_va), model)
            vl = nn.functional.binary_cross_entropy_with_logits(logit, yw).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, ep + 1


# ============================================================ A) FT-Transformer
class FTTransformer(nn.Module):
    def __init__(self, n_feat, dim=32, heads=4, layers=2, drop=0.2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_feat, dim))
        self.miss = nn.Parameter(torch.zeros(n_feat, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

    def forward(self, x, m):
        tok = x.unsqueeze(-1) * self.w + self.b
        tok = torch.where(m.unsqueeze(-1), self.miss.expand_as(tok), tok)
        h = self.enc(torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], 1))
        o = self.head(h[:, 0])
        return o[:, 0], o[:, 1]


def flip_df(X):
    Xf = X.copy()
    Xf[DIFF] = -Xf[DIFF]
    Xf["home_adv"] = -Xf["home_adv"]
    return Xf


def run_ft(tr, va):
    fit, es = tr[tr.season < tr.season.max()], tr[tr.season == tr.season.max()]
    mu, sd = fit[FEATS].mean(), fit[FEATS].std().replace(0, 1)
    sd[DIFF] = fit[DIFF].abs().mean().replace(0, 1)       # keep diffs zero-centred so flips stay exact
    mu[DIFF] = 0.0

    def tens(df):
        z = ((df[FEATS] - mu) / sd).clip(-6, 6)
        return torch.tensor(z.fillna(0).values, dtype=torch.float32), torch.tensor(z.isna().values)

    def aug(df):
        X = pd.concat([df[FEATS], flip_df(df[FEATS])], ignore_index=True)
        yw = np.r_[df.home_win.values, 1 - df.home_win.values]
        ym = np.r_[df.margin.values, -df.margin.values]
        x, m = tens(X)
        return x, m, torch.tensor(yw, dtype=torch.float32), torch.tensor(ym, dtype=torch.float32)

    D = {"train": aug(fit), "val": aug(es)}

    def mb(split, idx, model):
        x, m, yw, ym = D[split]
        idx = torch.as_tensor(idx)
        lo, ma = model(x[idx], m[idx])
        return lo, ma, yw[idx], ym[idx]

    xv, mv = tens(va[FEATS])
    xf, mf = tens(flip_df(va[FEATS]))
    ps, pms = [], []
    for s in SEEDS:
        model, ep = train_loop(FTTransformer(len(FEATS)), mb, len(D["train"][0]), len(D["val"][0]), seed=s)
        model.eval()
        with torch.no_grad():
            l1, m1 = model(xv, mv)
            l2, m2 = model(xf, mf)
        p = (torch.sigmoid(l1) + 1 - torch.sigmoid(l2)).numpy() / 2
        ps.append(p)
        pms.append(((m1 - m2) / 2).numpy() * MARGIN_SCALE)
    return np.mean(ps, 0), np.mean(pms, 0)


# ============================================================ B) history attention
def build_history():
    """Per team-game history tensors. Every history row used for a game on date D
    comes from a game strictly before D (asserted)."""
    g = pd.read_parquet(f"{PROC}/nfl_games_1999_2024.parquet")
    g["gameday"] = pd.to_datetime(g.gameday)
    tf = pd.read_parquet(f"{PROC}/nfl_team_features_pregame_1999_2024.parquet")
    tf["gameday"] = pd.to_datetime(tf.gameday)
    ps = pd.read_parquet(f"{PROC}/nfl_player_offense_stats_weekly_1999_2024.parquet")
    ps["team"] = ps.team.replace(FR)
    for c in ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]:
        ps[c] = ps[c].fillna(0)
    ps["fl"] = ps.sack_fumbles_lost + ps.rushing_fumbles_lost + ps.receiving_fumbles_lost
    bx = ps.groupby(["game_id", "team"]).agg(py=("passing_yards", "sum"), sy=("sack_yards", "sum"),
                                              td=("passing_tds", "sum"), it=("interceptions", "sum"),
                                              att=("attempts", "sum"), sk=("sacks", "sum"),
                                              ry=("rushing_yards", "sum"), car=("carries", "sum"),
                                              fl=("fl", "sum")).reset_index()
    bx["anya"] = (bx.py - bx.sy + 20 * bx.td - 45 * bx.it) / (bx.att + bx.sk).replace(0, np.nan)
    bx["ypc"] = bx.ry / bx.car.replace(0, np.nan)
    bx["to"] = bx.it + bx.fl
    # team-game table: tf's `team` column is already the franchise code
    t = tf[["game_id", "season", "gameday", "team", "opp", "is_home", "elo"]].copy()
    sc = pd.concat([g[["game_id", "home_score", "away_score"]].assign(is_home=1)
                    .rename(columns={"home_score": "pf", "away_score": "pa"}),
                    g[["game_id", "away_score", "home_score"]].assign(is_home=0)
                    .rename(columns={"away_score": "pf", "home_score": "pa"})])
    t = t.merge(sc, on=["game_id", "is_home"])
    t = t.merge(bx[["game_id", "team", "anya", "ypc", "to"]], on=["game_id", "team"], how="left")
    t = t.merge(bx[["game_id", "team", "anya", "ypc", "to"]].rename(
        columns={"team": "opp", "anya": "d_anya", "ypc": "d_ypc", "to": "takeaways"}), on=["game_id", "opp"], how="left")
    opp_elo = t[["game_id", "team", "elo"]].rename(columns={"team": "opp", "elo": "opp_elo"})
    t = t.merge(opp_elo, on=["game_id", "opp"])
    t = t.sort_values(["team", "gameday"]).reset_index(drop=True)
    # per-game "result" vector (only ever read for PAST games)
    R = pd.DataFrame({
        "pt_diff": (t.pf - t.pa) / 14, "pf": (t.pf - 21) / 10, "pa": (t.pa - 21) / 10,
        "win": (t.pf > t.pa).astype(float) - 0.5,
        "anya": (t.anya.fillna(5.5) - 5.5) / 2, "d_anya": (t.d_anya.fillna(5.5) - 5.5) / 2,
        "ypc": (t.ypc.fillna(4.2) - 4.2), "d_ypc": (t.d_ypc.fillna(4.2) - 4.2),
        "to_margin": (t.takeaways.fillna(1.5) - t.to.fillna(1.5)) / 2,
        "opp_elo": (t.opp_elo - 1500) / 100, "was_home": t.is_home - 0.5,
        # margin adjusted for opponent strength (points above what Elo expected)
        "adj_diff": ((t.pf - t.pa) - (t.elo - t.opp_elo) / 25) / 14,
    }).values.astype(np.float32)
    n_r = R.shape[1] + 2  # + days_ago, same_season
    H = np.zeros((len(t), K, n_r), np.float32)
    M = np.ones((len(t), K), bool)            # True = padding
    team = t.team.values
    day = t.gameday.values.astype("datetime64[D]").astype(np.int64)
    ssn = t.season.values
    start = 0
    for i in range(len(t)):
        if i == 0 or team[i] != team[i - 1]:
            start = i
        lo = max(start, i - K)
        past = np.arange(lo, i)               # rows strictly before i for this team
        if len(past):
            assert (day[past] < day[i]).all(), "history leak"
            k = len(past)
            H[i, K - k:, :R.shape[1]] = R[past]
            H[i, K - k:, R.shape[1]] = np.log1p(day[i] - day[past]) / 5
            H[i, K - k:, R.shape[1] + 1] = (ssn[past] == ssn[i]).astype(np.float32)
            M[i, K - k:] = False
    key = pd.Series(np.arange(len(t)), index=pd.MultiIndex.from_frame(t[["game_id", "team"]]))
    return t, H, M, key


STATIC = ["elo", "qb_last8_anya", "qb_career_anya", "qb_career_starts", "rest_days", "off_bye",
          "inj_exp_qb_out_or_doubtful", "qb_changed_last_game", "qb_new_vs_last_season",
          "coach_new_this_season", "prev_season_pt_diff_pg"]
CTX_B = ["is_playoff", "div_game", "is_indoor", "is_primetime"]


class HistoryAttn(nn.Module):
    def __init__(self, n_r, n_s, n_c, dim=32, heads=4, layers=1, drop=0.2):
        super().__init__()
        self.inp = nn.Linear(n_r, dim)
        self.pos = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.stat = nn.Sequential(nn.Linear(n_s * 2, dim), nn.GELU())
        self.team = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(drop))
        self.out = nn.Sequential(nn.Linear(dim + n_c + 1, dim), nn.GELU(), nn.Dropout(drop), nn.Linear(dim, 2))

    def team_vec(self, h, m, s, ms):
        z = self.inp(h) + self.pos
        empty = m.all(1)
        m = m.clone()
        m[empty, -1] = False                    # teams with no history: attend to one zero row
        z = torch.cat([self.cls.expand(h.shape[0], -1, -1), z], 1)
        m = torch.cat([torch.zeros(h.shape[0], 1, dtype=torch.bool), m], 1)
        e = self.enc(z, src_key_padding_mask=m)[:, 0]
        st = self.stat(torch.cat([s, ms.float()], 1))
        return self.team(torch.cat([e, st], 1))

    def forward(self, hh, mh, sh, msh, ha, ma, sa, msa, ctx, adv):
        # antisymmetric by construction: f(home, away) = -f(away, home) apart from home_adv
        vh, va = self.team_vec(hh, mh, sh, msh), self.team_vec(ha, ma, sa, msa)
        o1 = self.out(torch.cat([vh - va, ctx, adv.unsqueeze(1)], 1))
        o2 = self.out(torch.cat([va - vh, ctx, -adv.unsqueeze(1)], 1))
        o = (o1 - o2) / 2
        return o[:, 0], o[:, 1]


def run_hist(tr, va, HIST):
    t, H, M, key, S = HIST
    fit, es = tr[tr.season < tr.season.max()], tr[tr.season == tr.season.max()]
    smu = S.loc[S.season.isin(fit.season.unique()), STATIC].mean()
    ssd = S.loc[S.season.isin(fit.season.unique()), STATIC].std().replace(0, 1)

    def pack(df):
        out = []
        for side, team_col in [("h", "home_team"), ("a", "away_team")]:
            idx = key.loc[list(zip(df.game_id, df[team_col].replace(FR)))].values
            st = ((S.iloc[idx][STATIC] - smu) / ssd).clip(-6, 6)
            out += [torch.tensor(H[idx]), torch.tensor(M[idx]),
                    torch.tensor(st.fillna(0).values, dtype=torch.float32), torch.tensor(st.isna().values)]
        c = torch.tensor(df[CTX_B].fillna(0).values, dtype=torch.float32)
        adv = torch.tensor(df.home_adv.values, dtype=torch.float32)
        yw = torch.tensor(df.home_win.values, dtype=torch.float32)
        ym = torch.tensor(df.margin.values, dtype=torch.float32)
        return out + [c, adv], yw, ym

    D = {"train": pack(fit), "val": pack(es)}

    def mb(split, idx, model):
        X, yw, ym = D[split]
        idx = torch.as_tensor(idx)
        lo, ma = model(*[x[idx] for x in X])
        return lo, ma, yw[idx], ym[idx]

    Xv, _, _ = pack(va)
    ps, pms = [], []
    for s in SEEDS:
        model = HistoryAttn(H.shape[2], len(STATIC), len(CTX_B))
        model, ep = train_loop(model, mb, len(fit), len(es), seed=s)
        model.eval()
        with torch.no_grad():
            lo, ma = model(*Xv)
        ps.append(torch.sigmoid(lo).numpy())
        pms.append(ma.numpy() * MARGIN_SCALE)
    return np.mean(ps, 0), np.mean(pms, 0)


# ============================================================ main
if __name__ == "__main__":
    HIST = None
    if MODE == "hist":
        t, H, M, key = build_history()
        S = t[["game_id", "team", "season"]].merge(
            pd.read_parquet(f"{PROC}/nfl_team_features_pregame_1999_2024.parquet")[["game_id", "team"] + STATIC],
            on=["game_id", "team"], how="left")
        HIST = (t, H, M, key, S)
    os.makedirs("results/attention", exist_ok=True)
    oof = []
    t0 = time.time()
    for s in VAL_SEASONS:
        tr, va = d[d.season < s], d[d.season == s]
        p, pm = run_ft(tr, va) if MODE == "ft" else run_hist(tr, va, HIST)
        o = va[["game_id", "season", "home_win", "margin"]].copy()
        o["p"], o["pred_margin"] = p, pm
        oof.append(o)
        print(s, {k: round(v, 4) for k, v in win_metrics(o.home_win.values, p).items()},
              f"{time.time() - t0:.0f}s", flush=True)
    oof = pd.concat(oof)
    oof.to_parquet(f"results/attention/oof_{MODE}.parquet", index=False)
    r = {**win_metrics(oof.home_win.values, oof.p.values), **margin_metrics(oof.margin.values, oof.pred_margin.values)}
    print("OVERALL", MODE, {k: round(v, 4) for k, v in r.items()})
