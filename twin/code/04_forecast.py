"""
Stage 3: day-ahead (24 h) multi-task probabilistic forecasting of PV, wind and load.
Usage: python 04_forecast.py <model> [seed]
Models: persistence, ridge, xgb, xgbq, lstm, gru, cnnlstm, transformer,
        proposed, abl_single, abl_noattn, abl_novsn, abl_nofs, abl_nonwp, abl_raw
"""
import sys, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, xgboost as xgb
from sklearn.linear_model import Ridge
from common import *

import os; VERBOSE = os.environ.get('VERBOSE') == '1'; NO_ANCHOR = sys.argv[1] == 'abl_noanchor'
_SKIP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skip.txt')
_skip = {tuple(l.split()) for l in open(_SKIP_FILE) if l.strip()} if os.path.exists(_SKIP_FILE) else set()
if (sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '0') in _skip: sys.exit(0)
MODEL = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)
QS = [0.1, 0.5, 0.9]
src = "data/twin_rawfilled.csv" if MODEL == "abl_raw" else "data/twin_cleaned.csv"
df, truth, det, base, cs = load_frames(src)
tr, va, te = split_origins(df.time.values)
use_nwp = MODEL != "abl_nonwp"
PAST_COLS = PAST_CH + FUT_DET
FUT_COLS = FUT_DET + (FUT_NWP if use_nwp else [])
Xp_tr, Xf_tr, Y_tr, Yi_tr = build_arrays(df, truth, det, base, cs, tr, PAST_CH, use_nwp)
Xp_va, Xf_va, Y_va, Yi_va = build_arrays(df, truth, det, base, cs, va, PAST_CH, use_nwp)
Xp_te, Xf_te, Y_te, Yi_te = build_arrays(df, truth, det, base, cs, te, PAST_CH, use_nwp)
night_te = Xf_te[:, :, FUT_COLS.index("cos_zen")] <= 0.0

# ------------------------------ factor masks --------------------------------
fs = json.load(open("results/factor_selection.json"))
G2C = {"ghi": ["ghi"], "t_amb": ["t_amb"], "t_mod": ["t_mod"], "rh": ["rh"], "ws": ["ws"], "wd": ["wd_s", "wd_c"], "ps": ["ps"],
       "soiling": ["soiling_pct"], "p_pv": ["p_pv"], "p_wind": ["p_wind"], "p_load": ["p_load"], "q_load": ["q_load"],
       "solar_geom": ["cs", "cos_zen"], "calendar": ["hs", "hc", "ds", "dc"], "day_type": ["sun", "sat", "hol"],
       "nwp_ghi": ["nwp_ghi"], "nwp_t": ["nwp_t"], "nwp_ws": ["nwp_ws"]}
def masks(all_on=False):
    mp = np.zeros((3, len(PAST_COLS)), np.float32); mf = np.zeros((3, len(FUT_COLS)), np.float32)
    for k, y in enumerate(TARGETS):
        sel = list(G2C) if all_on else fs[y]["selected"] + [y]          # own history always available
        sel = sel + ["solar_geom", "calendar"]                           # positional context always on
        for g in sel:
            for c in G2C[g]:
                if c in PAST_COLS: mp[k, PAST_COLS.index(c)] = 1
                if c in FUT_COLS: mf[k, FUT_COLS.index(c)] = 1
    return torch.tensor(mp), torch.tensor(mf)

# ------------------------------ scaling -------------------------------------
mu_p, sd_p = Xp_tr.reshape(-1, Xp_tr.shape[-1]).mean(0), Xp_tr.reshape(-1, Xp_tr.shape[-1]).std(0) + 1e-6
mu_f, sd_f = Xf_tr.reshape(-1, Xf_tr.shape[-1]).mean(0), Xf_tr.reshape(-1, Xf_tr.shape[-1]).std(0) + 1e-6
ym = Yi_tr.reshape(-1, 3).mean(0).astype(np.float32); ysc = Yi_tr.reshape(-1, 3).std(0).astype(np.float32)
nz = lambda a, m, s: ((a - m) / s).astype(np.float32)
T = lambda a: torch.tensor(a)

night_va = Xf_va[:, :, FUT_COLS.index("cos_zen")] <= 0.0
def finish_val(pred_q):
    pred_q = np.maximum(pred_q, 0.0); pred_q[:, :, 0][night_va] = 0.0
    pred_q[:, :, 0] = np.minimum(pred_q[:, :, 0], 150.0); pred_q[:, :, 1] = np.minimum(pred_q[:, :, 1], 100.0)
    return np.sort(pred_q, -1)

def finish(pred_q):
    """pred_q: (N,H,3,Q) in kW -> physical post-processing shared by all models"""
    pred_q = np.maximum(pred_q, 0.0)
    pred_q[:, :, 0][night_te] = 0.0
    pred_q[:, :, 0] = np.minimum(pred_q[:, :, 0], 150.0); pred_q[:, :, 1] = np.minimum(pred_q[:, :, 1], 100.0)
    return np.sort(pred_q, -1)

# ------------------------------ networks ------------------------------------
class Proposed(nn.Module):
    """MFR-MTNet: multi-factor routed, multi-task CNN-BiLSTM encoder with horizon-wise cross-attention
    and quantile heads. Task-specific factor routing = hard mask from Stage 2 x soft context-dependent gates."""
    def __init__(s, Fp, Ff, mp, mf, d=64, mt=True, attn=True, vsn=True):
        super().__init__(); s.mt, s.attn, s.vsn = mt, attn, vsn
        s.register_buffer("mp", mp); s.register_buffer("mf", mf); K = 3
        s.gp = nn.ModuleList([nn.Sequential(nn.Linear(Fp, 32), nn.ReLU(), nn.Linear(32, Fp)) for _ in range(K)])
        s.gf = nn.ModuleList([nn.Sequential(nn.Linear(Ff, 32), nn.ReLU(), nn.Linear(32, Ff)) for _ in range(K)])
        s.inp = nn.Linear(Fp, d); s.task_emb = nn.Parameter(torch.randn(K, d)*0.1)
        s.conv = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.GELU(), nn.Conv1d(d, d, 3, padding=2, dilation=2), nn.GELU())
        s.lstm = nn.LSTM(d, d//2, batch_first=True, bidirectional=True)
        s.fut = nn.Linear(Ff, d); s.hemb = nn.Parameter(torch.randn(H, d)*0.1)
        s.mha = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        s.heads = nn.ModuleList([nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, len(QS))) for _ in range(K)])
        s.log_var = nn.Parameter(torch.zeros(K))
        # linear skip branch (seasonal-lag + covariate linear path); NN branch models the nonlinear residual
        s.lin = nn.ModuleList([nn.Linear(24*Fp + H*Ff, H) for _ in range(K)])
        s.drop = nn.Dropout(0.2)
    def lin_in(s, xp, xf, k):
        return torch.cat([(xp*s.mp[k])[:, -24:].flatten(1), (xf*s.mf[k]).flatten(1)], 1)
    def anchor(s, xp, xf, y, alpha=10.0):
        """closed-form ridge fit of the linear path on the training set, then frozen"""
        from sklearn.linear_model import Ridge
        for k in range(3):
            Z = s.lin_in(xp, xf, k).numpy(); r = Ridge(alpha=alpha).fit(Z, y[:, :, k].numpy())
            s.lin[k].weight.data = torch.tensor(r.coef_, dtype=torch.float32); s.lin[k].bias.data = torch.tensor(r.intercept_, dtype=torch.float32)
            for p_ in s.lin[k].parameters(): p_.requires_grad_(False)
    def gate(s, x, g, m):
        if not s.vsn: return x*m
        logits = g(x).masked_fill(m == 0, -1e4); w = torch.softmax(logits, -1)
        return x*w*m.sum(), w
    def forward(s, xp, xf, return_w=False):
        outs, ws = [], []
        for k in range(3):
            gp = s.gate(xp, s.gp[k], s.mp[k]); gf = s.gate(xf, s.gf[k], s.mf[k])
            if s.vsn: (xpk, wp), (xfk, wf) = gp, gf; ws.append((wp.mean((0, 1)), wf.mean((0, 1))))
            else: xpk, xfk = gp, gf
            h = s.inp(xpk) + s.task_emb[k]
            h = h + s.conv(h.transpose(1, 2)).transpose(1, 2)
            enc, (hn, _) = s.lstm(h)
            q = s.fut(xfk) + s.hemb
            if s.attn: ctx, _ = s.mha(q, enc, enc)
            else: ctx = enc[:, -1:, :].expand(-1, H, -1)
            skip = s.lin[k](s.lin_in(xp, xf, k)).unsqueeze(-1)
            outs.append(skip + s.heads[k](s.drop(torch.cat([q, ctx], -1))))
        out = torch.stack(outs, 2)                                     # (B,H,3,Q)
        return (out, ws) if return_w else out

class RNNBase(nn.Module):
    def __init__(s, Fp, Ff, cell="lstm", conv=False, d=64):
        super().__init__(); s.conv = nn.Sequential(nn.Conv1d(Fp, d, 3, padding=1), nn.ReLU()) if conv else None
        R = nn.LSTM if cell == "lstm" else nn.GRU; s.rnn = R(d if conv else Fp, d, num_layers=2, batch_first=True, dropout=0.1)
        s.head = nn.Sequential(nn.Linear(d + H*Ff, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, H*3))
    def forward(s, xp, xf):
        if s.conv is not None: xp = s.conv(xp.transpose(1, 2)).transpose(1, 2)
        o, _ = s.rnn(xp); z = torch.cat([o[:, -1], xf.flatten(1)], -1)
        return s.head(z).view(-1, H, 3, 1)

class TransformerBase(nn.Module):
    def __init__(s, Fp, Ff, d=64):
        super().__init__(); s.pp, s.pf = nn.Linear(Fp, d), nn.Linear(Ff, d)
        s.pos = nn.Parameter(torch.randn(L_IN + H, d)*0.05)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 128, 0.1, batch_first=True), 2)
        s.out = nn.Linear(d, 3)
    def forward(s, xp, xf):
        z = torch.cat([s.pp(xp), s.pf(xf)], 1) + s.pos
        return s.out(s.enc(z)[:, L_IN:]).view(-1, H, 3, 1)

# ------------------------------ state-of-the-art baselines -------------------
TIDX = [PAST_COLS.index(t) for t in TARGETS]

class DLinear(nn.Module):
    """Zeng et al. 2023: series decomposition + per-channel linear maps; +linear head on future covariates"""
    def __init__(s, Fp, Ff, k=25):
        super().__init__(); s.k = k
        s.lt = nn.ModuleList([nn.Linear(L_IN, H) for _ in range(3)]); s.ls = nn.ModuleList([nn.Linear(L_IN, H) for _ in range(3)])
        s.fx = nn.Linear(Ff, 3)
    def forward(s, xp, xf):
        y = xp[:, :, TIDX]; pad = s.k//2
        tr = nn.functional.avg_pool1d(nn.functional.pad(y.transpose(1, 2), (pad, pad), mode="replicate"), s.k, 1).transpose(1, 2)
        se = y - tr
        out = torch.stack([s.lt[c](tr[:, :, c]) + s.ls[c](se[:, :, c]) for c in range(3)], -1)
        return (out + s.fx(xf)).unsqueeze(-1)

class NHITS(nn.Module):
    """Challu et al. 2023: stacks of MLP blocks with multi-rate input pooling and hierarchical interpolation"""
    def __init__(s, Fp, Ff, d=256):
        super().__init__(); s.pools, s.ncoef = [8, 4, 1], [4, 8, 24]
        s.blocks = nn.ModuleList()
        for p, nc in zip(s.pools, s.ncoef):
            nin = 3*(L_IN//p) + H*Ff
            s.blocks.append(nn.ModuleDict({"mlp": nn.Sequential(nn.Linear(nin, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU()),
                                           "back": nn.Linear(d, 3*L_IN), "fore": nn.Linear(d, 3*nc)}))
    def forward(s, xp, xf):
        res = xp[:, :, TIDX]; fc = 0
        for b, p, nc in zip(s.blocks, s.pools, s.ncoef):
            z = nn.functional.max_pool1d(res.transpose(1, 2), p, p).flatten(1) if p > 1 else res.transpose(1, 2).flatten(1)
            h = b["mlp"](torch.cat([z, xf.flatten(1)], 1))
            res = res - b["back"](h).view(-1, 3, L_IN).transpose(1, 2)
            coef = b["fore"](h).view(-1, 3, nc)
            fc = fc + nn.functional.interpolate(coef, size=H, mode="linear", align_corners=True).transpose(1, 2)
        return fc.unsqueeze(-1)

class TiDE(nn.Module):
    """Das et al. 2023: dense encoder-decoder with covariate projection and temporal decoder"""
    def __init__(s, Fp, Ff, d=256, pr=4):
        super().__init__()
        res = lambda i, o: nn.ModuleDict({"a": nn.Sequential(nn.Linear(i, d), nn.ReLU(), nn.Linear(d, o), nn.Dropout(0.1)), "s": nn.Linear(i, o), "n": nn.LayerNorm(o)})
        s.res = res; s.proj = res(Ff, pr)
        nin = L_IN*Fp + H*pr
        s.enc = nn.ModuleList([res(nin, d), res(d, d)]); s.dec = nn.ModuleList([res(d, d), res(d, H*16)])
        s.tdec = res(16 + pr, 3); s.glob = nn.Linear(L_IN, H)
    def _r(s, m, x): return m["n"](m["a"](x) + m["s"](x))
    def forward(s, xp, xf):
        pf = s._r(s.proj, xf)
        z = torch.cat([xp.flatten(1), pf.flatten(1)], 1)
        for m in s.enc: z = s._r(m, z)
        for m in s.dec: z = s._r(m, z)
        out = s._r(s.tdec, torch.cat([z.view(-1, H, 16), pf], -1))
        out = out + s.glob(xp[:, :, TIDX].transpose(1, 2)).transpose(1, 2)
        return out.unsqueeze(-1)

class PatchTST(nn.Module):
    """Nie et al. 2023: channel-independent patching + Transformer encoder; +linear head on future covariates"""
    def __init__(s, Fp, Ff, pl=12, st=6, d=64):
        super().__init__(); s.pl, s.st = pl, st; n = (L_IN - pl)//st + 1
        s.emb = nn.Linear(pl, d); s.pos = nn.Parameter(torch.randn(n, d)*0.05)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 128, 0.2, batch_first=True), 3)
        s.head = nn.Linear(n*d, H); s.fx = nn.Linear(Ff, 3)
    def forward(s, xp, xf):
        y = xp[:, :, TIDX].transpose(1, 2)                       # (B,3,L)
        B = y.shape[0]; p = y.unfold(-1, s.pl, s.st)              # (B,3,n,pl)
        z = s.enc(s.emb(p.reshape(B*3, -1, s.pl)) + s.pos)
        out = s.head(z.flatten(1)).view(B, 3, H).transpose(1, 2)
        return (out + s.fx(xf)).unsqueeze(-1)

class GRN(nn.Module):
    def __init__(s, i, d, o=None):
        super().__init__(); o = o or d
        s.l1, s.l2, s.g = nn.Linear(i, d), nn.Linear(d, 2*o), nn.GLU(); s.skip = nn.Linear(i, o) if i != o else nn.Identity(); s.n = nn.LayerNorm(o)
    def forward(s, x): return s.n(s.skip(x) + s.g(s.l2(nn.functional.elu(s.l1(x)))))

class VSN(nn.Module):
    """variable selection network: per-variable linear embeddings (vectorised) + shared GRN, softmax selection weights"""
    def __init__(s, F, d):
        super().__init__(); s.W = nn.Parameter(torch.randn(F, d)*0.1); s.b = nn.Parameter(torch.zeros(F, d))
        s.w = GRN(F, d, F); s.g = GRN(d, d)
    def forward(s, x):
        w = torch.softmax(s.w(x), -1).unsqueeze(-1)
        e = s.g(x.unsqueeze(-1)*s.W + s.b)
        return (w*e).sum(-2)

class TFT(nn.Module):
    """Lim et al. 2021 (compact): variable selection, LSTM encoder-decoder, gated skip, interpretable attention, quantile outputs"""
    def __init__(s, Fp, Ff, d=48):
        super().__init__(); s.vp, s.vf = VSN(Fp, d), VSN(Ff, d)
        s.enc, s.dec = nn.LSTM(d, d, batch_first=True), nn.LSTM(d, d, batch_first=True)
        s.gate1 = nn.Sequential(nn.Linear(d, 2*d), nn.GLU()); s.n1 = nn.LayerNorm(d)
        s.enrich = GRN(d, d); s.att = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        s.gate2 = nn.Sequential(nn.Linear(d, 2*d), nn.GLU()); s.n2 = nn.LayerNorm(d); s.pw = GRN(d, d)
        s.out = nn.Linear(d, 3*len(QS))
        s.register_buffer("mask", torch.triu(torch.ones(L_IN + H, L_IN + H, dtype=torch.bool), 1))
    def forward(s, xp, xf):
        ep, ef = s.vp(xp), s.vf(xf)
        he, st = s.enc(ep); hd, _ = s.dec(ef, st)
        h = torch.cat([he, hd], 1); x = torch.cat([ep, ef], 1)
        h = s.n1(x + s.gate1(h)); e = s.enrich(h)
        a, _ = s.att(e, e, e, attn_mask=s.mask)
        z = s.n2(e + s.gate2(a)); z = s.pw(z)[:, L_IN:]
        return s.out(z).view(-1, H, 3, len(QS))

def pinball(pred, y, qs):
    q = torch.tensor(qs).view(1, 1, 1, -1); e = y.unsqueeze(-1) - pred
    return torch.maximum(q*e, (q - 1)*e).mean((0, 1, 3))           # per task

def train_nn(net, quantile, task_w=None, epochs=int(os.environ.get("EPOCHS", 40)), patience=10, bs=256, lr=float(os.environ.get("LR", 2e-3))):
    A = lambda Xp, Xf: (T(nz(Xp, mu_p, sd_p)), T(nz(Xf, mu_f, sd_f)))
    xp, xf = A(Xp_tr, Xf_tr); yt = T((Yi_tr - ym)/ysc); vxp, vxf = A(Xp_va, Xf_va); vy = T((Yi_va - ym)/ysc)
    if isinstance(net, Proposed) and not NO_ANCHOR: net.anchor(xp, xf, yt)
    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad], lr=lr, weight_decay=float(os.environ.get("WD", 1e-4)))
    SCHED = os.environ.get("SCHED", "cos")
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.02) if SCHED == "cos"
           else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3))
    best, bstate, bad = np.full(3, 1e9), [None]*3, 0
    tw = torch.ones(3) if task_w is None else torch.tensor(task_w, dtype=torch.float32)
    active = (tw > 0).numpy()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(xp))
        for i in range(0, len(perm), bs):
            b = perm[i:i+bs]; p = net(xp[b], xf[b])
            if quantile:
                lk = pinball(p, yt[b], QS)
                loss = (torch.exp(-net.log_var)*lk + net.log_var).sum() if (getattr(net, "mt", False) and task_w is None) else (lk*tw).sum()
            else: loss = (((p[..., 0] - yt[b])**2).mean((0, 1))*tw).sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(vxp[i:i+1024], vxf[i:i+1024]) for i in range(0, len(vxp), 1024)])
            vk = (pinball(pv, vy, QS) if quantile else ((pv[..., 0] - vy)**2).mean((0, 1))).numpy()
        vk_rel = vk / np.where(best < 1e8, best, vk)
        sch.step() if SCHED == "cos" else sch.step(float((vk_rel*active).sum()))
        if VERBOSE: print(ep, "val", vk.round(4), "rmse_kW", (torch.sqrt(((pv[..., pv.shape[-1]//2] - vy)**2).mean((0, 1))).numpy()*ysc).round(2), flush=True)
        improved = False
        for k in range(3):                                   # task-wise checkpoint selection
            if active[k] and vk[k] < best[k] - 1e-6:
                best[k], bstate[k], improved = vk[k], {n_: v.clone() for n_, v in net.state_dict().items()}, True
        bad = 0 if improved else bad + 1
        if bad >= patience: break
    res = []
    for (XP, XF) in ((Xp_te, Xf_te), (Xp_va, Xf_va)):
        txp, txf = A(XP, XF); outs = []
        for k in range(3):
            if not active[k]: outs.append(None); continue
            net.load_state_dict(bstate[k]); net.eval()
            with torch.no_grad():
                outs.append(torch.cat([net(txp[i:i+1024], txf[i:i+1024]) for i in range(0, len(txp), 1024)]).numpy())
        ref = next(o for o in outs if o is not None); p = np.zeros_like(ref)
        for k in range(3):
            if outs[k] is not None: p[:, :, k] = outs[k][:, :, k]
        res.append(p*ysc[None, None, :, None] + ym[None, None, :, None])
    net.load_state_dict(bstate[int(np.argmax(active))])
    global PVAL; PVAL = res[1] if PVAL is None else np.where(np.isnan(PVAL), res[1], PVAL)
    return res[0], ep + 1

# ------------------------------ tabular helpers ------------------------------
def tabular(Xp, Xf):
    """direct multi-horizon design: one row per (origin, h)"""
    N = len(Xp); last = Xp[:, -1]; m24 = Xp[:, -24:].mean(1)
    rows = []
    for h in range(H):
        rows.append(np.concatenate([np.full((N, 1), h + 1), last, m24, Xp[:, -24 + h], Xf[:, h]], 1))
    return np.concatenate(rows, 0)

t0 = time.time(); epochs = None; PVAL = None
if MODEL == "persistence":
    P = df[TARGETS].values.astype(np.float32); fut = np.arange(1, H + 1)
    pq = P[te[:, None] + fut[None] - 24][..., None]
elif MODEL == "ridge":
    Xtr_ = np.concatenate([nz(Xp_tr, mu_p, sd_p)[:, -24:].reshape(len(tr), -1), nz(Xf_tr, mu_f, sd_f).reshape(len(tr), -1)], 1)
    Xte_ = np.concatenate([nz(Xp_te, mu_p, sd_p)[:, -24:].reshape(len(te), -1), nz(Xf_te, mu_f, sd_f).reshape(len(te), -1)], 1)
    r = Ridge(alpha=10.0).fit(Xtr_, Yi_tr.reshape(len(tr), -1)); pq = r.predict(Xte_).reshape(len(te), H, 3, 1)
elif MODEL == "lgbm":
    import lightgbm as lgb
    sub = tr[::2]; Xp_s, Xf_s, _, Yi_s = build_arrays(df, truth, det, base, cs, sub, PAST_CH, use_nwp)
    Xtr_, Xte_ = tabular(Xp_s, Xf_s), tabular(Xp_te, Xf_te); pq = np.zeros((len(te), H, 3, 1), np.float32)
    for k in range(3):
        y = np.concatenate([Yi_s[:, h, k] for h in range(H)])
        m = lgb.LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, n_jobs=2, verbose=-1, random_state=SEED).fit(Xtr_, y)
        pq[:, :, k, 0] = m.predict(Xte_).reshape(H, len(te)).T
elif MODEL in ("xgb", "xgbq", "xgbq_fs"):
    sub = tr[::2]; Xp_s, Xf_s, _, Yi_s = build_arrays(df, truth, det, base, cs, sub, PAST_CH, use_nwp)
    Xtr_, Xte_, Xva_ = tabular(Xp_s, Xf_s), tabular(Xp_te, Xf_te), tabular(Xp_va, Xf_va)
    nq = 1 if MODEL == "xgb" else 3
    pq = np.zeros((len(te), H, 3, nq), np.float32); PVAL = np.zeros((len(va), H, 3, nq), np.float32)
    mp_, mf_ = masks()
    for k in range(3):
        y = np.concatenate([Yi_s[:, h, k] for h in range(H)])
        cols = np.ones(Xtr_.shape[1], bool)
        if MODEL == "xgbq_fs":
            m_ = mp_[k].numpy().astype(bool); cols = np.concatenate([[True], m_, m_, m_[:len(PAST_COLS)], mf_[k].numpy().astype(bool)])
        if MODEL == "xgb":
            m = xgb.XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, verbosity=0, random_state=SEED)
        else:
            m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QS), n_estimators=600, max_depth=7,
                                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, verbosity=0, random_state=SEED)
        m.fit(Xtr_[:, cols], y)
        pq[:, :, k] = m.predict(Xte_[:, cols]).reshape(H, len(te), -1).transpose(1, 0, 2)
        PVAL[:, :, k] = m.predict(Xva_[:, cols]).reshape(H, len(va), -1).transpose(1, 0, 2)
else:
    Fp, Ff = Xp_tr.shape[-1], Xf_tr.shape[-1]
    if MODEL in ("lstm", "gru", "cnnlstm"):
        net = RNNBase(Fp, Ff, "gru" if MODEL == "gru" else "lstm", conv=MODEL == "cnnlstm"); pq, epochs = train_nn(net, False)
    elif MODEL == "transformer":
        net = TransformerBase(Fp, Ff); pq, epochs = train_nn(net, False)
    elif MODEL in ("dlinear", "nhits", "tide", "patchtst"):
        net = {"dlinear": DLinear, "nhits": NHITS, "tide": TiDE, "patchtst": PatchTST}[MODEL](Fp, Ff); pq, epochs = train_nn(net, False)
    elif MODEL == "tft":
        net = TFT(Fp, Ff); pq, epochs = train_nn(net, True)
    elif MODEL == "dbg_load":
        mp, mf = masks(); net = Proposed(Fp, Ff, mp, mf, mt=False); pq, epochs = train_nn(net, True, task_w=[0, 0, 1])
    elif MODEL == "abl_single":                        # three independent single-task networks
        mp, mf = masks(); pq = np.zeros((len(te), H, 3, 3), np.float32); epochs = []
        for k in range(3):
            net = Proposed(Fp, Ff, mp, mf, mt=False); w = [0, 0, 0]; w[k] = 1
            prev = PVAL; PVAL = None; p, e = train_nn(net, True, task_w=w); pq[:, :, k] = p[:, :, k]; epochs.append(e)
            if prev is not None: prev[:, :, k] = PVAL[:, :, k]; PVAL = prev
    else:
        mp, mf = masks(all_on=(MODEL == "abl_nofs"))  # covers proposed, abl_noattn, abl_novsn, abl_nofs, abl_nonwp, abl_raw, abl_noanchor
        net = Proposed(Fp, Ff, mp, mf, attn=MODEL != "abl_noattn", vsn=MODEL != "abl_novsn")
        pq, epochs = train_nn(net, True)
        if MODEL == "proposed" and SEED == 0:          # export learned factor weights (interpretability)
            A = lambda Xp, Xf: (T(nz(Xp, mu_p, sd_p)), T(nz(Xf, mu_f, sd_f)))
            with torch.no_grad():
                _, ws = net(*A(Xp_te[:2048], Xf_te[:2048]), return_w=True)
            json.dump({y: {"past": dict(zip(PAST_COLS, ws[k][0].tolist())), "future": dict(zip(FUT_COLS, ws[k][1].tolist()))}
                       for k, y in enumerate(TARGETS)}, open("results/gate_weights.json", "w"), indent=1)
            torch.save(net.state_dict(), "results/proposed_seed0.pt")
elapsed = time.time() - t0
pq = finish(np.asarray(pq, np.float32))
pv_ = finish_val(np.asarray(PVAL, np.float32)) if PVAL is not None else None
np.savez_compressed(f"results/pred_{MODEL}_s{SEED}.npz", pq=pq, y=Y_te, te=te, **({"pq_val": pv_, "y_val": Y_va, "va": va} if pv_ is not None else {}))
json.dump({"model": MODEL, "seed": SEED, "train_seconds": elapsed, "epochs": epochs,
           "params": int(sum(p.numel() for p in net.parameters())) if "net" in dir() else None},
          open(f"results/meta_{MODEL}_s{SEED}.json", "w"))
med = pq[..., pq.shape[-1]//2]
print(MODEL, SEED, "RMSE", np.sqrt(((med - Y_te)**2).mean((0, 1))).round(3), "time %.0fs" % elapsed, "epochs", epochs, flush=True)
