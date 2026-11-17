"""Experimentos en IBM AML (protocolo de 3.8).

Uso:
    python -m experiments.run_ibm HI-Small          # contraste confirmatorio
    python -m experiments.run_ibm LI-Small          # réplica
    python -m experiments.run_ibm HI-Small --full   # sensibilidad sin recorte
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from experiments.ibm_features import COUNTERS, IBM_DIR, SNAP, TXN, build_ibm, load_ibm
from experiments.run_experiments import best_threshold, metrics

SEEDS = (42, 43, 44)
N_BOOT = 1000
CONFIGS = {
    "B transaccional": TXN,
    "T1 B + contadores": TXN + COUNTERS,
    "T2 B + instantaneas": TXN + SNAP,
    "T3 B + contadores + instantaneas": TXN + COUNTERS + SNAP,
    "T4 solo topologicas": COUNTERS + SNAP,
}


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, max_samples=0.25,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)


def run_one(df: pd.DataFrame, cols: list[str], kind: str, seed: int):
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy()
    s = df["split"].to_numpy()
    tr, va, te = s == "train", s == "val", s == "test"
    m = make_model(kind, seed).fit(X[tr], y[tr])
    thr = best_threshold(y[va], m.predict_proba(X[va])[:, 1])
    m = make_model(kind, seed).fit(X[tr | va], y[tr | va])
    p = m.predict_proba(X[te])[:, 1]
    return y[te], p, thr, getattr(m, "feature_importances_", None)


class _AP:
    """PR-AUC (average precision) ponderada sobre un orden fijo, con empates como sklearn."""

    def __init__(self, y: np.ndarray, p: np.ndarray):
        o = np.argsort(-p, kind="stable")
        self.o, self.y = o, y[o].astype(np.float64)
        ps = p[o]
        last = np.r_[ps[1:] != ps[:-1], True]            # fin de cada grupo de empate
        self.end = np.flatnonzero(last)
        self.grp = np.repeat(np.arange(len(self.end)), np.diff(np.r_[-1, self.end]))

    def __call__(self, w: np.ndarray) -> float:
        w = w[self.o]
        tp, n = np.cumsum(w * self.y)[self.end], np.cumsum(w)[self.end]
        prec = tp / np.maximum(n, 1e-12)
        dtp = np.diff(np.r_[0.0, tp])
        return float((dtp * prec).sum() / max(tp[-1], 1e-12))


def _f1_w(y, yhat, w) -> float:
    tp = (w * (y & yhat)).sum(); fp = (w * (~y & yhat)).sum(); fn = (w * (y & ~yhat)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1e-12))


def paired_bootstrap(y, p_a, thr_a, p_b, thr_b, n=N_BOOT, seed=0) -> dict:
    """IC 95 % de ΔF1 y ΔPR-AUC (b - a) con remuestreo pareado (pesos multinomiales)."""
    rng = np.random.default_rng(seed)
    yb = y.astype(bool)
    ha, hb = p_a >= thr_a, p_b >= thr_b
    ap_a, ap_b = _AP(y, p_a), _AP(y, p_b)
    d_f1, d_ap = [], []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(np.float64)
        d_f1.append(_f1_w(yb, hb, w) - _f1_w(yb, ha, w))
        d_ap.append(ap_b(w) - ap_a(w))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"delta_f1_ci95": q(d_f1), "delta_pr_auc_ci95": q(d_ap),
            "p_delta_f1<=0": float(np.mean(np.array(d_f1) <= 0)),
            "p_delta_pr_auc<=0": float(np.mean(np.array(d_ap) <= 0))}


def pattern_labels(name: str, truncate: bool) -> np.ndarray:
    """Tipo de patrón de lavado de cada transacción (alineado con load_ibm)."""
    raw = load_ibm(name, truncate)
    key = (raw["ts"].dt.strftime("%Y/%m/%d %H:%M") + "|" + raw["fb"] + "|" + raw["fa"] + "|"
           + raw["tb"] + "|" + raw["ta"] + "|" + raw["amt_paid"].map("{:.2f}".format))
    lab, cur = {}, None
    for line in (IBM_DIR / f"{name}_Patterns.txt").read_text().splitlines():
        if line.startswith("BEGIN"):
            cur = line.split(" - ")[1].split(":")[0].strip()
        elif line.startswith("END"):
            cur = None
        elif cur and line.strip():
            f = line.split(",")
            lab[f"{f[0]}|{f[1]}|{f[2]}|{f[3]}|{f[4]}|{float(f[7]):.2f}"] = cur
    return key.map(lab).fillna("").to_numpy()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "HI-Small"
    truncate = "--full" not in sys.argv
    tag = name + ("" if truncate else "_full")
    out = Path("results/ibm") / tag
    out.mkdir(parents=True, exist_ok=True)
    df = build_ibm(name, truncate)
    print(df.groupby("split")["y"].agg(["size", "sum"]).to_string(), flush=True)

    rows, preds, imps = [], {}, []
    kinds = ("rf", "hgb") if truncate and name == "HI-Small" else ("rf",)
    for kind in kinds:
        for cname, cols in CONFIGS.items():
            if kind == "hgb" and cname[:2] not in ("B ", "T3"):
                continue
            per = []
            for seed in SEEDS:
                y, p, thr, imp = run_one(df, cols, kind, seed)
                m = metrics(y, p, thr); per.append(m)
                if seed == SEEDS[0]:
                    preds[(kind, cname)] = (p, thr)
                    if imp is not None:
                        imps += [{"config": cname, "feature": c, "importance": float(v)}
                                 for c, v in zip(cols, imp)]
                print(f"[{kind}] {cname} seed={seed} F1={m['f1']:.4f} P={m['precision']:.3f} "
                      f"R={m['recall']:.3f} PR-AUC={m['pr_auc']:.4f}", flush=True)
            a = pd.DataFrame(per)
            r = {"model": kind, "config": cname, "n_features": len(cols)}
            for k in ("f1", "precision", "recall", "roc_auc", "pr_auc", "threshold"):
                r[k], r[k + "_std"] = a[k].mean(), a[k].std()
            rows.append(r)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(imps).to_csv(out / "feature_importance.csv", index=False)

    y = df.loc[df["split"] == "test", "y"].to_numpy()
    pb, tb = preds[("rf", "B transaccional")]
    boot = {}
    for cname in list(CONFIGS)[1:]:
        p1, t1 = preds[("rf", cname)]
        boot[cname] = paired_bootstrap(y, pb, tb, p1, t1)
        print(f"[bootstrap] {cname} vs B: {boot[cname]}", flush=True)
    (out / "paired_bootstrap.json").write_text(json.dumps(boot, indent=2))
    np.savez_compressed(out / "test_preds_seed42.npz", y=y,
                        **{f"{k[0]}|{k[1]}|p": v[0] for k, v in preds.items()},
                        **{f"{k[0]}|{k[1]}|thr": np.array(v[1]) for k, v in preds.items()})

    # Ablación leave-one-group-out sobre T3
    full = CONFIGS["T3 B + contadores + instantaneas"]
    abl = []
    for g, gc in {"txn": TXN, "counters": COUNTERS, "snap": SNAP}.items():
        y_, p_, thr_, _ = run_one(df, [c for c in full if c not in gc], "rf", SEEDS[0])
        m = metrics(y_, p_, thr_)
        abl.append({"removed_group": g, **{k: m[k] for k in ("f1", "precision", "recall", "pr_auc")}})
        print(f"[ablation] sin {g}: F1={m['f1']:.4f} PR-AUC={m['pr_auc']:.4f}", flush=True)
    pd.DataFrame(abl).to_csv(out / "ablation.csv", index=False)

    # Recall por tipo de patrón (semilla 42)
    pat = pattern_labels(name, truncate)[(df["split"] == "test").to_numpy()]
    rec = []
    for cname in ("B transaccional", "T3 B + contadores + instantaneas"):
        p, thr = preds[("rf", cname)]
        hit = p >= thr
        for t in sorted(set(pat[y == 1])):
            msk = (y == 1) & (pat == t)
            rec.append({"config": cname, "pattern": t or "(sin patrón)", "n": int(msk.sum()),
                        "recall": float(hit[msk].mean())})
    pd.DataFrame(rec).to_csv(out / "recall_by_pattern.csv", index=False)
    print(pd.DataFrame(rec).pivot(index="pattern", columns="config", values="recall").to_string())
    print("OK")


if __name__ == "__main__":
    main()
