"""Experimentos: baseline vs. features topológicas + ablation.

Protocolo (todo causal / sin leakage):
    * Split temporal: train = pasos 1-576 (días 1-24), test = pasos 577-744.
    * Validación (solo para elegir umbral): ajuste en pasos 1-504, validación 505-576.
    * Modelo final: se reajusta en todo train y se evalúa UNA vez en test.
    * Mismos hiperparámetros para todas las configuraciones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from experiments.features import BASE, DEGREE, MERCHANT, SNAPSHOT, build_all

OUT = Path("results")
TRAIN_END, FIT_END = 576, 504
SEEDS = (42, 43, 44)
N_BOOT = 1000

CONFIGS: dict[str, list[str]] = {
    "B0 baseline v0.1": BASE,
    "B1 baseline + merchant": BASE + MERCHANT,
    "T1 B1 + degree": BASE + MERCHANT + DEGREE,
    "T2 B1 + snapshot": BASE + MERCHANT + SNAPSHOT,
    "T3 B1 + degree + snapshot (completo)": BASE + MERCHANT + DEGREE + SNAPSHOT,
    "T4 solo topológicas (sin base)": DEGREE + SNAPSHOT,
}


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, max_samples=0.25,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    yhat = (p >= thr).astype(int)
    return {
        "f1": f1_score(y, yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "threshold": thr,
        "tp": int(((yhat == 1) & (y == 1)).sum()), "fp": int(((yhat == 1) & (y == 0)).sum()),
        "fn": int(((yhat == 0) & (y == 1)).sum()), "tn": int(((yhat == 0) & (y == 0)).sum()),
    }


def run_one(df: pd.DataFrame, cols: list[str], kind: str, seed: int):
    step = df["step"].to_numpy()
    y = df["isFraud"].to_numpy()
    fit, val = step <= FIT_END, (step > FIT_END) & (step <= TRAIN_END)
    train, test = step <= TRAIN_END, step > TRAIN_END
    X = df[cols].to_numpy(dtype=np.float32)

    m = make_model(kind, seed).fit(X[fit], y[fit])
    thr = best_threshold(y[val], m.predict_proba(X[val])[:, 1])
    m = make_model(kind, seed).fit(X[train], y[train])
    p = m.predict_proba(X[test])[:, 1]
    imp = getattr(m, "feature_importances_", None)
    return y[test], p, thr, imp


def paired_bootstrap(y, p_a, thr_a, p_b, thr_b, n=N_BOOT, seed=0):
    """IC 95 % de F1(b) - F1(a) y de F1 individual con remuestreo pareado."""
    rng = np.random.default_rng(seed)
    ya, yb = (p_a >= thr_a), (p_b >= thr_b)
    f_a, f_b, d = [], [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a, b = f1_score(y[i], ya[i], zero_division=0), f1_score(y[i], yb[i], zero_division=0)
        f_a.append(a); f_b.append(b); d.append(b - a)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(f_a), q(f_b), q(d), float(np.mean(np.array(d) <= 0))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = build_all()
    test_mask = df["step"] > TRAIN_END
    print(f"train={int((~test_mask).sum()):,} (fraude={int(df.loc[~test_mask,'isFraud'].sum())}) "
          f"test={int(test_mask.sum()):,} (fraude={int(df.loc[test_mask,'isFraud'].sum())})")

    rows, preds, importances = [], {}, []
    for kind in ("rf", "hgb"):
        for name, cols in CONFIGS.items():
            if kind == "hgb" and not name.startswith(("B1", "T3")):
                continue
            per_seed = []
            for seed in SEEDS:
                y, p, thr, imp = run_one(df, cols, kind, seed)
                m = metrics(y, p, thr)
                per_seed.append(m)
                if seed == SEEDS[0]:
                    preds[(kind, name)] = (p, thr)
                    if imp is not None:
                        importances += [{"config": name, "feature": c, "importance": float(v)}
                                        for c, v in zip(cols, imp)]
                print(f"[{kind}] {name} seed={seed} F1={m['f1']:.4f} P={m['precision']:.3f} "
                      f"R={m['recall']:.3f} PR-AUC={m['pr_auc']:.4f}", flush=True)
            agg = pd.DataFrame(per_seed)
            row = {"model": kind, "config": name, "n_features": len(cols)}
            for k in ("f1", "precision", "recall", "roc_auc", "pr_auc", "threshold"):
                row[k], row[k + "_std"] = agg[k].mean(), agg[k].std()
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)

    # Comparación pareada contra el baseline (semilla 42, RF)
    y = df.loc[test_mask, "isFraud"].to_numpy()
    p0, t0 = preds[("rf", "B0 baseline v0.1")]
    comp = []
    for name in CONFIGS:
        if name.startswith("B0"):
            continue
        p1, t1 = preds[("rf", name)]
        ci_a, ci_b, ci_d, p_val = paired_bootstrap(y, p0, t0, p1, t1)
        comp.append({"config": name, "f1_ci_baseline": ci_a, "f1_ci_config": ci_b,
                     "delta_f1_ci95": ci_d, "p_boot_delta<=0": p_val})
    (OUT / "paired_bootstrap.json").write_text(json.dumps(comp, indent=2))
    pd.DataFrame(importances).to_csv(OUT / "feature_importance.csv", index=False)

    # Ablation leave-one-group-out sobre el modelo completo (RF, semilla 42)
    full = CONFIGS["T3 B1 + degree + snapshot (completo)"]
    groups = {"base": BASE, "merchant": MERCHANT, "degree": DEGREE, "snapshot": SNAPSHOT}
    abl = []
    for g, gc in groups.items():
        cols = [c for c in full if c not in gc]
        y_, p_, thr_, _ = run_one(df, cols, "rf", SEEDS[0])
        m = metrics(y_, p_, thr_)
        abl.append({"removed_group": g, **{k: m[k] for k in ("f1", "precision", "recall", "pr_auc")}})
        print(f"[ablation] sin {g}: F1={m['f1']:.4f} PR-AUC={m['pr_auc']:.4f}", flush=True)
    pd.DataFrame(abl).to_csv(OUT / "ablation_leave_one_group_out.csv", index=False)

    # Umbral sobre el modelo completo: curva F1 vs umbral en test (solo descriptivo)
    p, _ = preds[("rf", "T3 B1 + degree + snapshot (completo)")]
    prec, rec, thr = precision_recall_curve(y, p)
    pd.DataFrame({"threshold": thr, "precision": prec[:-1], "recall": rec[:-1]}).to_csv(
        OUT / "pr_curve_full.csv", index=False)
    p, _ = preds[("rf", "B0 baseline v0.1")]
    prec, rec, thr = precision_recall_curve(y, p)
    pd.DataFrame({"threshold": thr, "precision": prec[:-1], "recall": rec[:-1]}).to_csv(
        OUT / "pr_curve_baseline.csv", index=False)
    print("OK")


if __name__ == "__main__":
    main()
