"""Línea 2: inyección controlada de fraude estructurado en PaySim (prueba de capacidad).

Diseño (3.8):
    * Se eliminan los fraudes originales de PaySim (que se detectan por el atajo de saldos);
      las únicas positivas son los patrones inyectados.
    * Patrones: cadena de mulas (3-5 saltos + CASH_OUT), ciclo (3-5 cuentas) y anillo
      fan-in (5-15 orígenes -> recolector -> CASH_OUT).
    * Camuflaje individual: cada transacción inyectada copia monto y saldos previos de una
      transacción legítima real del mismo tipo, de modo que sus features individuales
      siguen la distribución legítima.
    * Señuelos: estructuras idénticas etiquetadas como legítimas, repartidas en días en lugar
      de horas. Sin señuelos, cualquier estructura multisalto delataría el fraude porque el
      fondo de PaySim no la tiene.
    * Rejilla: prevalencia {0.05 %, 0.1 %, 0.5 %} x señuelos por patrón fraudulento {0, 1, 5}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.features import (BASE, DEGREE, MERCHANT, SNAPSHOT, base_features,
                                  degree_features, load_raw, snapshot_features)
from experiments.run_experiments import best_threshold, make_model, metrics
from experiments.run_ibm import paired_bootstrap

OUT = Path("results/injection")
PREVALENCES = (0.0005, 0.001, 0.005)
DECOYS = (0, 1, 5)
PATTERNS = ("chain", "cycle", "fanin")
TRAIN_END, FIT_END, MAX_STEP = 576, 504, 743


def _edges(kind: str, rng, new_node, existing_orig):
    """Devuelve lista de (orig, dest, type, hop) para un patrón."""
    if kind == "chain":
        k = rng.integers(3, 6)
        nodes = [rng.choice(existing_orig)] + [new_node() for _ in range(k)]
        e = [(nodes[i], nodes[i + 1], "TRANSFER", i) for i in range(k)]
        e.append((nodes[-1], new_node(), "CASH_OUT", k))
        return e
    if kind == "cycle":
        k = rng.integers(3, 6)
        nodes = [new_node() for _ in range(k)]
        return [(nodes[i], nodes[(i + 1) % k], "TRANSFER", i) for i in range(k)]
    k = rng.integers(5, 16)
    hub = new_node()
    e = [(rng.choice(existing_orig), hub, "TRANSFER", 0) for _ in range(k)]
    e.append((hub, new_node(), "CASH_OUT", 1))
    return e


def inject(raw: pd.DataFrame, prevalence: float, decoys: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    legit = raw[raw["isFraud"] == 0].reset_index(drop=True)
    pools = {t: legit.loc[legit["type"] == t, ["amount", "oldbalanceOrg", "oldbalanceDest"]]
             .to_numpy() for t in ("TRANSFER", "CASH_OUT")}
    existing = legit.loc[legit["type"] == "TRANSFER", "nameOrig"].to_numpy()
    legit_steps = legit["step"].to_numpy()
    counter = iter(range(10**9))
    new_node = lambda: f"CINJ{next(counter):09d}"

    target = int(prevalence * len(legit))
    rows, n_fraud, pid = [], 0, 0
    while n_fraud < target:
        kind = PATTERNS[pid % 3]
        for d in range(decoys + 1):
            is_fraud = int(d == 0)
            e = _edges(kind, rng, new_node, existing)
            # Saltos cronológicos: fraude en horas (1-2 h entre saltos, jitter <= 1 h);
            # señuelo en días (12-72 h entre saltos, jitter <= 47 h).
            # Inicio muestreado de la distribución temporal legítima (v2): con inicio uniforme
            # (v1) la tasa de positivos en test era 9x la de train y el baseline aprendía el tiempo.
            hop_start = int(min(rng.choice(legit_steps), MAX_STEP - 24))
            cur_hop, hop_max = 0, 0
            steps = []
            for (_, _, _, hop) in e:
                if hop != cur_hop:
                    gap = int(rng.integers(1, 3)) if is_fraud else int(rng.integers(12, 73))
                    hop_start, cur_hop = hop_max + gap, hop
                jitter = int(rng.integers(0, 2)) if is_fraud else int(rng.integers(0, 48))
                if kind != "fanin":
                    jitter = 0
                steps.append(min(hop_start + jitter, MAX_STEP))
                hop_max = max(hop_max, steps[-1])
            for (o, de, t, hop), step in zip(e, steps):
                amt, ob, od = pools[t][rng.integers(0, len(pools[t]))]
                rows.append({"step": step, "type": t, "amount": amt, "nameOrig": o,
                             "oldbalanceOrg": ob, "newbalanceOrig": max(ob - amt, 0.0),
                             "nameDest": de, "oldbalanceDest": od, "newbalanceDest": od + amt,
                             "isFraud": is_fraud, "isFlaggedFraud": 0,
                             "pattern": kind if is_fraud else f"decoy_{kind}"})
            n_fraud += is_fraud * len(e)
        pid += 1
    inj = pd.DataFrame(rows)
    base = legit.drop(columns=[c for c in ("orig_id", "dest_id", "dest_is_merchant") if c in legit])
    base["pattern"] = ""
    df = pd.concat([base, inj], ignore_index=True).sort_values("step", kind="stable")
    df = df.reset_index(drop=True)
    codes, _ = pd.factorize(pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True))
    n = len(df)
    df["orig_id"], df["dest_id"] = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(np.int8)
    return df


def evaluate(df: pd.DataFrame, seed: int = 42) -> dict:
    feats = pd.concat([base_features(df), degree_features(df),
                       snapshot_features(df, verbose=False)], axis=1)
    y, step = df["isFraud"].to_numpy(), df["step"].to_numpy()
    fit, val = step <= FIT_END, (step > FIT_END) & (step <= TRAIN_END)
    tr, te = step <= TRAIN_END, step > TRAIN_END
    res, preds = {}, {}
    for name, cols in {"B1": BASE + MERCHANT, "T3": BASE + MERCHANT + DEGREE + SNAPSHOT}.items():
        X = feats[cols].to_numpy(dtype=np.float32)
        m = make_model("rf", seed).fit(X[fit], y[fit])
        thr = best_threshold(y[val], m.predict_proba(X[val])[:, 1])
        m = make_model("rf", seed).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        preds[name] = (p, thr)
        r = metrics(y[te], p, thr)
        pat = df.loc[te, "pattern"].to_numpy()
        for k in PATTERNS:
            msk = (y[te] == 1) & (pat == k)
            r[f"recall_{k}"] = float((p[msk] >= thr).mean()) if msk.any() else float("nan")
        res[name] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                     for k, v in r.items()}
    res["bootstrap_T3_vs_B1"] = paired_bootstrap(y[te], *preds["B1"], *preds["T3"], n=500)
    res["n_test_pos"] = int(y[te].sum())
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    print(f"PaySim sin fraudes originales: {int((raw.isFraud == 0).sum()):,} filas", flush=True)
    summary = []
    for prev in PREVALENCES:
        for dec in DECOYS:
            df = inject(raw, prev, dec, seed=7)
            r = evaluate(df)
            (OUT / f"prev{prev}_decoy{dec}.json").write_text(json.dumps(r, indent=2))
            b = r["bootstrap_T3_vs_B1"]
            row = {"prevalence": prev, "decoys": dec, "n_test_pos": r["n_test_pos"]}
            for mname in ("B1", "T3"):
                for k in ("f1", "pr_auc", "recall_chain", "recall_cycle", "recall_fanin"):
                    row[f"{mname}_{k}"] = r[mname][k]
            row["dF1_lo"], row["dF1_hi"] = b["delta_f1_ci95"]
            row["dPR_lo"], row["dPR_hi"] = b["delta_pr_auc_ci95"]
            summary.append(row)
            pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
            print(f"prev={prev} decoys={dec} pos_test={r['n_test_pos']} "
                  f"B1 F1={r['B1']['f1']:.3f} PR={r['B1']['pr_auc']:.3f} | "
                  f"T3 F1={r['T3']['f1']:.3f} PR={r['T3']['pr_auc']:.3f} | "
                  f"dF1 [{b['delta_f1_ci95'][0]:+.3f},{b['delta_f1_ci95'][1]:+.3f}] "
                  f"rec T3 chain/cycle/fanin={r['T3']['recall_chain']:.2f}/"
                  f"{r['T3']['recall_cycle']:.2f}/{r['T3']['recall_fanin']:.2f}", flush=True)
    print("OK")


if __name__ == "__main__":
    main()
