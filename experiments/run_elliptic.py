"""Réplica 2: Elliptic (Weber et al., 2019).

Nodos = transacciones de Bitcoin; aristas = flujo de fondos; 49 periodos (~2 semanas c/u).
Split estándar: entrenamiento periodos 1-34 (ajuste 1-26, validación del umbral 27-34),
prueba 35-49. Solo nodos etiquetados (ilícito / lícito).

Grupos:
    L  -> 93 features locales (se excluye el número de periodo)
    AF -> 72 features agregadas de vecinos provistas por el dataset (topológicas)
    G  -> métricas de grafo propias, calculadas por periodo (el grafo de Elliptic no
          tiene aristas entre periodos)
"""
from __future__ import annotations

import json
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

from experiments.run_experiments import best_threshold, metrics
from experiments.run_ibm import make_model, paired_bootstrap

E_DIR = Path("data/raw/_kagglehub/datasets/ellipticco/elliptic-data-set/versions/1/"
             "elliptic_bitcoin_dataset")
OUT = Path("results/elliptic")
SEEDS = (42, 43, 44)
G = ["in_deg", "out_deg", "pagerank", "wcc_size", "btw", "comm_size", "coreness",
     "clustering", "in_reach2", "out_reach2"]


def _find(name: str) -> Path:
    hits = list(E_DIR.parent.rglob(name))
    assert hits, f"no se encontró {name}"
    return hits[0]


def load() -> pd.DataFrame:
    fe = pd.read_csv(_find("elliptic_txs_features.csv"), header=None)
    fe.columns = ["txId", "t"] + [f"L{i}" for i in range(93)] + [f"AF{i}" for i in range(72)]
    cl = pd.read_csv(_find("elliptic_txs_classes.csv"))
    ed = pd.read_csv(_find("elliptic_txs_edgelist.csv"))
    df = fe.merge(cl, on="txId")
    gf = graph_features(df[["txId", "t"]], ed)
    df = df.merge(gf, on="txId")
    return df[df["class"] != "unknown"].assign(y=lambda d: (d["class"] == "1").astype(int))


def graph_features(nodes: pd.DataFrame, ed: pd.DataFrame) -> pd.DataFrame:
    idx = pd.Series(np.arange(len(nodes)), index=nodes["txId"].to_numpy())
    s, d = idx[ed.iloc[:, 0]].to_numpy(), idx[ed.iloc[:, 1]].to_numpy()
    g = ig.Graph(n=len(nodes), edges=np.column_stack([s, d]).tolist(), directed=True)
    und = g.as_undirected(mode="collapse")
    comp = np.asarray(g.connected_components(mode="weak").membership)
    comm = np.asarray(und.community_multilevel().membership)
    out = pd.DataFrame({"txId": nodes["txId"].to_numpy()})
    out["in_deg"], out["out_deg"] = g.indegree(), g.outdegree()
    # PageRank por componente equivale a PageRank por periodo (no hay aristas entre periodos)
    out["pagerank"] = g.pagerank(directed=True)
    out["wcc_size"] = np.bincount(comp)[comp]
    out["btw"] = g.betweenness(directed=True, cutoff=4)
    out["comm_size"] = np.bincount(comm)[comm]
    out["coreness"] = und.coreness()
    out["clustering"] = np.nan_to_num(und.transitivity_local_undirected(mode="zero"))
    out["in_reach2"] = [len(x) - 1 for x in g.neighborhood(order=2, mode="in")]
    out["out_reach2"] = [len(x) - 1 for x in g.neighborhood(order=2, mode="out")]
    return out


def run_one(df, cols, seed):
    X, y, t = df[cols].to_numpy(dtype=np.float32), df["y"].to_numpy(), df["t"].to_numpy()
    fit, val, tr, te = t <= 26, (t > 26) & (t <= 34), t <= 34, t > 34
    m = make_model("rf", seed).fit(X[fit], y[fit])
    thr = best_threshold(y[val], m.predict_proba(X[val])[:, 1])
    m = make_model("rf", seed).fit(X[tr], y[tr])
    return y[te], m.predict_proba(X[te])[:, 1], thr, t[te]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    L = [f"L{i}" for i in range(93)]
    AF = [f"AF{i}" for i in range(72)]
    configs = {"L local": L, "L + AF": L + AF, "L + G": L + G, "L + AF + G": L + AF + G,
               "G solo grafo": G}
    te = df["t"] > 34
    print(f"etiquetados={len(df):,} ilícitos={int(df.y.sum()):,} | test={int(te.sum()):,} "
          f"ilícitos test={int(df.loc[te, 'y'].sum()):,}", flush=True)
    rows, preds = [], {}
    for name, cols in configs.items():
        per = []
        for seed in SEEDS:
            y, p, thr, tt = run_one(df, cols, seed)
            m = metrics(y, p, thr); per.append(m)
            preds[(name, seed)] = (p, thr)
            print(f"{name} seed={seed} F1={m['f1']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"PR-AUC={m['pr_auc']:.4f}", flush=True)
        a = pd.DataFrame(per)
        r = {"config": name, "n_features": len(cols)}
        for k in ("f1", "precision", "recall", "roc_auc", "pr_auc"):
            r[k], r[k + "_std"] = a[k].mean(), a[k].std()
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)

    # En Elliptic la varianza entre semillas es comparable a las diferencias: bootstrap por semilla.
    boot = {}
    for a_, b_ in [("L local", "L + G"), ("L local", "L + AF"), ("L local", "L + AF + G"),
                   ("L + AF", "L + AF + G")]:
        for seed in SEEDS:
            k = f"{b_} vs {a_} (semilla {seed})"
            boot[k] = paired_bootstrap(y, *preds[(a_, seed)], *preds[(b_, seed)])
            print(f"[bootstrap] {k}: dF1={[round(v, 4) for v in boot[k]['delta_f1_ci95']]} "
                  f"dPR={[round(v, 4) for v in boot[k]['delta_pr_auc_ci95']]}", flush=True)
    (OUT / "paired_bootstrap.json").write_text(json.dumps(boot, indent=2))

    # F1 por periodo (cierre del mercado oscuro en t=43; Weber et al., 2019)
    per_t = []
    for name in ("L local", "L + G", "L + AF + G"):
        p, thr = preds[(name, SEEDS[0])]
        for t in sorted(set(tt)):
            msk = tt == t
            yy, hh = y[msk].astype(bool), p[msk] >= thr
            tp = int((yy & hh).sum()); fp = int((~yy.astype(bool) & hh).sum()); fn = int((yy & ~hh).sum())
            per_t.append({"config": name, "t": int(t), "n_illicit": int(yy.sum()),
                          "f1": 2 * tp / max(2 * tp + fp + fn, 1)})
    pd.DataFrame(per_t).to_csv(OUT / "f1_by_timestep.csv", index=False)
    print("OK")


if __name__ == "__main__":
    main()
