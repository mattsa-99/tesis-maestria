"""Hipótesis rival R1: ¿la ganancia en IBM viene de la estructura o del comportamiento temporal?

Separa los contadores causales en:
    * ESTRUCTURALES: grado y fuerza previos, grado en ventana de 24 h (fan-in/fan-out),
      repetición del par y arista inversa (ciclo de longitud 2).
    * TEMPORALES: minutos desde la última transacción del origen y del destino.
Las variables de hora y día de la semana ya están en el modelo de referencia (B).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.ibm_features import SNAP, TXN, build_ibm
from experiments.run_experiments import metrics
from experiments.run_ibm import SEEDS, paired_bootstrap, run_one

TEMPORAL = ["src_gap", "dst_gap"]
STRUCT = ["src_out_prev", "src_in_prev", "dst_in_prev", "dst_out_prev", "src_out_amt_prev",
          "dst_in_amt_prev", "src_fanout_24h", "dst_fanin_24h", "pair_prev", "reverse_edge_prev"]
CONFIGS = {
    "B": TXN,
    "B + estructurales": TXN + STRUCT,
    "B + temporales": TXN + TEMPORAL,
    "B + estructurales + instantaneas": TXN + STRUCT + SNAP,
    "T3 completo": TXN + STRUCT + TEMPORAL + SNAP,
}


def main() -> None:
    out = Path("results/ibm/HI-Small")
    df = build_ibm("HI-Small", True)
    y = df.loc[df["split"] == "test", "y"].to_numpy()
    rows, preds = [], {}
    for name, cols in CONFIGS.items():
        per = []
        for seed in SEEDS:
            y_, p, thr, _ = run_one(df, cols, "rf", seed)
            m = metrics(y_, p, thr); per.append(m)
            if seed == SEEDS[0]:
                preds[name] = (p, thr)
            print(f"{name} seed={seed} F1={m['f1']:.4f} PR-AUC={m['pr_auc']:.4f}", flush=True)
        a = pd.DataFrame(per)
        rows.append({"config": name, "n_features": len(cols),
                     **{k: a[k].mean() for k in ("f1", "precision", "recall", "pr_auc")},
                     "f1_std": a["f1"].std(), "pr_auc_std": a["pr_auc"].std()})
        pd.DataFrame(rows).to_csv(out / "struct_vs_time.csv", index=False)
    boot = {}
    for a_, b_ in [("B", "B + estructurales"), ("B", "B + temporales"),
                   ("B + estructurales", "T3 completo"),
                   ("B", "B + estructurales + instantaneas")]:
        boot[f"{b_} vs {a_}"] = paired_bootstrap(y, *preds[a_], *preds[b_])
        print(f"[bootstrap] {b_} vs {a_}: {boot[f'{b_} vs {a_}']}", flush=True)
    (out / "struct_vs_time_bootstrap.json").write_text(json.dumps(boot, indent=2))
    print("OK")


if __name__ == "__main__":
    main()
