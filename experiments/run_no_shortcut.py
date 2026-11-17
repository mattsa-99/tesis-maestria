"""Análisis post hoc: ¿aporta la topología cuando se elimina el atajo de PaySim?

En PaySim el fraude vacía la cuenta de origen (amount ≈ oldbalanceOrg) y casi todas
las transacciones legítimas CASH_OUT/TRANSFER tienen saldo insuficiente. Las
features que reconstruyen esa relación dejan el baseline en el techo (F1 ≈ 0.998).
Aquí se retiran amount, saldo de origen, su razón e ``insufficient_funds``.

Este experimento NO estaba en el diseño original: se reporta como exploratorio.
"""
from __future__ import annotations

import json

import pandas as pd

from experiments.features import DEGREE, SNAPSHOT, build_all
from experiments.run_experiments import (OUT, SEEDS, TRAIN_END, metrics,
                                         paired_bootstrap, run_one)

NO_SHORTCUT = ["type_enc", "dest_balance_log", "dest_was_empty", "step", "dest_is_merchant"]
CONFIGS = {
    "N0 sin atajo": NO_SHORTCUT,
    "N1 sin atajo + degree": NO_SHORTCUT + DEGREE,
    "N2 sin atajo + snapshot": NO_SHORTCUT + SNAPSHOT,
    "N3 sin atajo + degree + snapshot": NO_SHORTCUT + DEGREE + SNAPSHOT,
}


def main() -> None:
    df = build_all()
    y = df.loc[df["step"] > TRAIN_END, "isFraud"].to_numpy()
    rows, preds, imps = [], {}, []
    for name, cols in CONFIGS.items():
        per_seed = []
        for seed in SEEDS:
            y_, p, thr, imp = run_one(df, cols, "rf", seed)
            m = metrics(y_, p, thr)
            per_seed.append(m)
            if seed == SEEDS[0]:
                preds[name] = (p, thr)
                imps += [{"config": name, "feature": c, "importance": float(v)}
                         for c, v in zip(cols, imp)]
            print(f"{name} seed={seed} F1={m['f1']:.4f} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} PR-AUC={m['pr_auc']:.4f}", flush=True)
        agg = pd.DataFrame(per_seed)
        row = {"config": name, "n_features": len(cols)}
        for k in ("f1", "precision", "recall", "roc_auc", "pr_auc", "threshold"):
            row[k], row[k + "_std"] = agg[k].mean(), agg[k].std()
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "metrics_no_shortcut.csv", index=False)

    p0, t0 = preds["N0 sin atajo"]
    comp = []
    for name in list(CONFIGS)[1:]:
        p1, t1 = preds[name]
        ci_a, ci_b, ci_d, pv = paired_bootstrap(y, p0, t0, p1, t1)
        comp.append({"config": name, "f1_ci_N0": ci_a, "f1_ci_config": ci_b,
                     "delta_f1_ci95": ci_d, "p_boot_delta<=0": pv})
    (OUT / "paired_bootstrap_no_shortcut.json").write_text(json.dumps(comp, indent=2))
    pd.DataFrame(imps).to_csv(OUT / "feature_importance_no_shortcut.csv", index=False)
    print("OK")


if __name__ == "__main__":
    main()
