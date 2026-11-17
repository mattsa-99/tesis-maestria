"""Verifica la exhaustividad por tipo (TRANSFER / CASH_OUT) de N0 y N3 en PaySim sin variables de saldo."""
import json
from pathlib import Path

import numpy as np

from experiments.features import DEGREE, SNAPSHOT, build_all
from experiments.run_experiments import TRAIN_END, run_one
from experiments.run_no_shortcut import NO_SHORTCUT

df = build_all()
types = df.loc[df["step"] > TRAIN_END, "type"].to_numpy()
out = {}
for name, cols in {"N0": NO_SHORTCUT, "N3": NO_SHORTCUT + DEGREE + SNAPSHOT}.items():
    y, p, thr, _ = run_one(df, cols, "rf", 42)
    hit = p >= thr
    out[name] = {t: {"n": int(((y == 1) & (types == t)).sum()),
                     "detectados": int((hit & (y == 1) & (types == t)).sum())}
                 for t in ("TRANSFER", "CASH_OUT")}
    out[name]["falsos_positivos_por_tipo"] = {t: int((hit & (y == 0) & (types == t)).sum())
                                              for t in np.unique(types)}
    print(name, out[name], flush=True)
Path("results/paysim_recall_by_type.json").write_text(json.dumps(out, indent=2))
