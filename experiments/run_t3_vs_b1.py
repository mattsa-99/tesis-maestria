"""Bootstrap pareado de T3 frente a B1 (comparación que define el contraste de H1)."""
import json

from experiments.features import BASE, DEGREE, MERCHANT, SNAPSHOT, build_all
from experiments.run_experiments import OUT, SEEDS, TRAIN_END, paired_bootstrap, run_one

df = build_all()
y = df.loc[df["step"] > TRAIN_END, "isFraud"].to_numpy()
_, p1, t1, _ = run_one(df, BASE + MERCHANT, "rf", SEEDS[0])
_, p3, t3, _ = run_one(df, BASE + MERCHANT + DEGREE + SNAPSHOT, "rf", SEEDS[0])
ci_a, ci_b, ci_d, pv = paired_bootstrap(y, p1, t1, p3, t3)
out = {"f1_ci_B1": ci_a, "f1_ci_T3": ci_b, "delta_f1_ci95": ci_d, "p_boot_delta<=0": pv}
(OUT / "paired_bootstrap_T3_vs_B1.json").write_text(json.dumps(out, indent=2))
print(out)
