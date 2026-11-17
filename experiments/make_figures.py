"""Genera las figuras del Capítulo 4 a partir de results/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Convención numérica de la tesis (RAE, 2010): punto decimal; es el formato por defecto de matplotlib.

R, F = Path("results"), Path("results/figuras")
F.mkdir(exist_ok=True)

m = pd.read_csv(R / "metrics.csv")
rf = m[m.model == "rf"]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(rf.config, rf.f1, xerr=rf.f1_std, color="#4c78a8")
ax.set_xlabel("F1 en prueba (media ± d.e., 3 semillas)"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(F / "f1_por_config.png", dpi=200); plt.close(fig)

n = pd.read_csv(R / "metrics_no_shortcut.csv")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
for a, col, t in zip(ax, ("f1", "pr_auc"), ("F1", "PR-AUC")):
    a.barh(n.config, n[col], color="#59a14f"); a.set_xlabel(t + " (prueba)"); a.invert_yaxis()
ax[1].set_yticklabels([])
fig.tight_layout(); fig.savefig(F / "sin_atajo.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
for f, lab in (("pr_curve_baseline.csv", "B0 baseline"), ("pr_curve_full.csv", "T3 completo")):
    c = pd.read_csv(R / f); ax.plot(c.recall, c.precision, label=lab)
ax.set_xlabel("Exhaustividad"); ax.set_ylabel("Precisión"); ax.legend(); ax.set_ylim(0.9, 1.005)
fig.tight_layout(); fig.savefig(F / "pr_curvas.png", dpi=200); plt.close(fig)

i = pd.read_csv(R / "feature_importance.csv")
i = i[i.config.str.startswith("B0")].sort_values("importance")
fig, ax = plt.subplots(figsize=(6, 3.4)); ax.barh(i.feature, i.importance, color="#e15759")
ax.set_xlabel("Importancia (B0)"); fig.tight_layout(); fig.savefig(F / "importancia_b0.png", dpi=200)

# --- IBM AML y Elliptic -------------------------------------------------------
ibm = {d: pd.read_csv(R / "ibm" / d / "metrics.csv") for d in ("HI-Small", "LI-Small")}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, (d, t) in zip(ax, ibm.items()):
    t = t[t.model == "rf"]
    a.barh(t.config, t.pr_auc, color="#4c78a8", label="PR-AUC")
    a.scatter(t.f1, t.config, color="#e15759", zorder=3, label="F1")
    a.set_title(f"IBM {d}"); a.set_xlim(0, 1); a.invert_yaxis() if d == "HI-Small" else None
ax[0].legend(loc="lower right")
fig.tight_layout(); fig.savefig(F / "ibm_configs.png", dpi=200); plt.close(fig)

rp = pd.read_csv(R / "ibm" / "HI-Small" / "recall_by_pattern.csv")
pv = rp.pivot(index="pattern", columns="config", values="recall")
fig, ax = plt.subplots(figsize=(7, 4))
pv.plot.barh(ax=ax, color=["#bab0ac", "#4c78a8"]); ax.set_xlabel("Exhaustividad en prueba (HI-Small)")
ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
fig.savefig(F / "ibm_recall_patron.png", dpi=200); plt.close(fig)

el = pd.read_csv(R / "elliptic" / "metrics.csv")
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.barh(el.config, el.f1, xerr=el.f1_std, color="#59a14f"); ax.invert_yaxis()
ax.set_xlabel("F1 en prueba (media ± d.e., 3 semillas)"); fig.tight_layout()
fig.savefig(F / "elliptic_f1.png", dpi=200); plt.close(fig)

inj = R / "injection" / "summary.csv"
if inj.exists():
    s = pd.read_csv(inj)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for a, (prev, g) in zip(ax, s.groupby("prevalence")):
        a.plot(g.decoys, g.B1_f1, "o-", color="#bab0ac", label="B1 (sin topología)")
        a.plot(g.decoys, g.T3_f1, "o-", color="#4c78a8", label="T3 (con topología)")
        a.set_title(f"prevalencia {prev * 100:.2f} %"); a.set_xlabel("señuelos por patrón"); a.set_xticks([0, 1, 5])
    ax[0].set_ylabel("F1 en prueba"); ax[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(F / "inyeccion_f1.png", dpi=200); plt.close(fig)
