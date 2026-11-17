"""Figuras del Capítulo 2 (marco teórico) sobre la red de ejemplo que recorre el capítulo.

La red tiene 7 cuentas (A–G) y 8 transacciones; contiene un fan-in hacia D, un fan-out desde E,
el ciclo dirigido D→E→F→D, un par repetido (A→D) y dos cuentas intermediarias (D y E).
Todos los valores que se citan en los ejemplos del capítulo se imprimen al ejecutar el script.

Uso: python docs/herramientas/figuras_marco.py
Paleta: referencia validada de la skill de visualización (azul, naranja, verde agua; secuencial azul).
"""
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path("docs/capitulos/img")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- estilo
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29d", "#e8e7e4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # rampa azul 100→700
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

# ---------------------------------------------------------------- red de ejemplo
V = list("ABCDEFG")
IX = {v: i for i, v in enumerate(V)}
TX = [  # (origen, destino, hora, monto)
    ("A", "D", 1, 500), ("B", "D", 2, 300), ("C", "D", 3, 200), ("D", "E", 5, 900),
    ("E", "F", 6, 450), ("E", "G", 6, 400), ("F", "D", 8, 430), ("A", "D", 9, 150),
]
POS = {"A": (0.0, 2.4), "B": (0.0, 1.2), "C": (0.0, 0.0), "D": (1.6, 1.2),
       "E": (3.2, 1.2), "F": (2.4, 2.55), "G": (4.6, 1.2)}
R = 0.24


def multigraph(tx=TX):
    g = ig.Graph(n=len(V), edges=[(IX[u], IX[v]) for u, v, _, _ in tx], directed=True)
    g.vs["name"] = V
    return g


def simple(tx=TX):
    g = multigraph(tx); g.simplify()
    return g


# ---------------------------------------------------------------- dibujo
def new_ax(ax=None, w=6.0, h=3.3):
    if ax is None:
        fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(-0.55, 5.15); ax.set_ylim(-0.5, 3.05); ax.set_aspect("equal"); ax.axis("off")
    return ax


def draw_nodes(ax, fill=None, text=None, label_extra=None, sizes=None):
    fill = fill or {}; text = text or {}; label_extra = label_extra or {}; sizes = sizes or {}
    for v, (x, y) in POS.items():
        r = sizes.get(v, R)
        ax.add_patch(Circle((x, y), r, facecolor=fill.get(v, "#f3f2ef"), edgecolor=INK2, lw=1.2, zorder=3))
        ax.text(x, y, v, ha="center", va="center", fontsize=11, fontweight="bold", color=text.get(v, INK), zorder=4)
        if v in label_extra:
            dy = r + 0.17 if y < 2.3 else -(r + 0.2)
            if v == "F": dy = r + 0.15
            ax.text(x, y + dy, label_extra[v], ha="center", va="center", fontsize=8.5, color=INK2, zorder=4)


def draw_edge(ax, u, v, color=INK2, lw=1.4, rad=0.0, ls="-", label=None, alpha=1.0, z=2):
    (x1, y1), (x2, y2) = POS[u], POS[v]
    a = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                        mutation_scale=13, lw=lw, color=color, linestyle=ls, alpha=alpha,
                        shrinkA=R * 72 / 1.0 * 0.0 + 15, shrinkB=17, zorder=z)
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy) or 1
        # lado derecho de la dirección (dy, -dx): hacia ahí se curva arc3 con rad > 0
        rx, ry = dy / L, -dx / L
        if rad:
            sgn = np.sign(rad)
            px, py = mx + rad * dy * 0.5 + rx * sgn * 0.14, my - rad * dx * 0.5 + ry * sgn * 0.14
        else:
            px, py = mx + rx * 0.2, my + ry * 0.2
        ax.text(px, py, label, fontsize=7.8, color=INK2, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9), zorder=5)


def draw_tx(ax, tx=TX, style=None, labels=True):
    """Dibuja las transacciones; las repetidas en un mismo par se curvan para no superponerse."""
    style = style or {}
    seen = {}
    for i, (u, v, t, w) in enumerate(tx):
        k = seen.get((u, v), 0); seen[(u, v)] = k + 1
        n_pair = sum(1 for a, b, _, _ in tx if (a, b) == (u, v))
        rad = 0.0 if n_pair == 1 else (0.22 if k == 0 else -0.22)
        st = dict(style.get(i, {}))
        draw_edge(ax, u, v, rad=rad, label=(f"t={t} · {w}" if labels else None), **st)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figura:", name)


# ---------------------------------------------------------------- figuras
def fig_red():
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); new_ax(ax)
    draw_tx(ax); draw_nodes(ax)
    save(fig, "fig2_01_red.png")


def fig_instantaneas():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.3))
    for ax, tau in zip(axs, (7, 10)):
        new_ax(ax)
        tx = [x for x in TX if x[2] < tau]
        draw_tx(ax, tx); draw_nodes(ax)
        ax.set_title(f"Instantánea $G_{{<{tau}}}$  ({len(tx)} transacciones)", fontsize=11, color=INK)
    save(fig, "fig2_02_instantaneas.png")


def fig_contadores():
    fig, ax = plt.subplots(figsize=(6.6, 3.5)); new_ax(ax)
    # transacción evaluada e8 = A→D en t=9; ventana Δ = 6 h: [3, 9)
    style = {}
    for i, (u, v, t, w) in enumerate(TX):
        if i == 7:
            style[i] = dict(color=ORANGE, lw=2.4)
        elif v == "D" and 3 <= t < 9:
            style[i] = dict(color=BLUE, lw=2.2)
        elif v == "D":
            style[i] = dict(color=MUTED, ls="--")
        else:
            style[i] = dict(color=GRID, lw=1.2)
    draw_tx(ax, style=style)
    draw_nodes(ax, fill={"A": "#fbe1d6", "D": "#cde2fb"})
    ax.text(3.15, -0.38, "naranja: transacción evaluada (A→D, t=9)\nazul: entradas de D en la ventana [3, 9)\n"
            "gris discontinuo: entradas de D anteriores a la ventana", fontsize=8, color=INK2, va="bottom")
    save(fig, "fig2_03_contadores.png")


def fig_componentes():
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); new_ax(ax)
    cyc = {("D", "E"), ("E", "F"), ("F", "D")}
    style = {i: (dict(color=BLUE, lw=2.4) if (u, v) in cyc else dict(color=MUTED)) for i, (u, v, _, _) in enumerate(TX)}
    draw_tx(ax, style=style, labels=False)
    draw_nodes(ax, fill={"D": BLUE, "E": BLUE, "F": BLUE}, text={"D": "white", "E": "white", "F": "white"})
    ax.text(2.4, -0.42, "azul: componente fuertemente conexa {D, E, F} = ciclo D→E→F→D", fontsize=8.5,
            color=INK2, ha="center")
    save(fig, "fig2_04_componentes.png")


def fig_adyacencia():
    tx = [x for x in TX if x[2] < 8]
    A = np.zeros((7, 7), int)
    for u, v, _, _ in tx:
        if u != v: A[IX[u], IX[v]] = 1
    A2 = A @ A
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.9))
    for ax, M, title in zip(axs, (A, A2), ("$A$ de $\\bar G_{<8}$", "$A^2$ (paseos de longitud 2)")):
        for i in range(7):
            for j in range(7):
                ax.add_patch(Rectangle((j + 0.04, 6 - i + 0.04), 0.92, 0.92,
                                       facecolor=(BLUE if M[i, j] else "#f3f2ef"), edgecolor="white", lw=2))
                if M[i, j]:
                    ax.text(j + 0.5, 6 - i + 0.5, str(M[i, j]), ha="center", va="center", color="white", fontsize=9)
        ax.set_xlim(0, 7); ax.set_ylim(0, 7); ax.set_aspect("equal")
        ax.set_xticks(np.arange(7) + 0.5, V); ax.set_yticks(np.arange(7) + 0.5, V[::-1])
        ax.xaxis.tick_top(); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, fontsize=11, pad=22)
        ax.set_xlabel("destino", fontsize=9); ax.set_ylabel("origen", fontsize=9)
    i, j = IX["D"], IX["F"]
    axs[1].add_patch(Rectangle((j, 6 - i), 1, 1, fill=False, edgecolor=ORANGE, lw=3))
    axs[1].text(7.15, 6 - i + 0.5, "$(A^2)_{DF} = 1$:\nD→E→F", fontsize=8.5, color=INK2, va="center")
    save(fig, "fig2_05_adyacencia.png")


def _seq_color(x, lo, hi):
    k = 0 if hi == lo else int(round((x - lo) / (hi - lo) * (len(SEQ) - 1)))
    return SEQ[k]


def fig_metric(values, name, fmt, title):
    fig, ax = plt.subplots(figsize=(6.2, 3.5)); new_ax(ax)
    lo, hi = min(values.values()), max(values.values())
    draw_tx(ax, style={i: dict(color=MUTED) for i in range(len(TX))}, labels=False)
    fill = {v: _seq_color(values[v], lo, hi) for v in V}
    text = {v: ("white" if SEQ.index(fill[v]) >= 3 else INK) for v in V}
    draw_nodes(ax, fill=fill, text=text, label_extra={v: fmt(values[v]) for v in V})
    ax.text(2.4, -0.45, title, fontsize=8.5, color=INK2, ha="center")
    save(fig, name)


def fig_comunidades(memb):
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); new_ax(ax)
    cols = [BLUE, ORANGE]
    draw_tx(ax, style={i: dict(color=MUTED) for i in range(len(TX))}, labels=False)
    fill = {v: cols[memb[IX[v]]] for v in V}
    text = {v: ("white" if fill[v] == BLUE else INK) for v in V}
    draw_nodes(ax, fill=fill, text=text)
    ax.text(2.4, -0.45, "azul: comunidad {A, B, C, D} · naranja: comunidad {E, F, G}", fontsize=8.5, color=INK2, ha="center")
    save(fig, "fig2_08_comunidades.png")


def fig_betti():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.3))
    variants = [
        (TX, "Red de ejemplo: ciclo dirigido D→E→F→D\n$\\beta_1 = 1$ y un ciclo dirigido"),
        ([("D", "F", t, w) if (u, v) == ("F", "D") else (u, v, t, w) for u, v, t, w in TX],
         "Variante con D→F en lugar de F→D\n$\\beta_1 = 1$ y ningún ciclo dirigido"),
    ]
    for ax, (tx, title) in zip(axs, variants):
        new_ax(ax)
        tri = {frozenset(p) for p in [("D", "E"), ("E", "F"), ("F", "D")]}
        style = {i: (dict(color=BLUE, lw=2.4) if frozenset((u, v)) in tri else dict(color=MUTED)) for i, (u, v, _, _) in enumerate(tx)}
        draw_tx(ax, tx, style=style, labels=False); draw_nodes(ax)
        ax.set_title(title, fontsize=10.5, color=INK)
    save(fig, "fig2_09_betti.png")


def fig_persistencia():
    # filtración por tiempo del grafo no dirigido: unión de componentes con la regla del mayor
    parent = {v: v for v in V}
    def find(x):
        while parent[x] != x: x = parent[x]
        return x
    bars0, bars1, seen = [], [], set()
    for u, v, t, _ in sorted(TX, key=lambda x: x[2]):
        if frozenset((u, v)) in seen:      # arista repetida: no cambia el complejo
            continue
        seen.add(frozenset((u, v)))
        ru, rv = find(u), find(v)
        if ru == rv:                       # cierra un ciclo: nace una clase de H1
            bars1.append(t)
        else:
            young, old = sorted((ru, rv), key=lambda x: -V.index(x))[0], sorted((ru, rv), key=lambda x: V.index(x))[0]
            parent[young] = old; bars0.append((young, t))
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    rows = [(f"$H_0$ ({v})", 0, d, BLUE) for v, d in bars0] + [("$H_0$ (A)", 0, None, BLUE)] + \
           [("$H_1$ (D–E–F)", b, None, ORANGE) for b in bars1]
    for k, (lab, b, d, c) in enumerate(rows[::-1]):
        end = d if d is not None else 10.6
        ax.plot([b, end], [k, k], color=c, lw=5, solid_capstyle="butt")
        if d is None: ax.annotate("", (10.9, k), (10.4, k), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5))
        ax.text(-0.25, k, lab, ha="right", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks([]); ax.set_xlim(-0.2, 11.2); ax.set_xticks(range(0, 11))
    ax.set_xlabel("instante τ de la filtración (horas)")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    save(fig, "fig2_10_persistencia.png")
    return bars0, bars1


# ---------------------------------------------------------------- valores para los ejemplos
def main():
    g, s = multigraph(), simple()
    u = s.as_undirected(mode="collapse")
    pr = dict(zip(V, g.pagerank(directed=True, damping=0.85)))
    btw = dict(zip(V, s.betweenness(directed=True)))
    comm = u.community_multilevel()
    print("Valores de la red de ejemplo")
    print("  grados entrada/salida:", {v: (g.indegree(IX[v]), g.outdegree(IX[v])) for v in V})
    print("  fuerza de entrada:", {v: sum(w for a, b, _, w in TX if b == v) for v in V})
    print("  PageRank:", {v: round(p, 3) for v, p in pr.items()})
    print("  intermediación:", btw)
    print("  comunidades:", [[V[i] for i in c] for c in comm], "Q =", round(comm.modularity, 3))
    print("  agrupamiento:", dict(zip(V, [round(x, 3) for x in u.transitivity_local_undirected(mode="zero")])))
    print("  número de núcleo:", dict(zip(V, u.coreness())))
    print("  beta0, beta1:", len(u.connected_components()), u.ecount() - u.vcount() + len(u.connected_components()))
    y, sc = [1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.4, 0.2]
    print("  métricas del ejemplo: AP =", round(average_precision_score(y, sc), 3), "ROC-AUC =", round(roc_auc_score(y, sc), 3))
    fig_red(); fig_instantaneas(); fig_contadores(); fig_componentes(); fig_adyacencia()
    fig_metric(pr, "fig2_06_pagerank.png", lambda x: f"{x:.3f}", "color y valor: PageRank (α = 0.85)")
    fig_metric(btw, "fig2_07_intermediacion.png", lambda x: f"{x:g}", "color y valor: intermediación")
    fig_comunidades(comm.membership); fig_betti()
    b0, b1 = fig_persistencia()
    print("  barras H0 (nodo que muere, instante):", b0, "| H1 nace en:", b1)


if __name__ == "__main__":
    main()
