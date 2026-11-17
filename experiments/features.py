"""Feature engineering causal para PaySim.

Principio: la feature de una transacción en el paso ``t`` solo usa información
de transacciones anteriores (sin leakage temporal), tanto en train como en test.

Grupos de features:
    * base        -> transaccionales (baseline v0.1)
    * merchant    -> ``dest_is_merchant`` (mejora no topológica, baseline v0.2)
    * degree      -> grado y fuerza (in/out) acumulados previos, causales
    * snapshot    -> PageRank, tamaño de componente, betweenness, comunidad y
                     ciclos calculados sobre el grafo acumulado hasta el día
                     anterior (rezago de 24 h)
"""
from __future__ import annotations

from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

RAW = Path("data/raw/_kagglehub/datasets/mtalaltariq/paysim-data/versions/1/paysim dataset.csv")
TYPE_CODES = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}

BASE = ["type_enc", "amount_log", "orig_balance_log", "dest_balance_log",
        "dest_was_empty", "amount_to_orig_ratio", "insufficient_funds", "step"]
MERCHANT = ["dest_is_merchant"]
DEGREE = ["dest_in_deg_prev", "dest_out_deg_prev", "orig_out_deg_prev",
          "orig_in_deg_prev", "dest_in_amt_prev", "dest_last_gap"]
SNAPSHOT = ["dest_pagerank", "orig_pagerank", "dest_wcc_size", "orig_wcc_size",
            "dest_betweenness", "orig_betweenness", "dest_comm_size"]
# Se calculan (dest_in_cycle, orig_in_cycle) pero valen 0 en todo PaySim: el grafo
# no contiene ciclos dirigidos. Se reportan como hallazgo y no entran al modelo.
CYCLE = ["dest_in_cycle", "orig_in_cycle"]


def load_raw(path: Path = RAW) -> pd.DataFrame:
    """Carga PaySim y asigna ids enteros compartidos para origen y destino."""
    df = pd.read_csv(path)
    codes, _ = pd.factorize(pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True))
    n = len(df)
    df["orig_id"] = codes[:n].astype(np.int64)
    df["dest_id"] = codes[n:].astype(np.int64)
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(np.int8)
    assert df["step"].is_monotonic_increasing, "se asume el archivo ordenado por step"
    return df


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features transaccionales. Excluye newbalance* (leakage de PaySim)."""
    out = pd.DataFrame(index=df.index)
    out["type_enc"] = df["type"].map(TYPE_CODES).astype(np.int8)
    out["amount_log"] = np.log1p(df["amount"])
    out["orig_balance_log"] = np.log1p(df["oldbalanceOrg"])
    out["dest_balance_log"] = np.log1p(df["oldbalanceDest"])
    out["dest_was_empty"] = (df["oldbalanceDest"] == 0).astype(np.int8)
    out["amount_to_orig_ratio"] = (df["amount"] / (df["oldbalanceOrg"] + 1.0)).clip(upper=1e3)
    out["insufficient_funds"] = (df["amount"] > df["oldbalanceOrg"]).astype(np.int8)
    out["step"] = df["step"]
    out["dest_is_merchant"] = df["dest_is_merchant"]
    return out


def degree_features(df: pd.DataFrame) -> pd.DataFrame:
    """Grado / fuerza acumulados *previos* a cada transacción (causales)."""
    n = len(df)
    # Eventos en orden temporal: 2*i = rol origen, 2*i+1 = rol destino de la fila i.
    node = np.empty(2 * n, dtype=np.int64)
    node[0::2], node[1::2] = df["orig_id"].to_numpy(), df["dest_id"].to_numpy()
    is_out = np.zeros(2 * n, dtype=np.int32)
    is_out[0::2] = 1
    ev = pd.DataFrame({"node": node, "out": is_out, "inn": 1 - is_out})
    g = ev.groupby("node", sort=False)
    out_prev = (g["out"].cumsum() - ev["out"]).to_numpy()   # salidas previas del nodo
    in_prev = (g["inn"].cumsum() - ev["inn"]).to_numpy()    # entradas previas del nodo

    res = pd.DataFrame(index=df.index)
    res["orig_out_deg_prev"] = out_prev[0::2]
    res["orig_in_deg_prev"] = in_prev[0::2]
    res["dest_out_deg_prev"] = out_prev[1::2]
    res["dest_in_deg_prev"] = in_prev[1::2]
    amt = df["amount"]
    by_dest = amt.groupby(df["dest_id"])
    res["dest_in_amt_prev"] = np.log1p(by_dest.cumsum() - amt)
    last = df["step"].groupby(df["dest_id"]).shift(1)
    res["dest_last_gap"] = (df["step"] - last).fillna(-1)       # -1 = sin historial
    return res


def _day_of(step: pd.Series) -> pd.Series:
    return (step - 1) // 24 + 1  # 1..31


def snapshot_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Métricas de grafo sobre el grafo acumulado hasta el día anterior."""
    day = _day_of(df["step"]).to_numpy()
    n_nodes = int(max(df["orig_id"].max(), df["dest_id"].max())) + 1
    src, dst = df["orig_id"].to_numpy(), df["dest_id"].to_numpy()
    cols = {c: np.zeros(len(df), dtype=np.float32) for c in SNAPSHOT + CYCLE}

    for d in range(2, int(day.max()) + 1):
        mask_hist = day < d
        rows = np.flatnonzero(day == d)
        if rows.size == 0:
            continue
        edges = np.column_stack([src[mask_hist], dst[mask_hist]])
        g = ig.Graph(n=n_nodes, edges=edges.tolist(), directed=True)
        pr = np.asarray(g.pagerank(directed=True), dtype=np.float32)

        comp = np.asarray(g.connected_components(mode="weak").membership)
        comp_size = np.bincount(comp)
        indeg, outdeg = np.asarray(g.indegree()), np.asarray(g.outdegree())
        bridge = np.flatnonzero((indeg > 0) & (outdeg > 0))  # únicos nodos con betweenness > 0

        btw = np.zeros(n_nodes, dtype=np.float32)
        comm_size = comp_size[comp].astype(np.float32)
        in_cycle = np.zeros(n_nodes, dtype=np.float32)
        if bridge.size:
            keep = np.flatnonzero(np.isin(comp, np.unique(comp[bridge])))
            sub = g.induced_subgraph(keep)
            btw[keep] = np.asarray(sub.betweenness(directed=True), dtype=np.float32)
            und = sub.as_undirected(mode="collapse")
            memb = np.asarray(und.community_multilevel().membership)
            sizes = np.bincount(memb)
            comm_size[keep] = sizes[memb]
            scc = np.asarray(sub.connected_components(mode="strong").membership)
            in_cycle[keep] = (np.bincount(scc)[scc] > 1).astype(np.float32)

        o, de = src[rows], dst[rows]
        cols["dest_pagerank"][rows], cols["orig_pagerank"][rows] = pr[de], pr[o]
        cols["dest_wcc_size"][rows], cols["orig_wcc_size"][rows] = comp_size[comp[de]], comp_size[comp[o]]
        cols["dest_betweenness"][rows], cols["orig_betweenness"][rows] = btw[de], btw[o]
        cols["dest_comm_size"][rows] = comm_size[de]
        cols["dest_in_cycle"][rows], cols["orig_in_cycle"][rows] = in_cycle[de], in_cycle[o]
        if verbose:
            print(f"  día {d:2d}: aristas={edges.shape[0]:,} puentes={bridge.size}", flush=True)
    return pd.DataFrame(cols, index=df.index)


def build_all(cache: Path = Path("data/processed/features.parquet")) -> pd.DataFrame:
    """Construye (o lee de caché) el dataset completo de features + etiqueta."""
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_raw()
    feats = pd.concat([base_features(df), degree_features(df), snapshot_features(df)], axis=1)
    feats["isFraud"] = df["isFraud"].to_numpy()
    feats["type"] = df["type"].to_numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache)
    return feats


if __name__ == "__main__":
    out = build_all()
    print(out.shape)
    print(out.describe().T.to_string())
