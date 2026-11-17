"""Features causales para IBM AML (HI-Small / LI-Small).

Principio igual que en PaySim: la feature de una transacción en el instante ``t``
solo usa transacciones estrictamente anteriores (contadores) o el grafo acumulado
hasta el inicio del bloque de 6 h previo (instantáneas).

Grupos:
    * TXN       -> transaccionales (baseline B)
    * COUNTERS  -> grado/fuerza previos, fan-in/out 24 h, repetición del par,
                   arista inversa previa (ciclo de longitud 2)
    * SNAP      -> PageRank, componente fuerte (ciclos), cierre de ciclo de
                   longitud 3, comunidad (Louvain), intermediación (caminos <= 4)
"""
from __future__ import annotations

import sys
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp

IBM_DIR = Path("data/raw/_kagglehub/datasets/ealtman2019/"
               "ibm-transactions-for-anti-money-laundering-aml/versions/8")
CUTOFF_DATE = "2022-09-11"          # se excluye lo posterior al 2022-09-10 (3.8)
BLOCK_MIN = 6 * 60                   # instantáneas cada 6 h
BTW_CUTOFF = 4

TXN = ["amt_paid_log", "amt_rec_log", "cur_paid", "cur_rec", "same_currency",
       "fmt", "same_bank", "self_loop", "hour", "weekday"]
COUNTERS = ["src_out_prev", "src_in_prev", "dst_in_prev", "dst_out_prev",
            "src_out_amt_prev", "dst_in_amt_prev", "src_fanout_24h", "dst_fanin_24h",
            "src_gap", "dst_gap", "pair_prev", "reverse_edge_prev"]
SNAP = ["src_pagerank", "dst_pagerank", "src_scc_size", "dst_scc_size", "same_scc",
        "closes_3cycle", "src_comm_size", "dst_comm_size", "src_btw", "dst_btw"]


def load_ibm(name: str = "HI-Small", truncate: bool = True) -> pd.DataFrame:
    df = pd.read_csv(IBM_DIR / f"{name}_Trans.csv", dtype={1: str, 2: str, 3: str, 4: str})
    df.columns = ["ts", "fb", "fa", "tb", "ta", "amt_rec", "cur_rec", "amt_paid",
                  "cur_paid", "fmt", "y"]
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    if truncate:
        df = df[df["ts"] < CUTOFF_DATE].reset_index(drop=True)
    codes, _ = pd.factorize(pd.concat([df["fb"] + "_" + df["fa"], df["tb"] + "_" + df["ta"]],
                                      ignore_index=True))
    n = len(df)
    df["src"], df["dst"] = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    df["minute"] = ((df["ts"] - df["ts"].iloc[0]).dt.total_seconds() // 60).astype(np.int64)
    return df


def txn_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["amt_paid_log"] = np.log1p(df["amt_paid"])
    out["amt_rec_log"] = np.log1p(df["amt_rec"])
    cur = pd.factorize(pd.concat([df["cur_paid"], df["cur_rec"]], ignore_index=True))[0]
    out["cur_paid"], out["cur_rec"] = cur[:len(df)], cur[len(df):]
    out["same_currency"] = (df["cur_paid"] == df["cur_rec"]).astype(np.int8)
    out["fmt"] = pd.factorize(df["fmt"], sort=True)[0]
    out["same_bank"] = (df["fb"] == df["tb"]).astype(np.int8)
    out["self_loop"] = (df["src"] == df["dst"]).astype(np.int8)
    out["hour"] = df["ts"].dt.hour
    out["weekday"] = df["ts"].dt.weekday
    return out


def _window_count(node: np.ndarray, minute: np.ndarray, window: int) -> np.ndarray:
    """Eventos del mismo nodo en [t - window, t) (minutos estrictamente anteriores)."""
    key = node * 10**7 + minute
    srt = np.sort(key)
    return np.searchsorted(srt, key, side="left") - np.searchsorted(srt, key - window, side="left")


def counter_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    src, dst, minute = df["src"].to_numpy(), df["dst"].to_numpy(), df["minute"].to_numpy()
    node = np.empty(2 * n, dtype=np.int64); node[0::2], node[1::2] = src, dst
    is_out = np.zeros(2 * n, dtype=np.int32); is_out[0::2] = 1
    ev = pd.DataFrame({"node": node, "out": is_out, "inn": 1 - is_out})
    g = ev.groupby("node", sort=False)
    out_prev = (g["out"].cumsum() - ev["out"]).to_numpy()
    in_prev = (g["inn"].cumsum() - ev["inn"]).to_numpy()

    res = pd.DataFrame(index=df.index)
    res["src_out_prev"], res["src_in_prev"] = out_prev[0::2], in_prev[0::2]
    res["dst_out_prev"], res["dst_in_prev"] = out_prev[1::2], in_prev[1::2]
    amt = df["amt_paid"]
    res["src_out_amt_prev"] = np.log1p(amt.groupby(df["src"]).cumsum() - amt)
    res["dst_in_amt_prev"] = np.log1p(amt.groupby(df["dst"]).cumsum() - amt)
    res["src_fanout_24h"] = _window_count(src, minute, 24 * 60)
    res["dst_fanin_24h"] = _window_count(dst, minute, 24 * 60)
    m = df["minute"]
    res["src_gap"] = (m - m.groupby(df["src"]).shift(1)).fillna(-1)
    res["dst_gap"] = (m - m.groupby(df["dst"]).shift(1)).fillna(-1)

    n_nodes = int(max(src.max(), dst.max())) + 1
    pair = src * n_nodes + dst
    res["pair_prev"] = pd.Series(pair).groupby(pair).cumcount().to_numpy()
    first_min = pd.Series(minute).groupby(pair).min()            # primera vez de cada arista
    rev = pd.Series(dst * n_nodes + src).map(first_min).to_numpy()
    res["reverse_edge_prev"] = (~np.isnan(rev) & (rev < minute)).astype(np.int8)
    return res


def snapshot_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    src, dst, minute = df["src"].to_numpy(), df["dst"].to_numpy(), df["minute"].to_numpy()
    n_nodes = int(max(src.max(), dst.max())) + 1
    block = minute // BLOCK_MIN
    cols = {c: np.zeros(len(df), dtype=np.float32) for c in SNAP}
    for b in range(1, int(block.max()) + 1):
        rows = np.flatnonzero(block == b)
        if rows.size == 0:
            continue
        hist = block < b
        e = np.unique(src[hist] * n_nodes + dst[hist])
        es, ed = e // n_nodes, e % n_nodes
        keep = es != ed
        es, ed = es[keep], ed[keep]
        g = ig.Graph(n=n_nodes, edges=np.column_stack([es, ed]).tolist(), directed=True)
        pr = np.asarray(g.pagerank(directed=True), dtype=np.float32)
        scc = np.asarray(g.connected_components(mode="strong").membership)
        scc_size = np.bincount(scc)[scc].astype(np.float32)
        comm = np.asarray(g.as_undirected(mode="collapse").community_multilevel().membership)
        comm_size = np.bincount(comm)[comm].astype(np.float32)
        btw = np.asarray(g.betweenness(directed=True, cutoff=BTW_CUTOFF), dtype=np.float32)

        A = sp.csr_matrix((np.ones(len(es), dtype=np.float32), (es, ed)), shape=(n_nodes, n_nodes))
        AT = A.T.tocsr()
        u, v = src[rows], dst[rows]
        # cierre de ciclo de longitud 3: existe w con v -> w y w -> u
        closes = np.asarray(A[v].multiply(AT[u]).sum(axis=1)).ravel() > 0

        cols["src_pagerank"][rows], cols["dst_pagerank"][rows] = pr[u], pr[v]
        cols["src_scc_size"][rows], cols["dst_scc_size"][rows] = scc_size[u], scc_size[v]
        cols["same_scc"][rows] = ((scc[u] == scc[v]) & (scc_size[u] > 1)).astype(np.float32)
        cols["closes_3cycle"][rows] = closes.astype(np.float32)
        cols["src_comm_size"][rows], cols["dst_comm_size"][rows] = comm_size[u], comm_size[v]
        cols["src_btw"][rows], cols["dst_btw"][rows] = btw[u], btw[v]
        if verbose:
            print(f"  bloque {b:2d}: aristas únicas={len(es):,} filas={rows.size:,} "
                  f"nodos en ciclos={int((scc_size > 1).sum()):,}", flush=True)
    return pd.DataFrame(cols, index=df.index)


def build_ibm(name: str = "HI-Small", truncate: bool = True) -> pd.DataFrame:
    tag = name + ("" if truncate else "_full")
    cache = Path(f"data/processed/ibm_{tag}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_ibm(name, truncate)
    feats = pd.concat([txn_features(df), counter_features(df), snapshot_features(df)], axis=1)
    feats["y"] = df["y"].to_numpy()
    feats["minute"] = df["minute"].to_numpy()
    n = len(df)
    split = np.full(n, "test", dtype=object)
    split[: int(0.6 * n)] = "train"
    split[int(0.6 * n): int(0.8 * n)] = "val"
    feats["split"] = split
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache)
    return feats


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "HI-Small"
    truncate = "--full" not in sys.argv
    out = build_ibm(name, truncate)
    print(out.shape)
    print(out.groupby("split")["y"].agg(["size", "sum"]))
    print(out.groupby("y")[COUNTERS + SNAP].mean().T.to_string())
