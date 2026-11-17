"""Tests de causalidad de experiments/ibm_features.py."""
import numpy as np
import pandas as pd

from experiments.ibm_features import (BLOCK_MIN, COUNTERS, SNAP, counter_features,
                                      snapshot_features)


def _toy() -> pd.DataFrame:
    # 0->1 (t=0), 1->2 (t=10), 2->0 (t=20, cierra ciclo 0->1->2->0), 1->0 (t=30, arista inversa),
    # 0->1 (t=2000, repetición del par), luego 2->0 en un bloque posterior (cierre visible en instantánea)
    return pd.DataFrame({
        "src": [0, 1, 2, 1, 0, 2],
        "dst": [1, 2, 0, 0, 1, 0],
        "minute": [0, 10, 20, 30, 2000, 2 * BLOCK_MIN + 5],
        "amt_paid": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


def test_counters_strictly_past():
    f = counter_features(_toy())
    assert f["src_out_prev"].tolist() == [0, 0, 0, 1, 1, 1]
    assert f["dst_in_prev"].tolist() == [0, 0, 0, 1, 1, 2]
    assert f["pair_prev"].tolist() == [0, 0, 0, 0, 1, 1]
    # 1->0 en t=30: la inversa 0->1 existe desde t=0; 0->1 en t=2000: la inversa 1->0 existe desde t=30
    assert f["reverse_edge_prev"].tolist() == [0, 0, 0, 1, 1, 0]
    # fan-out en las 24 h previas (minutos estrictamente anteriores)
    assert f["src_fanout_24h"].tolist() == [0, 0, 0, 1, 0, 1]


def test_counters_ignore_future_rows():
    df = _toy()
    full, head = counter_features(df), counter_features(df.iloc[:4].reset_index(drop=True))
    for c in COUNTERS:
        if c == "reverse_edge_prev":
            continue  # se valida aparte: depende solo de aristas con minuto anterior
        assert np.allclose(full[c].iloc[:4].to_numpy(), head[c].to_numpy()), c


def test_snapshot_lagged_by_block():
    df = _toy()
    s = snapshot_features(df, verbose=False)
    # Bloque 0 no ve ningún grafo
    assert (s.loc[df["minute"] < BLOCK_MIN, SNAP].to_numpy() == 0).all()
    # La última fila (bloque 2) ve el ciclo 0->1->2->0 completo: cierra ciclo de 3 y misma SCC
    last = s.iloc[-1]
    assert last["closes_3cycle"] == 1.0
    assert last["same_scc"] == 1.0


def test_snapshot_ignores_future_blocks():
    df = _toy()
    a = snapshot_features(df, verbose=False)
    extra = pd.concat([df, pd.DataFrame({"src": [2], "dst": [1], "minute": [10 * BLOCK_MIN],
                                         "amt_paid": [1.0]})], ignore_index=True)
    b = snapshot_features(extra, verbose=False)
    assert np.allclose(a[SNAP].to_numpy(), b[SNAP].iloc[:len(df)].to_numpy())
