"""Tests de las features causales de experiments/features.py."""
import numpy as np
import pandas as pd

from experiments.features import (BASE, CYCLE, DEGREE, SNAPSHOT, base_features,
                                  degree_features, snapshot_features)


def _toy() -> pd.DataFrame:
    # A->C (paso 1), B->C (paso 2), C->D (paso 30), A->C (paso 50)
    df = pd.DataFrame({
        "step": [1, 2, 30, 50],
        "type": ["TRANSFER", "TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount": [10.0, 20.0, 30.0, 40.0],
        "oldbalanceOrg": [10.0, 50.0, 30.0, 100.0],
        "oldbalanceDest": [0.0, 10.0, 0.0, 30.0],
        "nameDest": ["C", "C", "D", "C"],
        "orig_id": [0, 1, 2, 0], "dest_id": [2, 2, 3, 2],
        "dest_is_merchant": [0, 0, 0, 0],
    })
    return df


def test_degree_features_are_strictly_past():
    f = degree_features(_toy())
    assert f["dest_in_deg_prev"].tolist() == [0, 1, 0, 2]       # C recibe: 0,1,(D:0),2 previos
    assert f["orig_out_deg_prev"].tolist() == [0, 0, 0, 1]      # A ya había enviado una vez
    assert f["dest_out_deg_prev"].tolist() == [0, 0, 0, 1]      # C actuó como origen en paso 30
    assert f["dest_last_gap"].tolist() == [-1, 1, -1, 48]       # -1 = sin historial


def test_degree_features_ignore_future():
    df = _toy()
    full = degree_features(df)
    head = degree_features(df.iloc[:3].reset_index(drop=True))
    for col in DEGREE:
        assert np.allclose(full[col].iloc[:3].to_numpy(), head[col].to_numpy())


def test_base_features_exclude_newbalance_and_leakage():
    f = base_features(_toy())
    assert not any(c.startswith("newbalance") for c in f.columns)
    assert set(BASE) <= set(f.columns)


def test_snapshot_uses_only_previous_days():
    df = _toy()
    full = snapshot_features(df, verbose=False)
    # Transacciones del día 1 (pasos 1 y 2) no pueden ver ninguna arista.
    assert (full[SNAPSHOT].iloc[:2].to_numpy() == 0).all()
    # Una arista añadida en el futuro no cambia lo observado antes.
    # Misma población de nodos (el PageRank se normaliza por |V|): solo cambia el futuro.
    extra = pd.concat([df, df.iloc[[0]].assign(step=700, orig_id=1, dest_id=3)], ignore_index=True)
    again = snapshot_features(extra, verbose=False)
    assert np.allclose(full[SNAPSHOT].to_numpy(), again[SNAPSHOT].iloc[:4].to_numpy())


def test_snapshot_detects_no_cycles_in_dag():
    df = _toy()
    out = snapshot_features(df, verbose=False)
    assert (out[CYCLE] == 0).all().all()
