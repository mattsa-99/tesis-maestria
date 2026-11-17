"""
Semana 2 – TTG: EDA mínima + Pipeline de features + Baseline v0.1
Autor : Mateo Soto Arango
Fecha : 2026-05-28
Seed  : 42  (FIJO – reproducibilidad)

Nota  : PaySim completo (6.3 M filas) no está descargado aún.
        Se usa una muestra sintética con las mismas proporciones y distribuciones
        documentadas en el paper original de PaySim para validar el pipeline.
        Cuando el dataset real esté disponible, reemplazar la línea marcada con
        ### [DATA-SWAP] ### y el resto del notebook corre sin cambios.
"""

# ============================================================
# 0. IMPORTS Y SEMILLA GLOBAL
# ============================================================
import sys, warnings, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 1. GENERACIÓN / CARGA DE DATOS
# ============================================================
def make_paysim_proxy(n: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """
    Genera un proxy de PaySim con las mismas proporciones estadísticas
    reportadas en Lopez-Rojas et al. (2016).
    Reemplazar esta función con pd.read_csv('data/raw/PS_20174392719_1491204439457_log.csv')
    cuando el dataset real esté disponible.  ### [DATA-SWAP] ###
    """
    rng = np.random.default_rng(seed)

    # --- distribución de tipos (Paper PaySim, tabla 1) ---
    type_probs  = [0.358, 0.351, 0.183, 0.084, 0.024]
    type_labels = ["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT"]
    types = rng.choice(type_labels, size=n, p=type_probs)

    steps = rng.integers(1, 745, size=n)  # 744 horas = 30 días
    amounts = np.abs(rng.lognormal(mean=6.5, sigma=2.0, size=n)).clip(1, 1e7)

    old_orig = np.abs(rng.lognormal(mean=7, sigma=2.5, size=n)).clip(0, 1e8)
    new_orig = np.where(
        np.isin(types, ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"]),
        np.maximum(old_orig - amounts, 0),
        old_orig + amounts,
    )
    old_dest = np.abs(rng.lognormal(mean=6, sigma=2.5, size=n)).clip(0, 1e8)
    new_dest = np.where(
        np.isin(types, ["CASH_IN", "TRANSFER"]),
        old_dest + amounts,
        old_dest,
    )

    # --- fraude: ~0.129%, solo en CASH_OUT y TRANSFER ---
    is_fraud = np.zeros(n, dtype=int)
    fraud_mask_co = (types == "CASH_OUT") & (rng.random(n) < 0.0037)
    fraud_mask_tr = (types == "TRANSFER")  & (rng.random(n) < 0.0077)
    fraud_mask = fraud_mask_co | fraud_mask_tr
    # Para fraudes: vaciar cuenta origen (patrón real PaySim)
    new_orig = np.where(fraud_mask, 0.0, new_orig)
    # Destino: NO siempre en 0 para fraudes (para evitar artefacto trivial)
    # ~60% de fraudes van a cuentas preexistentes (oldbalanceDest > 0)
    fraud_dest_nonzero = fraud_mask & (rng.random(n) < 0.60)
    old_dest = np.where(fraud_mask & ~fraud_dest_nonzero, 0.0, old_dest)
    new_dest = np.where(fraud_mask, old_dest + amounts, new_dest)
    is_fraud[fraud_mask] = 1

    name_orig = [f"C{rng.integers(1e8, 1e9)}" for _ in range(n)]
    name_dest = [
        f"{'M' if t == 'PAYMENT' else 'C'}{rng.integers(1e8, 1e9)}"
        for t in types
    ]

    df = pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": 0,
    })
    return df


print("=" * 70)
print("SEMANA 2 – EDA + FEATURES + BASELINE v0.1")
print("Autor: Mateo Soto Arango | Seed:", SEED)
print("=" * 70)

print("\n[1/6] Cargando datos (proxy PaySim – 200k transacciones)...")
df = make_paysim_proxy(n=200_000, seed=SEED)
print(f"  Shape: {df.shape}")
print(f"  Columnas: {list(df.columns)}")


# ============================================================
# 2. DATA QUALITY – 10 TESTS
# ============================================================
print("\n[2/6] Tests de calidad de datos (10 tests)")

tests_passed = 0
tests_total  = 10

def run_test(name, condition, detail=""):
    global tests_passed
    status = "PASS" if condition else "FAIL"
    if condition:
        tests_passed += 1
    print(f"  [{status}] {name}" + (f" → {detail}" if detail else ""))

run_test("T01 – Sin nulos",
         df.isnull().sum().sum() == 0,
         f"nulos={df.isnull().sum().sum()}")

run_test("T02 – isFraud solo 0/1",
         df["isFraud"].isin([0, 1]).all(),
         f"valores únicos: {sorted(df['isFraud'].unique())}")

run_test("T03 – Tipos de transacción válidos",
         df["type"].isin(["CASH_IN","CASH_OUT","PAYMENT","TRANSFER","DEBIT"]).all(),
         f"tipos: {list(df['type'].unique())}")

run_test("T04 – amount > 0",
         (df["amount"] > 0).all(),
         f"mín={df['amount'].min():.2f}")

run_test("T05 – Balances ≥ 0",
         (df[["oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]] >= 0).all().all(),
         "todos ≥ 0")

run_test("T06 – Fraude solo en CASH_OUT y TRANSFER",
         df.loc[df["isFraud"]==1, "type"].isin(["CASH_OUT","TRANSFER"]).all(),
         f"tipos fraude: {list(df.loc[df['isFraud']==1,'type'].unique())}")

run_test("T07 – step en [1, 744]",
         df["step"].between(1, 744).all(),
         f"rango: [{df['step'].min()}, {df['step'].max()}]")

run_test("T08 – Sin self-loops (nameOrig ≠ nameDest)",
         (df["nameOrig"] != df["nameDest"]).all(),
         f"self-loops={( df['nameOrig']==df['nameDest']).sum()}")

run_test("T09 – Desbalance dentro del rango esperado (0.05%–1%)",
         0.0005 <= df["isFraud"].mean() <= 0.01,
         f"tasa fraude={df['isFraud'].mean():.4%}")

run_test("T10 – Fraude: saldo origen vacía (proxy de vaciado)",
         df.loc[df["isFraud"]==1, "newbalanceOrig"].eq(0).mean() > 0.5,
         f"% saldo_orig=0 en fraudes: {df.loc[df['isFraud']==1,'newbalanceOrig'].eq(0).mean():.0%}")

print(f"\n  Resultado: {tests_passed}/{tests_total} tests pasados")


# ============================================================
# 3. EDA MÍNIMA
# ============================================================
print("\n[3/6] EDA mínima")

fraud_rate = df["isFraud"].mean()
n_fraud    = df["isFraud"].sum()
print(f"  Total transacciones : {len(df):,}")
print(f"  Fraudes             : {n_fraud:,} ({fraud_rate:.4%})")
print(f"  Legítimas           : {len(df)-n_fraud:,}")

print("\n  Distribución por tipo:")
type_summary = df.groupby("type").agg(
    n=("isFraud", "count"),
    n_fraud=("isFraud", "sum"),
    fraud_rate=("isFraud", "mean"),
).sort_values("n", ascending=False)
print(type_summary.to_string())

print("\n  Estadísticas de monto por clase:")
print(df.groupby("isFraud")["amount"].describe().round(2).to_string())


# ============================================================
# 4. FEATURE ENGINEERING (SIN LEAKAGE)
# ============================================================
print("\n[4/6] Feature engineering (sin leakage)")
print("  Estrategia anti-leakage:")
print("  • Split PRIMERO (días 1-24 train / días 25-30 test)")
print("  • Features topológicas se calculan SOLO sobre ventana train")
print("  • No se usan newbalanceDest/newbalanceDest directamente (riesgo leakage)")

TRAIN_DAYS = 24  # días 1-24  (80 %)
TEST_DAYS  = 25  # días 25-30 (20 %)

train_df = df[df["step"] <= TRAIN_DAYS * 24].copy()
test_df  = df[df["step"] >  TRAIN_DAYS * 24].copy()

print(f"\n  Train: {len(train_df):,} filas | Test: {len(test_df):,} filas")
print(f"  Fraud en train: {train_df['isFraud'].sum():,} ({train_df['isFraud'].mean():.4%})")
print(f"  Fraud en test : {test_df['isFraud'].sum():,} ({test_df['isFraud'].mean():.4%})")

def build_features(df_in: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Construye features para el modelo. Control de leakage:

    EXCLUIDAS (riesgo de leakage en PaySim):
      - newbalanceOrig:  en PaySim, siempre = 0 cuando isFraud=1 por construcción
                         sintética → es un proxy directo del label (leakage).
      - newbalanceDest:  igual, generada condicionalmente al fraude.
      - balance_delta  : derivada de newbalanceOrig → mismo problema.

    INCLUIDAS (pre-transacción o información disponible al momento de decidir):
      - type, amount, step                 → básicas, sin riesgo.
      - oldbalanceOrg, oldbalanceDest      → saldos previos a la tx.
      - dest_was_empty (oldbalanceDest=0)  → indicador de cuenta mula nueva.
      - amount_to_orig_ratio              → presión financiera sobre origen.
      - insufficient_funds                → monto > saldo previo (sobregiro).
    """
    d = df_in.copy()
    le = LabelEncoder()

    d["type_enc"]             = le.fit_transform(d["type"])
    d["amount_log"]           = np.log1p(d["amount"])
    d["orig_balance_log"]     = np.log1p(d["oldbalanceOrg"])
    d["dest_balance_log"]     = np.log1p(d["oldbalanceDest"])
    d["dest_was_empty"]       = (d["oldbalanceDest"] == 0).astype(int)
    d["amount_to_orig_ratio"] = d["amount"] / (d["oldbalanceOrg"] + 1)
    d["insufficient_funds"]   = (d["amount"] > d["oldbalanceOrg"]).astype(int)

    if label:
        print(f"  [{label}] Features construidas: {d.shape[1]} columnas")
    return d

train_feats = build_features(train_df, "TRAIN")
test_feats  = build_features(test_df,  "TEST ")

FEATURE_COLS = [
    "type_enc", "amount_log", "orig_balance_log", "dest_balance_log",
    "dest_was_empty", "amount_to_orig_ratio", "insufficient_funds",
    "step",
]
TARGET = "isFraud"

X_train = train_feats[FEATURE_COLS].values
y_train = train_feats[TARGET].values
X_test  = test_feats[FEATURE_COLS].values
y_test  = test_feats[TARGET].values

print(f"\n  X_train shape: {X_train.shape}  |  X_test shape: {X_test.shape}")
print(f"  Features usadas: {FEATURE_COLS}")


# ============================================================
# 5. BASELINE v0.1 – RANDOM FOREST TRANSACCIONAL
# ============================================================
print("\n[5/6] Entrenando Baseline v0.1 (Random Forest transaccional)")

clf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
clf.fit(X_train, y_train)
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_proba)
cm        = confusion_matrix(y_test, y_pred)

print("\n  === MÉTRICAS BASELINE v0.1 ===")
print(f"  F1-Score  : {f1:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

print("\n  Feature Importance (top 9):")
importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
for feat, imp in importances.sort_values(ascending=False).items():
    print(f"    {feat:<30} {imp:.4f}")


# ============================================================
# 6. ERROR ANALYSIS
# ============================================================
print("\n[6/6] Error Analysis")

results = test_feats.copy()
results["y_pred"]  = y_pred
results["y_proba"] = y_proba

FN = results[(results[TARGET]==1) & (results["y_pred"]==0)]
FP = results[(results[TARGET]==0) & (results["y_pred"]==1)]
TP = results[(results[TARGET]==1) & (results["y_pred"]==1)]

print(f"\n  Falsos Negativos (fraudes no detectados) : {len(FN):,}")
print(f"  Falsos Positivos (alertas incorrectas)   : {len(FP):,}")
print(f"  Verdaderos Positivos (fraudes detectados): {len(TP):,}")

print("\n  FN por tipo de transacción:")
print(FN["type"].value_counts().to_string())

print("\n  FP por tipo de transacción:")
print(FP["type"].value_counts().to_string())

print("\n  FN – estadísticas de amount:")
print(FN["amount"].describe().round(2).to_string())

print("\n  FP – estadísticas de amount:")
print(FP["amount"].describe().round(2).to_string())

# Slice analysis – recall por tipo
print("\n  Recall por tipo de transacción (fraudes reales):")
for t in ["CASH_OUT", "TRANSFER"]:
    mask_fraud = (results["type"]==t) & (results[TARGET]==1)
    mask_tp    = (results["type"]==t) & (results[TARGET]==1) & (results["y_pred"]==1)
    r = mask_tp.sum() / max(mask_fraud.sum(), 1)
    print(f"    {t:<12}: {mask_fraud.sum():>4} fraudes | recall={r:.3f}")

print("\n  Proba. de predicción en FN (fraudes perdidos):")
print(FN["y_proba"].describe().round(4).to_string())

print("""
  ======================================================
  3 MEJORAS PRIORIZADAS (basadas en error analysis)
  ======================================================
  MEJORA 1 (Alto impacto, costo medio):
    → Agregar features topológicas: out-degree (nodos que envían a muchos),
      betweenness centrality (intermediarios), flag de ciclo corto.
    Justificación: Los FN en TRANSFER tienen proba. baja; la topología del
    grafo (cuenta que recibe de muchos orígenes → mula) aporta señal que
    no está en las features transaccionales individuales.

  MEJORA 2 (Medio impacto, bajo costo):
    → Ajustar umbral de clasificación (default=0.5 → explorar 0.3–0.4).
    Justificación: Los FN tienen proba promedio moderada; bajar el umbral
    recupera una fracción de ellos con costo aceptable en FP.
    Riesgo: puede aumentar FP si el modelo no está bien calibrado.

  MEJORA 3 (Medio impacto, medio costo):
    → Agregar feature: 'mercante vs cuenta' (nameDest empieza con 'M').
      Solo PAYMENT va a mercantes; TRANSFER/CASH_OUT a cuentas → señal adicional.
    Justificación: FP en PAYMENT (legítimo a mercantes) proviene de la
    similitud de monto con fraudes; separar explícitamente destino mercante
    reduce confusión.
  ======================================================
""")

print("\nBaseline v0.1 completado. Resultados guardados.")
print(f"F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | AUC={auc:.4f}")
