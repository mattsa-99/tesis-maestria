# Bitácora Semana 2
**Fecha**: 28 de mayo de 2026
**Autor**: Mateo Soto Arango

## Resumen Semanal

Semana de diseño experimental: EDA mínima, feature engineering con control de
leakage, entrenamiento del Baseline v0.1 y error analysis con 3 mejoras
priorizadas. Se entregó la Actividad Asincrónica Semana 2 (TTG_S2_SotoArango_Mateo.docx).

## Actividades Realizadas

### 2.1 EDA mínima (notebooks/02_eda_features_baseline.py)
- Dataset proxy PaySim: 200 000 transacciones, 374 fraudes (0.187 %).
- 10/10 tests de calidad pasados (nulos, rangos, tipos, desbalance, self-loops).
- Distribución por tipo: CASH_OUT 35.1 %, CASH_IN 35.8 %, PAYMENT 18.3 %,
  TRANSFER 8.4 %, DEBIT 2.4 %. Fraude solo en CASH_OUT (0.36 %) y TRANSFER (0.70 %).
- Estadísticas de monto: legítimas media=$4 878; fraudulentas media=$3 722
  → distribuciones solapadas, el monto solo no discrimina.

### 2.2 Feature engineering sin leakage
- Features incluidas (8): type_enc, amount_log, orig_balance_log,
  dest_balance_log, dest_was_empty, amount_to_orig_ratio, insufficient_funds, step.
- Features excluidas: newbalanceOrig, newbalanceDest (leakage directo en PaySim
  sintético — en fraudes estos valores son siempre 0 por construcción del simulador).
- Estrategia: split temporal PRIMERO → features calculadas por separado en
  train (días 1–24) y aplicadas en test (días 25–30). Sin data de test en el fit.

### 2.3 Baseline v0.1
- Modelo: Random Forest, 100 árboles, class_weight='balanced', seed=42.
- Split: train=154 938 / test=45 062 (temporal, 80/20 aprox.).
- **Resultados (test, días 25-30):**
  - F1-Score : **0.4762** ← métrica principal
  - Precision: 1.0000
  - Recall   : 0.3125
  - AUC-ROC  : 0.6628
  - TN=44 966, FP=0, FN=66, TP=30
- Feature importance top-3: dest_balance_log (0.241), type_enc (0.224),
  orig_balance_log (0.115).

### 2.4 Error analysis
- 66 FN (fraudes no detectados): 43 CASH_OUT + 23 TRANSFER.
- Probabilidad predicha promedio en FN: 0.001 (IQR=[0.0, 0.0]).
  → El modelo es muy confiante de que son legítimos: no es problema de
    calibración, es falta de señal discriminativa a nivel individual.
- 0 FP: Precision perfecta (el modelo es hiperconservador).

### 2.5 3 Mejoras priorizadas
1. **Topológicas (Alto impacto)**: out-degree, betweenness, PageRank, ciclos.
   Justificación: FN en TRANSFER corresponden a transacciones cuya anomalía
   es invisible individualmente pero visible en la red (cuentas mula reciben
   de múltiples orígenes simultáneamente).
2. **Ajuste de umbral (Medio impacto)**: explorar umbral 0.3–0.4 post-topología.
   Impacto actual limitado por probas ≈ 0 en FN.
3. **Feature dest_is_merchant (Medio impacto)**: nameDest.startswith('M') →
   permite al modelo eliminar PAYMENT con certeza, enfocando capacidad en
   CASH_OUT y TRANSFER.

## Decisiones de Diseño

| Decisión | Alternativa descartada | Justificación |
|----------|------------------------|---------------|
| Split temporal días 1-24/25-30 | CV aleatorio k-fold | Datos temporales: CV rompería causalidad |
| Excluir newbalanceOrig/Dest | Incluirlas como feature | Leakage directo en datos sintéticos PaySim |
| RF class_weight='balanced' | Resampling SMOTE | Menor riesgo de overfitting; más interpretable |
| F1-Score como métrica principal | Accuracy / AUC solo | Desbalance severo (0.187 %): accuracy trivial 99.8 % |

## Próximos Pasos (Semana 3)

- [ ] Descargar dataset completo PaySim (6.3 M tx) y reemplazar proxy.
- [ ] Construir grafo dirigido sobre ventana train (networkx DiGraph).
- [ ] Calcular features topológicas: degree, betweenness, PageRank, community_id.
- [ ] Reentrenar RF con features topológicas y comparar vs. Baseline v0.1.
- [ ] Ablation study: ΔF1 por grupo de feature.
- [ ] Meta: F1 ≥ 0.80 (H1) o documentar por qué no se alcanza (H0).

## Referencias
- Lopez-Rojas et al. (2016). PaySim. EMSS.
- Gebru et al. (2021). Datasheets for Datasets. ACM.
- ISO/IEC 25012 (2008). Data Quality Model.
