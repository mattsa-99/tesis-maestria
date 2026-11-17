# Anexos

## Anexo A. Glosario

| Término | Definición | Término en inglés |
|---|---|---|
| Arista inversa previa | Indicador de que la transacción $`v \to u`$ ocurrió antes que $`u \to v`$ (ciclo de longitud 2) | *reverse edge* |
| *Bootstrap* pareado | Remuestreo con reemplazo de las transacciones de prueba, evaluando dos modelos sobre las mismas muestras para estimar la distribución de su diferencia | *paired bootstrap* |
| Componente fuerte | Conjunto maximal de nodos mutuamente alcanzables respetando la dirección | *strongly connected component* |
| Conjunto de datos | Colección de transacciones con sus etiquetas | *dataset* |
| Exhaustividad | Proporción de positivos reales que el modelo detecta | *recall* |
| *Fan-in* / *fan-out* | Número de transacciones recibidas / enviadas por una cuenta en una ventana de tiempo | — |
| Fuga de información | Uso, en una variable, de información no disponible al momento de la predicción o que codifica la etiqueta | *leakage* |
| Instantánea | Grafo reconstruido con las transacciones anteriores al inicio de un bloque de tiempo | *snapshot* |
| Intermediación | Fracción de caminos mínimos que pasan por un nodo | *betweenness centrality* |
| Modelo de referencia | Modelo con solo variables transaccionales, contra el que se mide la mejora | *baseline* |
| Partición | División de los datos en ajuste, validación y prueba | *split* |
| PR-AUC | Área bajo la curva precisión-exhaustividad (precisión promedio) | *average precision* |
| Precisión | Proporción de alertas que son positivos reales | *precision* |
| Señuelo | Estructura inyectada idéntica a un patrón fraudulento, pero etiquetada como legítima | *decoy* |
| Variable | Atributo de entrada del modelo | *feature* |
| Variable causal | Variable que, para una transacción en $`t`$, usa solo información anterior a $`t`$ | — |

## Anexo B. Catálogo de variables

### B.1 PaySim

| Variable | Grupo | Definición |
|---|---|---|
| `type_enc` | Base | Tipo de transacción codificado |
| `amount_log` | Base | $`\log(1 + \text{monto})`$ |
| `orig_balance_log` | Base | $`\log(1 + \text{saldo previo del origen})`$ |
| `dest_balance_log` | Base | $`\log(1 + \text{saldo previo del destino})`$ |
| `dest_was_empty` | Base | Saldo previo del destino igual a cero |
| `amount_to_orig_ratio` | Base | Monto / (saldo previo del origen + 1), truncado en 1 000 |
| `insufficient_funds` | Base | Monto mayor que el saldo previo del origen |
| `step` | Base | Paso temporal (hora) |
| `dest_is_merchant` | Comercio | El destino es un comercio (nombre con prefijo "M") |
| `orig_out_deg_prev`, `orig_in_deg_prev` | Grado (estructural) | Transacciones previas enviadas / recibidas por el origen |
| `dest_out_deg_prev`, `dest_in_deg_prev` | Grado (estructural) | Transacciones previas enviadas / recibidas por el destino |
| `dest_in_amt_prev` | Grado (estructural) | $`\log(1 + \text{suma de montos recibidos previamente por el destino})`$ |
| `dest_last_gap` | Grado (temporal) | Pasos desde la última transacción recibida por el destino (−1 si no hay) |
| `orig_pagerank`, `dest_pagerank` | Instantánea (global) | PageRank en el grafo del día anterior |
| `orig_wcc_size`, `dest_wcc_size` | Instantánea (global) | Tamaño de la componente débil |
| `orig_betweenness`, `dest_betweenness` | Instantánea (global) | Intermediación exacta |
| `dest_comm_size` | Instantánea (global) | Tamaño de la comunidad de Louvain del destino |
| `orig_in_cycle`, `dest_in_cycle` | Instantánea (global) | Pertenencia a una componente fuerte de más de un nodo (constantes en cero; no se usan) |

### B.2 IBM AML

| Variable | Grupo | Definición |
|---|---|---|
| `amt_paid_log`, `amt_rec_log` | Transaccional | $`\log(1 + \text{monto pagado / recibido})`$ |
| `cur_paid`, `cur_rec` | Transaccional | Moneda de pago / recepción |
| `same_currency` | Transaccional | Monedas de pago y recepción iguales |
| `fmt` | Transaccional | Formato de pago (ACH, cheque, tarjeta, efectivo, transferencia, Bitcoin, reinversión) |
| `same_bank` | Transaccional | Origen y destino en el mismo banco |
| `self_loop` | Transaccional | Autotransferencia |
| `hour`, `weekday` | Transaccional | Hora del día y día de la semana |
| `src_out_prev`, `src_in_prev` | Contador estructural | Transacciones previas enviadas / recibidas por el origen |
| `dst_in_prev`, `dst_out_prev` | Contador estructural | Transacciones previas recibidas / enviadas por el destino |
| `src_out_amt_prev`, `dst_in_amt_prev` | Contador estructural | Fuerza previa de salida del origen / de entrada del destino (log) |
| `src_fanout_24h`, `dst_fanin_24h` | Contador estructural | *Fan-out* del origen / *fan-in* del destino en las 24 h previas |
| `pair_prev` | Contador estructural | Transacciones previas del mismo par origen → destino |
| `reverse_edge_prev` | Contador estructural | Arista inversa previa (ciclo de longitud 2) |
| `src_gap`, `dst_gap` | Contador temporal | Minutos desde la última transacción del origen / del destino (−1 si no hay) |
| `src_pagerank`, `dst_pagerank` | Instantánea (global) | PageRank en el grafo del bloque de 6 h anterior |
| `src_scc_size`, `dst_scc_size` | Instantánea (global) | Tamaño de la componente fuerte |
| `same_scc` | Instantánea (global) | Origen y destino en la misma componente fuerte de más de un nodo |
| `closes_3cycle` | Instantánea (estructural) | Existe $`w`$ con $`v \to w \to u`$ (la transacción cierra un ciclo de longitud 3) |
| `src_comm_size`, `dst_comm_size` | Instantánea (global) | Tamaño de la comunidad de Louvain |
| `src_btw`, `dst_btw` | Instantánea (global) | Intermediación restringida a caminos de longitud ≤ 4 |

### B.3 Elliptic

| Variable | Grupo | Definición |
|---|---|---|
| `L0`–`L92` | Locales | 93 variables locales anonimizadas del conjunto |
| `AF0`–`AF71` | Agregadas | 72 estadísticos de los vecinos provistos por el conjunto |
| `in_deg`, `out_deg` | Grafo | Grado de entrada y de salida |
| `pagerank` | Grafo | PageRank |
| `wcc_size` | Grafo | Tamaño de la componente débil (equivale al periodo, porque no hay aristas entre periodos) |
| `btw` | Grafo | Intermediación restringida a caminos de longitud ≤ 4 |
| `comm_size` | Grafo | Tamaño de la comunidad de Louvain |
| `coreness` | Grafo | Número de núcleo |
| `clustering` | Grafo | Coeficiente de agrupamiento local (no dirigido) |
| `in_reach2`, `out_reach2` | Grafo | Nodos alcanzables en dos pasos de entrada / de salida |

## Anexo C. Código y reproducibilidad

El código, las pruebas y los resultados de esta tesis están en el repositorio público <https://github.com/mattsa-99/tesis-maestria>. **Versión del código usada para los resultados:** *commit* `39835fc` (`39835fc8b31d1850592ab5aabec1dcc34bbe2bd0`), disponible en <https://github.com/mattsa-99/tesis-maestria/tree/39835fc8b31d1850592ab5aabec1dcc34bbe2bd0>.

### C.1 Entorno

Los experimentos se ejecutaron en macOS (procesador Apple, 24 GB de RAM) con Python 3.14.8 y las siguientes librerías: igraph 1.0.0 (Csardi y Nepusz, 2006), scikit-learn 1.8.0 (Pedregosa et al., 2011), numpy 2.3.5, pandas 3.0.0, scipy 1.17.0, pyarrow 26.0.1, matplotlib 3.10.8 y kagglehub 1.0.3. Las pruebas usan pytest 9.0.2. Las semillas aleatorias son fijas (42, 43 y 44).

### C.2 Datos

Los conjuntos de datos no se incluyen en el repositorio por su tamaño (≈ 2.2 GB) y se descargan con `kagglehub`:

| Conjunto | Identificador en Kaggle | Archivos |
|---|---|---|
| PaySim | `mtalaltariq/paysim-data` | `paysim dataset.csv` |
| IBM AML | `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | `HI-Small_Trans.csv`, `HI-Small_Patterns.txt`, `LI-Small_Trans.csv`, `LI-Small_Patterns.txt` |
| Elliptic | `ellipticco/elliptic-data-set` | `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` |

### C.3 Organización del código

| Módulo | Función |
|---|---|
| `experiments/features.py` | Variables causales de PaySim (3.4) |
| `experiments/ibm_features.py` | Variables causales de IBM AML: contadores e instantáneas (3.4) |
| `experiments/run_experiments.py` | Protocolo común: partición, umbral, métricas; Estudio 1 |
| `experiments/run_ibm.py` | Estudio 2 en IBM AML y *bootstrap* pareado con pesos (3.6) |
| `experiments/run_elliptic.py` | Réplica en Elliptic |
| `experiments/injection.py` | Inyección controlada (3.7.3) |
| `tests/test_features.py`, `tests/test_ibm_features.py` | Pruebas de causalidad (3.4.3) |
| `docs/herramientas/figuras_marco.py` | Red de ejemplo, valores y figuras del Capítulo 2 |
| `docs/herramientas/generar_referencias.py` | Lista de referencias en APA 7 a partir de `bibliografia.bib` |
| `docs/herramientas/compatibilizar_github.py` | Conversión de fórmulas a la sintaxis que GitHub renderiza |

### C.4 Correspondencia entre scripts y resultados

**Tabla C.1.** Script que produce cada resultado del Capítulo 4. Las salidas se guardan en `results/`.

| Resultado | Comando | Salida |
|---|---|---|
| Tablas 4.2, 4.3 y Figuras 4.1, 4.2 | `python -m experiments.run_experiments` | `metrics.csv`, `paired_bootstrap.json`, `feature_importance.csv` |
| Contraste de H1 en PaySim (4.2) | `python -m experiments.run_t3_vs_b1` | `paired_bootstrap_T3_vs_B1.json` |
| Tabla 4.4 y Figura 4.3 | `python -m experiments.run_no_shortcut` | `metrics_no_shortcut.csv` |
| Exhaustividad por tipo (4.4) | `python -m experiments.verify_paysim_by_type` | `paysim_recall_by_type.json` |
| Tablas 4.6, 4.10, 4.12 y Figuras 4.4, 4.6 | `python -m experiments.run_ibm HI-Small` | `ibm/HI-Small/` |
| Tabla 4.7 | `python -m experiments.run_ibm LI-Small` | `ibm/LI-Small/` |
| Sección 4.7.2 | `python -m experiments.run_ibm HI-Small --full` | `ibm/HI-Small_full/` |
| Tabla 4.11 | `python -m experiments.run_ibm_struct_vs_time` | `ibm/HI-Small/struct_vs_time*` |
| Tablas 4.8, 4.9 y Figura 4.5 | `python -m experiments.run_elliptic` | `elliptic/` |
| Tabla 4.13 y Figura 4.7 | `python -m experiments.injection` | `injection/summary.csv` |
| Figuras | `python experiments/make_figures.py` | `figuras/` |

Antes de los experimentos de IBM se deben calcular sus variables con `python -m experiments.ibm_features HI-Small` (y `LI-Small`, y `HI-Small --full`); quedan en caché en `data/processed/`. Las pruebas se ejecutan con `python -m pytest tests/test_features.py tests/test_ibm_features.py`.

Los tiempos aproximados de cómputo en el equipo descrito son: variables de PaySim, 1.5 min; variables de IBM HI-Small, 6.5 min; Estudio 1, 31 min; Estudio 2 en HI-Small, 23 min; inyección, 75 min.
