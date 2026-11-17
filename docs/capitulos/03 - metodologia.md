# 3. Metodología

Este capítulo describe cómo se investigaron las preguntas de 1.2. Presenta el diseño general en dos estudios (3.1), los conjuntos de datos (3.2), su preparación y partición (3.3), las variables y cómo se garantizó su causalidad (3.4), los modelos (3.5), las métricas y la inferencia (3.6), el criterio de decisión y las configuraciones comparadas (3.7), la validez del diseño (3.8), las desviaciones respecto de lo planificado (3.9) y la reproducibilidad (3.10).

## 3.1 Diseño general

La investigación es un **diseño experimental comparativo** sobre datos observacionales: se entrenan modelos que difieren solo en el grupo de variables que reciben, se evalúan sobre el mismo conjunto de prueba y se atribuye la diferencia de desempeño a ese grupo. Este diseño permite aislar el aporte de la información de red, siempre que se controlen las hipótesis rivales de 2.6.3.

La investigación se organizó en **dos estudios** sucesivos.

- **Estudio 1 (PaySim).** Es el diseño original de la tesis. Contrasta la hipótesis en PaySim y caracteriza su grafo. Sus resultados mostraron que PaySim no permitía contrastarla (Capítulo 4) y motivaron el Estudio 2.
- **Estudio 2 (datos con estructura de red).** Contrasta la hipótesis en IBM AML HI-Small, que es el **contraste confirmatorio**, con réplicas en IBM AML LI-Small y Elliptic, ablaciones e inyección controlada en PaySim. Su criterio de decisión se escribió el 16 de noviembre de 2026, **antes** de ejecutar cualquier experimento sobre esos conjuntos, y no se modificó después (3.7).

Se distinguen, además, los análisis **planificados** de los **post hoc**, es decir, definidos después de ver resultados. Los análisis post hoc se reportan como exploratorios.

## 3.2 Conjuntos de datos

Los tres conjuntos, resumidos en la Tabla 3.1, se eligieron por cubrir condiciones distintas del marco explicativo (2.6.2). PaySim es el estándar de fraude en dinero móvil. IBM AML contiene patrones de red etiquetados. Elliptic es real y tiene una estructura distinta (grafo acíclico).

**Tabla 3.1.** Conjuntos de datos.

| Conjunto | Origen | Transacciones | Nodos | Positivos | Prevalencia | Periodo |
|---|---|---|---|---|---|---|
| PaySim | Lopez-Rojas et al. (2016) | 6 362 620 | 9 073 900 cuentas | 8 213 fraudes | 0.129 % | 743 pasos horarios (≈ 31 días) |
| IBM AML HI-Small | Altman et al. (2023) | 5 078 345 | 515 088 cuentas | 5 177 ilícitas | 0.102 % | 2022-09-01 a 2022-09-18 |
| IBM AML LI-Small | Altman et al. (2023) | 6 924 049 | 705 907 cuentas | 3 565 ilícitas | 0.051 % | 2022-09-01 a 2022-09-17 |
| Elliptic | Weber et al. (2019) | 234 355 aristas | 203 769 transacciones | 4 545 ilícitas etiquetadas | 9.8 % de las etiquetadas | 49 periodos |

**PaySim** simula transacciones de dinero móvil calibradas con registros de un servicio real. Sus variables son el paso temporal, el tipo (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER), el monto, las cuentas de origen y destino y sus saldos antes y después. Todos los fraudes son de tipo TRANSFER (4 097) o CASH_OUT (4 116). Se verificó que no hay valores nulos ni filas duplicadas. La variable `isFlaggedFraud` (16 casos) implementa una regla del simulador y no se usa.

**IBM AML** (versiones HI-Small y LI-Small, con alta y baja prevalencia) genera transacciones bancarias entre cuentas de distintos bancos, con montos, monedas y formatos de pago. El lavado se inyecta según ocho tipologías (*fan-in*, *fan-out*, ciclo, bipartito, *stack*, *scatter-gather*, *gather-scatter* y aleatorio), documentadas en un archivo de patrones que lista 370 intentos y 3 209 transacciones en HI-Small. Las restantes 1 968 transacciones ilícitas de HI-Small (38 %) no pertenecen a ningún patrón listado.

**Elliptic** contiene transacciones reales de Bitcoin como nodos y flujos de fondos como aristas. Cada transacción tiene 93 variables locales anonimizadas, 72 variables agregadas sobre sus vecinos y el número de periodo (aproximadamente dos semanas por periodo). Solo el 23 % de las transacciones está etiquetado como lícito o ilícito; el resto se excluye del entrenamiento y la evaluación.

**Datos descartados.** Un primer modelo de referencia (bitácora, semana 2) se entrenó sobre un conjunto de 200 000 filas generado aleatoriamente como sustituto de PaySim. Como su etiqueta se asignaba al azar, sus resultados no son válidos y no se usan.

## 3.3 Preparación y partición

En los tres conjuntos la partición es **temporal** (Tabla 3.2): se entrena con el pasado y se evalúa con el futuro, como ocurriría en operación. Dentro del entrenamiento, un tramo final se reserva para elegir el umbral de decisión. Luego el modelo se reajusta con todo el entrenamiento y se evalúa **una sola vez** en prueba.

**Tabla 3.2.** Partición temporal de cada conjunto.

| Conjunto | Ajuste | Validación (umbral) | Prueba | Positivos en prueba |
|---|---|---|---|---|
| PaySim | pasos 1–504 | pasos 505–576 | pasos 577–743 (161 426 tx.) | 1 840 |
| IBM HI-Small | primer 60 % de filas | siguiente 20 % | último 20 % (1 015 448 tx.) | 1 143 |
| IBM LI-Small | primer 60 % de filas | siguiente 20 % | último 20 % (1 384 766 tx.) | 791 |
| Elliptic | periodos 1–26 | periodos 27–34 | periodos 35–49 (16 670 tx.) | 1 083 |

Decisiones específicas:

- **PaySim.** El corte por tiempo (24 de 31 días) deja el 97 % de las filas en entrenamiento, porque la actividad del simulador se concentra al inicio. Se mantuvo el corte por tiempo para no mezclar periodos.
- **IBM AML: recorte temporal.** Se excluyen las transacciones posteriores al 2022-09-10. Tras esa fecha la actividad legítima casi desaparece, pero el generador sigue completando patrones de lavado: entre 57 % y 73 % de las transacciones de cada día son ilícitas. Sin el recorte, el modelo podría identificar el lavado por la fecha (hipótesis rival R2). Se conservan 5 077 237 filas (99.98 %) y 4 522 ilícitas (87.3 %) en HI-Small. El conjunto completo se usa como análisis de sensibilidad.
- **IBM AML: partición por filas.** Se usa 60/20/20 por filas en orden temporal, la práctica habitual con este conjunto. Una partición por días dejaría muy pocos positivos en prueba, porque el lavado se concentra en los primeros diez días.
- **Elliptic.** Se usa la partición estándar de Weber et al. (2019): entrenamiento hasta el periodo 34 y prueba del 35 al 49.

## 3.4 Variables

### 3.4.1 Definición operacional

Una variable se considera **transaccional** si se calcula solo con los atributos de la transacción evaluada, y **topológica** si requiere información de otras transacciones del grafo. Las topológicas se dividen en tres tipos:

- **Estructurales locales**: cuentan relaciones del nodo con su entorno inmediato. Incluyen grado y fuerza previos, grado en una ventana de 24 horas (*fan-in*/*fan-out*), repetición del par de cuentas y ciclos cortos.
- **Temporales de red**: describen el ritmo de actividad de la cuenta en la red, como el tiempo desde su última transacción.
- **Globales**: dependen de todo el grafo. Incluyen PageRank, tamaño de componente, intermediación, comunidad y componente fuerte.

Esta distinción permite responder P3 y evaluar la hipótesis rival R1. El catálogo completo de variables, con su definición y su grupo, está en el Anexo B.

### 3.4.2 Variables por conjunto

**PaySim.**
- *Base (8)*: tipo, monto (log), saldo previo de origen y de destino (log), destino con saldo cero, razón monto/saldo de origen, fondos insuficientes y paso temporal.
- *Comercio (1)*: destino es comercio (nombre que empieza por "M"). Se separa de la base para no atribuir a la topología una mejora que no le corresponde.
- *Grado y fuerza causales (6)*: grados previos de entrada y salida de origen y destino, fuerza de entrada previa del destino y tiempo desde su última entrada.
- *Instantáneas diarias (7)*: PageRank, tamaño de componente débil e intermediación de origen y destino, y tamaño de comunidad del destino.

**IBM AML.**
- *Transaccional, B (10)*: montos pagado y recibido (log), monedas, coincidencia de monedas, formato de pago, mismo banco, autotransferencia, hora y día de la semana. No se usa la fecha absoluta.
- *Contadores (12)*: diez estructurales (grados y fuerzas previos, *fan-in*/*fan-out* en 24 h, repetición del par y arista inversa previa) y dos temporales (minutos desde la última transacción del origen y del destino).
- *Instantáneas cada 6 horas (10)*: PageRank, tamaño de componente fuerte, origen y destino en la misma componente fuerte, cierre de ciclo de longitud 3, tamaño de comunidad e intermediación.

**Elliptic.**
- *Locales, L (93)*: las variables locales del conjunto. Se excluye el número de periodo.
- *Agregadas, AF (72)*: estadísticos de los vecinos que provee el conjunto.
- *Grafo, G (10)*: grados de entrada y salida, PageRank, tamaño de componente, intermediación, tamaño de comunidad, número de núcleo (*coreness*), coeficiente de agrupamiento y número de nodos alcanzables en dos pasos de entrada y de salida.

Las variables `newbalanceOrig` y `newbalanceDest` de PaySim se excluyen porque reflejan el resultado de la transacción y, en los fraudes, el simulador las deja en valores que revelan la etiqueta.

### 3.4.3 Causalidad

Todas las variables de PaySim e IBM son **causales** en el sentido de la Definición 2.5: para una transacción en el instante $`t`$ solo usan transacciones anteriores. Se implementan de dos formas.

- **Contadores** (Definición 2.7): acumulados según el orden $`\prec`$ de la Definición 2.3, que excluyen la transacción actual. Las ventanas de *fan-in*/*fan-out* cuentan solo instantes estrictamente anteriores a $`t`$.
- **Instantáneas** (Definición 2.4): se divide el tiempo en bloques de duración $`b`$ ($`b`$ = 24 h en PaySim y 6 h en IBM). A una transacción en el instante $`t`$ se le asignan las métricas de la instantánea $`G_{<\tau(t)}`$, donde $`\tau(t) = b\lfloor t / b \rfloor`$ es el inicio de su bloque. En PaySim se usa el multigrafo $`G_{<\tau}`$ y en IBM su versión simple $`\bar{G}_{<\tau}`$. Como $`\tau(t) \le t`$, estas variables son causales, con un rezago de a lo sumo $`b`$. Los nodos aún no vistos reciben valor 0.

La intermediación se calcula de forma **exacta** en PaySim sobre las componentes que contienen nodos con entradas y salidas; es equivalente al cálculo global, porque solo esos nodos pueden tener intermediación positiva. En IBM y Elliptic se **restringe a caminos de longitud ≤ 4**. La pertenencia a ciclos se calcula en PaySim, pero vale cero en todos los casos, por lo que no entra al modelo.

La causalidad se verificó con **pruebas automatizadas** (`tests/test_features.py` y `tests/test_ibm_features.py`, 9 pruebas). Comprueban que las variables de cada fila no cambian al agregar transacciones futuras y que su valor coincide con el esperado en grafos pequeños construidos a mano. Ninguna variable usa la etiqueta, lo que descarta la fuga directa (hipótesis rival R3).

En Elliptic, las métricas de grafo propias se calculan sobre el grafo de cada periodo, que puede incluir transacciones posteriores dentro del mismo periodo. Es la práctica estándar con este conjunto, pero no es estrictamente causal.

## 3.5 Modelos

- **Modelo principal**: *Random Forest* (Breiman, 2001) con 200 árboles, mínimo 3 observaciones por hoja, submuestreo del 25 % por árbol y ponderación de clases por submuestra. Se eligió porque es robusto, requiere poco ajuste y es estable frente a variables irrelevantes, de modo que agregar un grupo inútil no debería mejorar el desempeño por sí mismo (hipótesis rival R4).
- **Modelo de robustez**: *gradient boosting* con histogramas (300 iteraciones, ponderación de clases), evaluado en el modelo de referencia y en el completo, para comprobar que la conclusión no depende de la familia de modelo.
- Los hiperparámetros son **idénticos en todas las configuraciones** y no se ajustaron, para que la única diferencia entre modelos sea el grupo de variables.
- Cada configuración se entrena con tres semillas (42, 43 y 44).

## 3.6 Métricas e inferencia

- **Métrica principal**: $`F_1`$ en prueba (Definición 2.20) con el umbral $`\theta^{*} = \arg\max_{\theta} F_1^{\text{val}}(\theta)`$, elegido sobre los puntajes de validación del modelo ajustado sin ella.
- **Secundarias**: precisión, exhaustividad, PR-AUC y ROC-AUC (Definiciones 2.20 a 2.22). La PR-AUC no depende del umbral y es la comparación más robusta entre modelos (Saito y Rehmsmeier, 2015).
- **Incertidumbre de muestreo: *bootstrap* pareado.** Sea $`\mathcal{D} = \{(y_i, s^{A}_i, s^{B}_i)\}_{i=1}^{n}`$ el conjunto de prueba, con los puntajes de dos modelos $`A`$ y $`B`$ para cada transacción, y sea $`M`$ una métrica ($`F_1`$ con umbrales fijos $`\theta^{*}_A`$ y $`\theta^{*}_B`$, o PR-AUC). Para $`r = 1, \dots, R`$ se extrae una muestra $`\mathcal{D}^{*r}`$ de $`n`$ índices con reemplazo y probabilidad uniforme, y se calcula

```math
\Delta^{*r} = M\big(B; \mathcal{D}^{*r}\big) - M\big(A; \mathcal{D}^{*r}\big).
```

El intervalo de confianza del 95 % es el intervalo de percentiles $`[\Delta^{*}_{(0.025)}, \Delta^{*}_{(0.975)}]`$ (Efron, 1979; Efron y Tibshirani, 1993). Se usa $`R = 1\,000`$ (500 en la inyección). El remuestreo es **pareado** porque ambos modelos se evalúan sobre las mismas transacciones, lo que elimina la variabilidad debida a la composición de la muestra. Se reporta además $`\hat{p} = R^{-1} \sum_r \mathbb{1}[\Delta^{*r} \le 0]`$.
- **Implementación.** Remuestrear con reemplazo equivale a ponderar cada transacción con un peso $`c_i`$, donde $`(c_1, \dots, c_n) \sim \text{Multinomial}(n; 1/n, \dots, 1/n)`$. $`F_1`$ y PR-AUC se calculan con esos pesos sobre un orden fijo de los puntajes, agrupando los empates como en la Definición 2.21. Se verificó numéricamente que coincide con scikit-learn (Pedregosa et al., 2011).
- **Incertidumbre de entrenamiento**: media y desviación estándar sobre las tres semillas. Si la desviación entre semillas es comparable a la diferencia entre modelos, como en Elliptic, el *bootstrap* se calcula para cada semilla (hipótesis rival R5).

## 3.7 Criterio de decisión y configuraciones

### 3.7.1 Estudio 1 (PaySim)

Las configuraciones del Estudio 1 se muestran en la Tabla 3.3.

**Tabla 3.3.** Configuraciones del Estudio 1.

| Código | Variables | Propósito |
|---|---|---|
| B0 | Base | Modelo de referencia inicial |
| B1 | Base + comercio | Modelo de referencia para el contraste |
| T1 | B1 + grado y fuerza | Aporte de las variables locales |
| T2 | B1 + instantáneas | Aporte de las variables globales |
| T3 | B1 + grado + instantáneas | Modelo topológico completo |
| T4 | Grado + instantáneas | Topología sin variables transaccionales |

**Criterio.** H1 se apoya si $`F_1(\text{T3}) \ge 0.80`$ y el límite inferior del IC 95 % de $`\Delta F_1 = F_1(\text{T3}) - F_1(\text{B1})`$ es positivo. Además se realizó una ablación (T3 sin cada grupo). El umbral de 0.80 proviene del plan inicial de la tesis (bitácora, semana 2), donde se fijó como desempeño mínimo para considerar útil un sistema de detección. En el Estudio 2 se abandonó, porque un umbral absoluto de $`F_1`$ depende de la prevalencia y de la dificultad de cada conjunto, mientras que H1 afirma una **mejora relativa** frente al modelo de referencia.

**Análisis post hoc.** Las configuraciones N0–N3 repiten B1 a T3 sin las cuatro variables que reconstruyen el atajo de saldos de PaySim (4.3): monto, saldo de origen, su razón y fondos insuficientes.

### 3.7.2 Estudio 2 (criterio fijado antes de los experimentos)

Las configuraciones del Estudio 2 se muestran en la Tabla 3.4. Son análogas a las del Estudio 1, con los grupos de variables propios de IBM AML.

**Tabla 3.4.** Configuraciones del Estudio 2 (IBM AML).

| Código | Variables (IBM) | Propósito |
|---|---|---|
| B | Transaccional | Modelo de referencia |
| T1 | B + contadores | Aporte de las variables locales y temporales |
| T2 | B + instantáneas | Aporte de las variables globales y de ciclos |
| T3 | B + contadores + instantáneas | Modelo topológico completo |
| T4 | Contadores + instantáneas | Topología sin variables transaccionales |

**Criterio de H1 (confirmatorio).** En IBM HI-Small, con *Random Forest* y el protocolo de 3.3, **H1 se apoya si los límites inferiores de los IC 95 % de $`\Delta F_1`$ y de $`\Delta`$PR-AUC de T3 frente a B son ambos estrictamente positivos.** En caso contrario se mantiene H0.

**Réplicas.** IBM LI-Small (mismas configuraciones) y Elliptic (L, L + AF, L + G, L + AF + G, G), con el mismo criterio. Se reportan como evidencia de generalización, no como condición de H1.

**Análisis complementarios.**
- *Planificados*: ablación (T3 sin cada grupo), exhaustividad por tipo de patrón, sensibilidad sin el recorte temporal e inyección controlada.
- *Post hoc*: separación de los contadores en estructurales y temporales (hipótesis rival R1). Se añadió tras la revisión del borrador, una vez conocidos los resultados principales.

### 3.7.3 Inyección controlada (prueba de capacidad)

En PaySim se eliminan los fraudes originales y se inyectan tres tipos de patrón:

- cadenas de 3 a 5 cuentas mula que terminan en un retiro;
- ciclos de 3 a 5 cuentas;
- anillos de 5 a 15 orígenes que envían a una cuenta recolectora, la cual luego retira los fondos.

Cada transacción inyectada copia el monto y los saldos previos de una transacción legítima real del mismo tipo, de modo que sea **indistinguible a nivel individual**.

Como el fondo legítimo de PaySim no contiene estructuras de varios saltos, cualquier estructura delataría el fraude. Para evitarlo se inyectan también **señuelos**: estructuras idénticas etiquetadas como legítimas, que se reparten a lo largo de días en lugar de horas. El experimento cruza tres prevalencias (0.05 %, 0.1 % y 0.5 %) con tres niveles de señuelos (0, 1 y 5 por patrón fraudulento) y compara B1 con T3.

Como los patrones fueron diseñados por el autor, este experimento muestra **cuándo** funciona el método, pero no prueba que esos patrones existan en la realidad.

## 3.8 Validez

**Validez interna.** La atribución de la mejora a la topología se apoya en cuatro elementos:
- los modelos solo difieren en sus variables;
- la evaluación es única y sobre las mismas transacciones de prueba;
- las variables son causales y están verificadas por pruebas;
- cada hipótesis rival tiene un control (2.6.3).

La principal amenaza restante es que alguna variable topológica capture una regularidad del generador que no sea estructural. Se mitiga caracterizando cada conjunto antes de interpretar y buscando atajos explícitamente, como en el atajo de saldos de PaySim y la cola temporal de IBM.

**Validez externa.** Dos de los tres conjuntos son sintéticos, de modo que los resultados indican que la topología captura los patrones **que esos generadores implementan**. Elliptic es real, pero sus variables están anonimizadas y su grafo tiene una estructura particular (acíclico, con nodos que son transacciones). Generalizar a redes bancarias reales requiere validación con datos reales etiquetados.

**Supuestos.**
- Las etiquetas son correctas y completas, lo que se cumple por construcción en los conjuntos sintéticos y solo en la fracción etiquetada de Elliptic.
- El orden de las transacciones dentro de un mismo minuto o paso es el del archivo original.
- Los hiperparámetros fijos son razonables para todas las configuraciones.

## 3.9 Desviaciones respecto de lo planificado

Ninguna de estas desviaciones modifica el criterio de H1.

- **Intermediación.** Se planificó aproximarla por muestreo. Se usó la restricción a caminos de longitud ≤ 4, que es determinista y viable en todas las instantáneas, porque el grafo de IBM tiene unas 648 000 aristas únicas.
- ***Bootstrap* en Elliptic.** Se calculó para cada semilla, porque la desviación entre semillas del modelo local (0.030 en $`F_1`$) es comparable a las diferencias observadas. En IBM, esa desviación (≈ 0.007) es despreciable frente a la diferencia, por lo que se usó la semilla 42.
- **Inyección, versión 2.** En la primera versión los patrones se iniciaban de forma uniforme en el tiempo. Como la actividad legítima de PaySim se concentra al inicio, la tasa de positivos en prueba era nueve veces la de entrenamiento y el modelo de referencia aprendía el tiempo ($`F_1`$ = 0.41 con 0.5 % de prevalencia). En la versión final el inicio se muestrea de la distribución temporal de las transacciones legítimas: la tasa de positivos queda en 0.50 % en entrenamiento y 0.56 % en prueba. Los resultados de la versión 1 se conservan en `results/injection_v1_inicio_uniforme/`.
- **Exhaustividad por patrón en LI-Small.** No se calculó, porque el archivo de patrones no estaba disponible en la corrida principal.
- **Separación estructural/temporal.** Se añadió como análisis post hoc (3.7.2).

## 3.10 Reproducibilidad

El código está en el repositorio público de la tesis (Anexo C), en la carpeta `experiments/`, y las semillas son fijas. Las métricas de grafo se calculan con igraph (Csardi y Nepusz, 2006) y los modelos con scikit-learn (Pedregosa et al., 2011). Los conjuntos de datos se descargan con `kagglehub`.

```bash
python -m experiments.features                  # variables de PaySim
python -m experiments.run_experiments           # Estudio 1
python -m experiments.run_t3_vs_b1              # contraste de H1 en PaySim (T3 frente a B1)
python -m experiments.run_no_shortcut           # PaySim sin atajo (post hoc)
python -m experiments.verify_paysim_by_type     # exhaustividad por tipo en PaySim sin atajo
python -m experiments.ibm_features HI-Small     # variables de IBM (también LI-Small, --full)
python -m experiments.run_ibm HI-Small          # contraste confirmatorio
python -m experiments.run_ibm_struct_vs_time    # rival R1 (post hoc)
python -m experiments.run_elliptic              # réplica en Elliptic
python -m experiments.injection                 # inyección controlada
python experiments/make_figures.py              # figuras
```

El Anexo C detalla el entorno de ejecución, la versión del código, el origen de los datos y qué script produce cada tabla y figura (Tabla C.1).
