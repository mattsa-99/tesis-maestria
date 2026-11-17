# 5. Discusión

Este capítulo interpreta los resultados del Capítulo 4 a la luz del marco explicativo (2.6.2) y de la literatura (2.5). Discute primero bajo qué condiciones aporta la topología (5.1), luego qué tipo de información de red explica la mejora (5.2) y qué fraude queda sin detectar (5.3). Después examina los atajos encontrados en los datos sintéticos (5.4), las implicaciones prácticas (5.5), el estado de las hipótesis rivales (5.6) y las limitaciones (5.7).

## 5.1 Cuándo aporta la topología

El resultado central de la tesis es que **la utilidad de las variables topológicas depende del conjunto de datos, y esa dependencia se relaciona con su estructura**. El marco explicativo de 2.6.2 propone dos condiciones para que la topología aporte: que el fraude forme estructuras de red que el comportamiento legítimo no reproduzca, y que el fraude no sea ya separable por atributos individuales. Los resultados de los cuatro escenarios son consistentes con ese marco. Como se formuló a partir de los propios resultados (2.6.2), esa consistencia no constituye una prueba independiente; muestra que el marco es una explicación plausible que conviene contrastar en conjuntos nuevos.

En **IBM AML** se cumplen ambas condiciones. El lavado se genera según tipologías de red. Las cuentas ilícitas están en ciclos con una frecuencia 5.5 veces mayor que la base (Tabla 4.5), y el modelo transaccional apenas detecta el lavado ($F_1$ = 0.136). Es el escenario en que el marco implica la mayor ganancia, y es donde se observa: el $F_1$ se multiplica por 4.5 en HI-Small y por 6 en LI-Small, y la exhaustividad sube a 76–93 % en los ocho tipos de patrón. La mejora es, además, robusta a la familia de modelo, a la semilla y al recorte temporal. Los trabajos previos sobre IBM AML no reportan un modelo solo transaccional (2.5.2), de modo que esta comparación aporta un dato que faltaba: **sin información de red, el lavado de IBM AML es prácticamente indetectable**.

El desempeño del modelo completo es del mismo orden que el publicado. Con el conjunto sin recorte, que es el protocolo más cercano al de Altman et al. (2023) y Egressy et al. (2024) (partición temporal 60/20/20, sin identificadores de cuenta), el $F_1$ de T3 es 0.717 (4.7.2). Los modelos de árboles con conteo de subgrafos obtienen 0.629–0.632 y la mejor red neuronal de grafos, 0.682 (Tabla 2.1). Esta comparación no es controlada y no permite afirmar que un enfoque supere a otro. Los modelos difieren en las variables (aquí se incluyen contadores temporales), en la forma de elegir el umbral (aquí se maximiza $F_1$ en validación) y, posiblemente, en el preprocesamiento. Lo que sí indica es que un conjunto reducido de contadores locales y causales alcanza un desempeño comparable al de métodos considerablemente más costosos. Además, todas esas cifras, incluida la de esta tesis sin recorte, se obtienen sobre un conjunto de prueba que contiene la cola temporal descrita en 4.5, que eleva el desempeño del modelo de referencia (4.7.2). Con el recorte, el $F_1$ de T3 baja a 0.614.

En **PaySim** fallan ambas condiciones. El grafo no tiene ciclos y casi ninguna cuenta recibe y envía dinero a la vez (Tabla 4.1), de modo que no hay estructuras multisalto que medir. Además, el fraude se separa casi perfectamente por la relación entre monto y saldo (Tabla 4.3). El modelo de referencia queda en el techo ($F_1$ = 0.998) y agregar topología lo empeora levemente, un resultado típico cuando se añaden variables sin señal a un modelo que ya no tiene margen. Cuando se retiran las variables de saldo (Tabla 4.4), la topología vuelve a aportar ($\Delta F_1 \approx +0.07$), pero solo mejora la precisión: reduce los falsos positivos un 36 % (de 1 799 a 1 154) sin detectar ningún fraude adicional (4.4). Con el grafo de PaySim, las variables de red sirven para descartar falsos positivos, no para encontrar fraude nuevo.

En **Elliptic** el marco no pudo evaluarse por completo. El grafo es acíclico por construcción, pero contiene 111 023 nodos intermedios, de modo que hay rutas de varios saltos (Tabla 4.5); no se dispone de información sobre si el fraude forma estructuras propias, porque Elliptic no etiqueta patrones. La segunda condición se cumple solo en parte, porque las variables locales ya alcanzan un ROC-AUC de 0.91. El resultado es intermedio: las variables agregadas de vecinos mejoran levemente la PR-AUC de forma consistente, pero la diferencia en $F_1$ no supera la variabilidad entre semillas. Esto coincide con Weber et al. (2019), cuyos resultados en este conjunto muestran que la información de vecinos aporta mejoras que dependen del modelo y del periodo.

La **inyección controlada** permite variar directamente la primera condición, manteniendo constante la segunda: las transacciones inyectadas son indistinguibles a nivel individual. Sin señuelos, la topología detecta prácticamente todo el fraude ($F_1$ ≈ 0.91–0.97). A medida que el fondo legítimo contiene estructuras idénticas, la ventaja se reduce de forma gradual, pero no desaparece ($F_1$ ≈ 0.6 con cinco señuelos por patrón, frente a ≈ 0 sin topología). Esa ventaja residual proviene de la única diferencia entre fraude y señuelo: la velocidad del patrón. Es una primera indicación de que el tiempo forma parte de la señal, punto que se retoma en 5.2.

En conjunto, estos resultados ofrecen una respuesta a la pregunta general, sujeta a la validación prospectiva del marco: las variables topológicas mejoran la detección **cuando el fraude deja una huella estructural que el comportamiento legítimo no reproduce, y cuando las variables individuales no bastan para separarlo**. Cuando falta cualquiera de las dos condiciones, la mejora es pequeña o nula.

## 5.2 Qué información de red explica la mejora

La segunda conclusión es menos esperada: **la ganancia no proviene de las métricas globales del grafo, sino de contadores locales actualizados transacción a transacción, tanto estructurales como temporales**. En IBM, retirar las instantáneas (PageRank, intermediación, comunidades, componentes y ciclo de longitud 3) apenas cambia el desempeño. Retirar los contadores, en cambio, reduce el $F_1$ a la mitad (Tabla 4.10). El mismo patrón aparece en PaySim sin las variables de saldo: grado y fuerza alcanzan lo mismo que el conjunto completo.

Dentro de los contadores conviven dos tipos de información: la estructural (grados, *fan-in*/*fan-out*, repetición del par, arista inversa) y la temporal (tiempo desde la última transacción de cada cuenta). Separarlas (Tabla 4.11) muestra que **ambas aportan por sí solas una mejora grande y de magnitud similar**: cada una multiplica el $F_1$ del modelo de referencia por algo más de tres. Además son **complementarias**, porque juntas superan a la estructura sola en 0.14 a 0.19 puntos de $F_1$ (IC 95 %). La hipótesis rival R1 queda, por tanto, *parcialmente* apoyada: una parte sustancial de la ganancia es temporal. Sin embargo, la estructura sola, con o sin instantáneas, también cumple el criterio de H1 frente al modelo transaccional. Así, la conclusión principal no depende de las variables temporales.

Esto tiene sentido desde la definición de los patrones: un *fan-out* de lavado no es solo una cuenta con muchos destinatarios, sino una cuenta que los acumula **en poco tiempo** tras un periodo de inactividad. El resultado coincide con dos antecedentes. Van Vlasselaer et al. (2015) encontraron, en fraude con tarjetas, que las variables de recencia y frecuencia y las de red están estrechamente entrelazadas y que el mejor modelo combina ambas. Starnini et al. (2021) definen el *smurfing* precisamente por la concentración temporal de transacciones en un subgrafo. En un sentido análogo, Egressy et al. (2024) reportan que, en otro conjunto de datos, agregar estadísticos del tiempo entre transacciones a las variables de grafo eleva el $F_1$ de 0.27 a más de 0.50. La inyección controlada apunta en la misma dirección. Cuando fraude y señuelos tienen la misma estructura, lo único que los distingue es la velocidad, y el modelo aún separa buena parte de los casos (5.1). La información de red útil es, en rigor, **estructural y temporal a la vez**, lo que coincide con la perspectiva de las redes temporales, en la que el orden y el ritmo de las aristas forman parte de la estructura (Holme y Saramäki, 2012).

Dos mecanismos pueden explicar que las métricas globales aporten poco. El primero es la **latencia**. Las instantáneas se recalculan cada 6 o 24 horas, mientras que muchos patrones de lavado se completan en menos tiempo. Cuando la instantánea refleja el patrón, la mayoría de sus transacciones ya ocurrieron. Los contadores, en cambio, reflejan cada transacción en cuanto ocurre. El segundo mecanismo es la **localidad**. El lavado es un fenómeno de unas pocas cuentas, y una métrica como el PageRank resume la posición del nodo en todo el grafo, por lo que diluye la señal local. Las métricas de ciclo ilustran la diferencia entre ser diagnóstica y ser útil. Que una transacción cierre un ciclo de longitud 3 es 140 veces más frecuente en el lavado que en las transacciones legítimas, pero ocurre en solo el 2 % de las ilícitas, de modo que su aporte agregado es pequeño.

La variable estructural más informativa es la **repetición del par** de cuentas. Las relaciones legítimas son recurrentes (7.5 transacciones previas en promedio), mientras que el lavado usa contrapartes nuevas (1.5). Es una forma de "memoria relacional" que no está en la lista clásica de métricas topológicas (2.2), pero que se deriva naturalmente de representar las transacciones como un multigrafo. Sugiere que la información de red más útil para el lavado no es la posición del nodo en el grafo, sino la novedad de sus relaciones.

## 5.3 Lo que la topología no detecta

Hay tres tipos de fraude que ninguna configuración detecta, y cada uno indica un límite del enfoque.

- **El lavado sin patrón listado en IBM (38 % de las transacciones ilícitas)**, con exhaustividad del 1 % en todos los modelos. Si estas transacciones no siguen ninguna tipología, es coherente que la topología no las detecte. El modelo transaccional tampoco lo hace, lo que indica que tampoco son atípicas a nivel individual. Egressy et al. (2024) observan lo mismo con redes neuronales de grafos: exhaustividad cercana a cero en ese grupo, que explica buena parte del bajo $F_1$ en LI-Small, donde es el 71 % del lavado. Que dos enfoques tan distintos fallen en el mismo subconjunto sugiere que el límite está en la información disponible, no en el modelo.
- **El fraude CASH_OUT en PaySim sin las variables de saldo.** Los retiros fraudulentos no se distinguen ni por sus atributos ni por su entorno de red.
- **Los ciclos en la inyección controlada**, el patrón con menor exhaustividad (0.64–0.76). En PaySim no se calcularon contadores de cierre de ciclo, y un ciclo se completa antes de que la instantánea diaria lo refleje.

Los tres casos apuntan a lo mismo: la topología detecta lo que deja una estructura **visible a tiempo** en las variables disponibles.

## 5.4 Atajos en los conjuntos de datos sintéticos

Un resultado metodológico transversal es la frecuencia con que aparecieron **atajos**: regularidades de los datos que permiten acertar sin capturar el fenómeno de interés. Se encontraron tres.

1. **PaySim, saldos.** El simulador genera el fraude vaciando la cuenta de origen, de modo que la razón monto/saldo identifica el fraude casi perfectamente.
2. **IBM AML, cola temporal.** Tras el 2022-09-10 el generador sigue completando patrones de lavado cuando la actividad legítima ya terminó. Sin el recorte, el $F_1$ del modelo de referencia sube de 0.136 a 0.238.
3. **La propia inyección controlada en su primera versión.** Iniciar los patrones de forma uniforme en el tiempo, sobre un fondo cuya actividad se concentra al inicio, hizo que el modelo de referencia aprendiera el tiempo ($F_1$ = 0.41).

El tercer caso es el más instructivo, porque muestra que el riesgo existe incluso cuando quien diseña los datos lo tiene presente. Estos hallazgos son instancias del aprendizaje por atajos documentado por Geirhos et al. (2020) y tienen una consecuencia práctica para la investigación en fraude. **Un resultado sobre un conjunto sintético solo es interpretable si antes se examina cómo el generador produce las etiquetas.** En particular, PaySim no permite evaluar métodos basados en grafos: su grafo no tiene la estructura que esos métodos buscan, y su fraude se identifica sin ella.

## 5.5 Implicaciones prácticas

Para una entidad que quiera incorporar información de red en su detección de lavado, los resultados sugieren tres pautas.

1. **Priorizar contadores locales en línea sobre métricas globales.** Grado y fuerza recientes, *fan-in*/*fan-out*, repetición del par y tiempos entre transacciones son baratos de mantener, se actualizan con cada transacción y explican casi toda la ganancia. Las métricas globales exigen recalcular el grafo completo y aportan poco sobre ellos.
2. **Caracterizar la red antes de invertir en métodos de grafos.** Si casi ninguna cuenta recibe y envía dinero, o si no hay ciclos, no hay estructura que explotar. Medir esas propiedades (Tablas 4.1 y 4.5) es barato y orienta sobre si la inversión puede tener retorno.
3. **Combinar la topología con otras señales.** Incluso en el mejor escenario queda una fracción del lavado sin estructura que ningún modelo detecta.

Estas pautas se derivan de datos mayoritariamente sintéticos. Su aplicación a una red bancaria real requiere validarlas con datos propios (5.7).

## 5.6 Estado de las hipótesis rivales

La Tabla 5.1 resume la evidencia sobre cada hipótesis rival planteada en 2.6.3.

**Tabla 5.1.** Evaluación de las hipótesis rivales (2.6.3).

| Rival | Evidencia | Conclusión |
|---|---|---|
| R1 Comportamiento temporal | Los contadores temporales solos mejoran tanto como los estructurales solos; juntos son complementarios; la estructura sola también cumple el criterio de H1 (Tabla 4.11) | Parcialmente apoyada: el tiempo explica una parte sustancial de la ganancia, pero no toda |
| R2 Artefacto del generador | Se identificaron y controlaron dos atajos (saldos en PaySim, cola temporal en IBM); con y sin recorte la mejora es similar; la exhaustividad mejora en los ocho tipos de patrón | Descartada para los artefactos identificados; no se puede excluir que existan otros en el generador de IBM |
| R3 Fuga de información | Variables causales verificadas por pruebas; ninguna usa la etiqueta | Descartada |
| R4 Más variables, más capacidad | En PaySim, agregar 13 variables topológicas no mejora; en IBM LI-Small, agregar las 10 variables de instantánea apenas cambia el $F_1$ (0.044 a 0.051), mientras que agregar los 12 contadores lo multiplica por 6 (Tabla 4.7) | Descartada: la mejora depende de qué variables se agregan, no de cuántas |
| R5 Azar | IC por *bootstrap* lejos de cero en IBM; desviación entre semillas despreciable frente a la diferencia; en Elliptic, donde la varianza importa, el resultado se declara no robusto | Descartada en IBM |

## 5.7 Limitaciones

- **Validez externa.** IBM AML y PaySim son sintéticos. La mejora observada indica que la topología captura los patrones **que el generador implementa**, no que el lavado real los siga. Elliptic es real, pero tiene variables anonimizadas y una estructura particular.
- **Representación del grafo.** Las métricas globales se calcularon por bloques de 6 o 24 horas. Un grafo incremental podría cambiar el balance entre contadores e instantáneas, y por tanto la conclusión de 5.2.
- **Aproximaciones.** En IBM y Elliptic la intermediación se restringe a caminos de longitud ≤ 4. En Elliptic, las métricas de grafo propias no son estrictamente causales dentro de un periodo.
- **Modelos.** Solo se evaluaron clasificadores convencionales con hiperparámetros fijos. Las redes neuronales de grafos podrían extraer más información de la estructura.
- **Análisis post hoc.** El análisis de PaySim sin las variables de saldo y la separación entre contadores estructurales y temporales se definieron después de ver los resultados principales. El segundo se añadió para responder a la hipótesis rival R1; su conclusión, que la estructura sola cumple el criterio, debe leerse como exploratoria.
- **Alcance topológico.** No se implementó homología persistente sobre filtraciones. La topología se estudió mediante métricas de grafos dirigidos. Los ciclos dirigidos que se usan como variables son un refinamiento que el número de Betti $\beta_1$ no captura (2.3).
