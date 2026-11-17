# 1. Introducción

Esta tesis estudia cuándo la estructura de una red de transacciones, descrita mediante métricas de grafos dirigidos, ayuda a detectar fraude y lavado de dinero más allá de lo que revelan las transacciones individuales. El capítulo presenta el contexto (1.1), el problema y las preguntas de investigación (1.2), su justificación (1.3), las definiciones de los términos centrales (1.4), la hipótesis (1.5), los objetivos (1.6), el alcance (1.7) y la organización del documento (1.8).

## 1.1 Contexto

La detección automática de fraude financiero se ha apoyado tradicionalmente en reglas y en modelos que evalúan cada transacción por separado: su monto, su tipo, los saldos involucrados y el momento en que ocurre. Este enfoque es eficaz cuando la transacción fraudulenta es atípica por sí misma. Sin embargo, una parte relevante del fraude y, sobre todo, del lavado de dinero, se organiza como una operación coordinada entre varias cuentas: fondos que se dispersan hacia muchos destinatarios (*fan-out*), que se concentran en una cuenta recolectora (*fan-in*), que pasan por cadenas de cuentas intermediarias o que regresan a su origen formando ciclos (Akoglu et al., 2015; Altman et al., 2023). En estos casos, cada transacción puede parecer normal y lo anómalo es el patrón que forman en conjunto.

Esta observación motiva el uso de la teoría de grafos: si las cuentas son nodos y las transacciones son aristas dirigidas, los patrones de lavado se convierten en estructuras del grafo que pueden medirse. La literatura ha propuesto métricas de centralidad, detección de comunidades y búsqueda de ciclos, así como modelos que aprenden directamente sobre el grafo (Akoglu et al., 2015; Weber et al., 2019). Lo que está menos claro es **en qué condiciones** esa información estructural mejora efectivamente la detección. La respuesta depende de propiedades de los datos que rara vez se examinan antes de evaluar un método: si el grafo contiene las estructuras que se quieren medir, y si el fraude ya se distingue por atributos individuales.

## 1.2 Problema y preguntas de investigación

El planteamiento inicial de esta tesis asumía que las métricas topológicas mejoran la detección de fraude en general, y se diseñó sobre PaySim (Lopez-Rojas et al., 2016), el conjunto de datos público más usado para fraude en dinero móvil. Los primeros experimentos mostraron que ese supuesto no podía evaluarse en PaySim: su grafo no contiene ciclos y casi ninguna cuenta recibe y envía dinero a la vez, y su fraude se identifica casi perfectamente por una regla del simulador (Capítulo 4). Esto reorientó el problema hacia una pregunta más precisa.

**Pregunta general.** ¿Bajo qué condiciones las variables topológicas derivadas de grafos dirigidos de transacciones mejoran la detección de fraude y lavado de dinero frente a modelos que usan solo variables transaccionales?

**Preguntas específicas.**

- **P1.** En un conjunto de datos cuyo lavado sigue patrones de red (IBM AML), ¿mejoran las variables topológicas la detección frente al modelo transaccional, según un criterio definido antes de los experimentos?
- **P2.** ¿Se replica esa mejora en un conjunto con menor prevalencia (IBM AML LI) y en un conjunto real con otra estructura (Elliptic)?
- **P3.** ¿Qué tipo de información de red explica la mejora: la estructura local (grados, *fan-in*/*fan-out*, repetición de contrapartes, ciclos cortos), el comportamiento temporal de las cuentas o las métricas globales (PageRank, intermediación, comunidades)?
- **P4.** ¿Qué propiedades de los datos y del fraude condicionan la mejora? Se estudia mediante el caso de PaySim y un experimento de inyección controlada de patrones de fraude.

## 1.3 Justificación

**Contribución científica.** Los conjuntos de datos sintéticos son la base de buena parte de la investigación en detección de fraude, porque los datos reales etiquetados rara vez son públicos. Si un método basado en grafos se evalúa en un conjunto cuyo grafo no tiene estructura, o cuyo fraude se delata por una regla del generador, el resultado dice poco sobre el método. Esta tesis aporta evidencia sobre cuándo la topología aporta información, identifica atajos en dos conjuntos de uso común (PaySim e IBM AML) y propone un protocolo de evaluación con variables causales y un criterio de hipótesis fijado de antemano.

**Contribución práctica.** Para una entidad financiera, saber qué tipo de variable de red aporta la mayor parte de la señal orienta la inversión: las métricas globales del grafo son costosas de recalcular, mientras que los contadores locales pueden mantenerse en tiempo real. Distinguir entre ambos tipos, como se hace en esta tesis, permite decidir cuál conviene implementar.

## 1.4 Definiciones

- **Fraude financiero**: transacción no autorizada o engañosa cuyo fin es apropiarse de fondos ajenos. En PaySim corresponde a la toma de control de una cuenta y el retiro de sus fondos.
- **Lavado de dinero**: conjunto de transacciones cuyo fin es ocultar el origen ilícito de los fondos, típicamente mediante su dispersión, estratificación e integración en el sistema financiero. En IBM AML y Elliptic se etiquetan como transacciones ilícitas.
- **Grafo de transacciones**: grafo dirigido cuyos nodos son cuentas (o transacciones, en Elliptic) y cuyas aristas representan flujos de dinero con dirección, monto y marca temporal.
- **Variable topológica**: variable calculada a partir de la estructura del grafo y no de los atributos de una transacción aislada. En esta tesis se distinguen tres tipos: **estructurales locales** (grado, *fan-in*/*fan-out*, repetición del par, ciclos cortos), **temporales de red** (tiempo desde la última transacción de la cuenta) y **globales** (PageRank, intermediación, comunidades, componentes). La definición operacional está en 3.4.
- **Variable causal**: variable que, para una transacción en el instante $t$, usa solo información de transacciones anteriores a $t$.
- **Modelo de referencia**: modelo que usa solo variables transaccionales y contra el cual se mide la mejora.

## 1.5 Hipótesis

**H1.** En conjuntos de datos donde el fraude forma estructuras de red, un modelo que incorpora variables topológicas causales detecta las transacciones ilícitas mejor que un modelo que usa solo variables transaccionales, medido por $F_1$ y por el área bajo la curva precisión-exhaustividad (PR-AUC).

La operacionalización y el criterio de decisión se fijaron antes de los experimentos y se describen en 3.7. Las hipótesis rivales, es decir, explicaciones alternativas de una eventual mejora, se presentan en 2.6.

## 1.6 Objetivos

**Objetivo general.** Evaluar, en conjuntos de datos con distinta estructura de red, si las variables topológicas derivadas de grafos dirigidos mejoran la detección de fraude y lavado de dinero, y qué tipo de variable explica esa mejora.

**Objetivos específicos.**

1. Construir grafos dirigidos de transacciones para PaySim, IBM AML (HI y LI) y Elliptic, y caracterizar su estructura (ciclos, nodos intermediarios, componentes).
2. Diseñar variables topológicas causales (estructurales, temporales y globales) sin fuga de información, verificadas mediante pruebas automatizadas.
3. Comparar modelos con y sin variables topológicas bajo un protocolo con partición temporal, umbral elegido en validación y evaluación única en prueba.
4. Determinar la contribución de cada grupo de variables mediante ablación e hipótesis rivales.
5. Delimitar las condiciones en que la topología aporta, mediante el caso de PaySim y un experimento de inyección controlada.

## 1.7 Alcance y limitaciones

**Alcance.** Se estudian tres conjuntos de datos: PaySim (fraude en dinero móvil, sintético), IBM AML (lavado de dinero, sintético, con patrones de red etiquetados; conjunto principal del contraste) y Elliptic (transacciones reales de Bitcoin con variables anonimizadas). El modelo principal es *Random Forest* y el de robustez, *gradient boosting*; no se evalúan redes neuronales de grafos. La implementación usa Python con igraph, pandas y scikit-learn.

**Limitaciones.**
- Dos de los tres conjuntos son sintéticos: sus patrones de fraude son los que implementa el generador.
- El grafo se analiza mediante contadores causales e instantáneas por bloques de tiempo, no como un grafo dinámico continuo.
- La evaluación es en lote (*batch*), no en flujo continuo.
- No se implementa homología persistente sobre filtraciones. La topología se estudia mediante métricas de grafos dirigidos (2.2); de la homología solo se usan los números de Betti $\beta_0$ y $\beta_1$ como descripción de los grafos (2.3).

## 1.8 Organización del documento

- **Capítulo 2. Marco teórico**: conceptos de grafos y métricas topológicas, estado del conocimiento sobre detección de fraude con grafos, vacío que motiva la tesis e hipótesis rivales.
- **Capítulo 3. Metodología**: conjuntos de datos, diseño en dos estudios, variables, modelos, inferencia, criterio de decisión y validez.
- **Capítulo 4. Resultados**: Estudio 1 (PaySim) y Estudio 2 (IBM AML, réplicas, ablación e inyección controlada).
- **Capítulo 5. Discusión**: interpretación de los resultados frente al marco teórico, implicaciones prácticas, estado de las hipótesis rivales y limitaciones.
- **Capítulo 6. Conclusiones**: respuesta a las preguntas de investigación, contribuciones y trabajo futuro.
- **Referencias** y **Anexos**: glosario, catálogo de variables, y código y reproducibilidad.

**Convención numérica.** Se sigue la recomendación de la *Ortografía de la lengua española* (RAE y ASALE, 2010): punto para separar los decimales (0.614) y espacio para agrupar los miles (6 362 620).
