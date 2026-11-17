# 6. Conclusiones

## 6.1 Síntesis

Esta tesis partió de una hipótesis general: que las métricas topológicas de grafos dirigidos mejoran la detección de fraude financiero. El primer estudio, sobre PaySim, mostró que esa hipótesis no podía contrastarse en el conjunto de datos más usado para fraude en dinero móvil, porque su grafo carece de estructura y su fraude se identifica por una regla del simulador. Eso llevó a reformular el problema como una pregunta sobre condiciones y a diseñar un segundo estudio. Su criterio de decisión se fijó antes de los experimentos: un contraste confirmatorio en IBM AML, réplicas en otros dos conjuntos, ablaciones, hipótesis rivales y una inyección controlada de fraude estructurado.

## 6.2 Respuesta a las preguntas de investigación

**P1. ¿Mejoran las variables topológicas la detección en un conjunto con lavado estructurado?** Sí. En IBM AML HI-Small, el modelo con variables topológicas causales supera ampliamente al transaccional en $F_1$ y en PR-AUC. Los intervalos de confianza de ambas diferencias excluyen el cero, de modo que, según el criterio definido de antemano, los resultados apoyan H1.

**P2. ¿Se replica la mejora?** En IBM AML LI-Small, sí, y también en HI-Small sin el recorte temporal. En Elliptic, no de forma robusta: la información de vecinos mejora levemente la PR-AUC, pero la diferencia en $F_1$ depende de la semilla.

**P3. ¿Qué tipo de información de red explica la mejora?** Los contadores locales actualizados con cada transacción, y no las métricas globales del grafo (PageRank, intermediación, comunidades). Dentro de esos contadores, la información estructural y la temporal aportan, cada una por separado, una mejora grande y de magnitud similar, y son complementarias. La estructura sola basta para cumplir el criterio de H1, pero el mejor modelo combina ambas. La variable estructural más informativa es la repetición del par de cuentas: el lavado usa contrapartes nuevas.

**P4. ¿Qué condiciona la mejora?** Los resultados son consistentes con dos condiciones: que el fraude forme estructuras de red que el comportamiento legítimo no reproduzca, y que no sea separable por sus atributos individuales. PaySim incumple ambas. En Elliptic las variables locales ya son muy informativas, y no se pudo establecer si el fraude forma estructuras propias. La inyección controlada muestra que, cuando el fraude es indistinguible a nivel individual, la topología lo detecta casi por completo si el fondo legítimo no contiene estructuras iguales, y que su ventaja disminuye gradualmente a medida que las contiene.

## 6.3 Contribuciones

1. **Evidencia sobre cuándo la topología aporta** a la detección de fraude y lavado, junto con un marco explicativo de dos condiciones que es consistente con los resultados de los cuatro escenarios estudiados y que queda planteado para ser contrastado en trabajos futuros.
2. **Una distinción entre tipos de información de red** (estructural local, temporal y global). Muestra que la mayor parte de la ganancia proviene de variables baratas y calculables en línea, y que la estructura y el tiempo se complementan.
3. **La identificación de atajos** en dos conjuntos sintéticos de uso común: los saldos en PaySim y la cola temporal en IBM AML. También la advertencia de que PaySim no sirve para evaluar métodos basados en grafos.
4. **Un protocolo reproducible**:
   - variables causales verificadas por pruebas;
   - umbral elegido en validación y evaluación única en prueba;
   - criterio de decisión fijado de antemano;
   - hipótesis rivales con sus controles;
   - *bootstrap* pareado, por semilla cuando la varianza lo exige.

## 6.4 Trabajo futuro

- **Contraste prospectivo del marco.** Poner a prueba el marco de dos condiciones en conjuntos de datos que no se usaron para formularlo, midiendo primero sus propiedades estructurales y prediciendo la ganancia antes de entrenar.
- **Datos reales.** Validar con transacciones bancarias etiquetadas, donde los patrones no estén definidos por un generador.
- **Lavado sin estructura.** Estudiar la fracción del lavado que no sigue ninguna tipología y que ningún modelo detectó.
- **Grafo incremental.** Calcular las métricas globales y de ciclos en línea, para determinar si su escaso aporte se debe a la latencia de las instantáneas o a su naturaleza global.
- **Redes neuronales de grafos.** Comparar las variables manuales con modelos que aprenden la representación sobre multigrafos dirigidos (Egressy et al., 2024).
- **Homología persistente.** Construir filtraciones por tiempo o monto sobre subgrafos locales y usar su persistencia como variable.
- **Estabilidad temporal.** Repetir la evaluación con varias particiones mediante ventana deslizante y con ajuste de hiperparámetros.

## 6.5 Reflexión final

La pregunta "¿mejora la topología la detección de fraude?" no tiene una respuesta única. Depende de si el fraude deja en la red una huella que el comportamiento legítimo no deja, y de si esa huella es visible a tiempo. Cuando lo es, como en los patrones de lavado de IBM AML, la información de red no solo mejora la detección: la hace posible, porque las transacciones individuales son indistinguibles de las legítimas. Cuando no lo es, agregar topología no ayuda, y el conjunto de datos puede hacer creer lo contrario si no se examina antes. Esta tesis propone, en consecuencia, que antes de preguntar si un método de grafos funciona, se pregunte si los datos contienen la estructura que ese método busca.
