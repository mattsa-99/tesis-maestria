# 1. Introducción

El fraude financiero ha evolucionado hacia estructuras de red complejas. Esta tesis propone ir más allá del análisis transaccional punto a punto, utilizando la "forma" de los datos (topología) para detectar comportamientos ilícitos.

## 1.1 Contexto y Motivación

Las transacciones financieras fraudulentas representan un desafío creciente en la era digital. Los métodos tradicionales de detección basados en reglas y análisis individual de transacciones presentan limitaciones significativas:

- **Sofisticación de ataques**: Los defraudadores utilizan redes de cuentas coordinadas
- **Imitación de comportamiento legítimo**: Las transacciones individuales pueden parecer normales
- **Alto volumen de falsos positivos**: Sistemas que generan alertas excesivas

Sin embargo, aunque las transacciones individuales puedan parecer legítimas, la **estructura de red** subyacente revela patrones anómalos detectables mediante análisis topológico.

## 1.2 Problema de Investigación

**¿Cómo puede la topología de grafos dirigidos mejorar la detección de fraude financiero en comparación con métodos basados únicamente en características transaccionales?**

Los sistemas actuales fallan ante:
- Ataques coordinados de múltiples cuentas
- Esquemas de lavado de dinero con transacciones circulares
- Redes de "money mules" que dispersan fondos ilícitos
- Comportamientos que imitan patrones humanos a nivel individual pero mantienen estructura de grafo anómala

## 1.3 Hipótesis

Las métricas topológicas extraídas de grafos dirigidos de transacciones (centralidad, detección de comunidades, análisis de ciclos) mejoran significativamente el desempeño de modelos de detección de fraude en comparación con modelos que utilizan únicamente características transaccionales básicas.

## 1.4 Objetivos

### Objetivo General
Desarrollar un sistema de detección de fraude financiero que aproveche las propiedades topológicas de grafos dirigidos para identificar patrones anómalos en redes de transacciones.

### Objetivos Específicos

1. **Construir grafos dirigidos** que representen redes de transacciones financieras del dataset PaySim
   - Nodos: Cuentas bancarias/entidades
   - Aristas dirigidas: Transacciones (origen → destino)
   - Pesos: Montos transaccionales

2. **Extraer métricas topológicas** relevantes para caracterización de nodos y transacciones
   - Centralidad (degree, betweenness, PageRank)
   - Coeficiente de clustering
   - Detección de comunidades
   - Análisis de ciclos

3. **Implementar modelo baseline** de detección de fraude usando Random Forest con features transaccionales básicas
   - Establecer desempeño de referencia
   - Definir métricas de evaluación (F1-Score, Precision, Recall)

4. **Desarrollar modelos mejorados** que incorporen features topológicas
   - Comparar desempeño contra baseline
   - Analizar contribución de cada tipo de feature topológica

5. **Validar experimentalmente** la mejora en detección mediante:
   - Split temporal de datos (80/20)
   - Evaluación con F1-Score como métrica principal
   - Análisis de casos detectados/no detectados

## 1.5 Contribución Esperada

Esta tesis busca demostrar que:
1. El análisis topológico de redes de transacciones revela patrones no capturables por análisis transaccional individual
2. La integración de features topológicas mejora métricas de detección (especialmente recall de fraudes complejos)
3. Es factible implementar este enfoque en sistemas reales de detección de fraude

## 1.6 Alcance y Limitaciones

### Alcance
- Dataset: PaySim (transacciones sintéticas basadas en datos reales)
- Tipo de fraude: Fraude en transacciones móviles
- Enfoque: Análisis topológico de grafos dirigidos
- Implementación: Python con NetworkX y scikit-learn

### Limitaciones
- Datos sintéticos (no transacciones reales)
- No considera aspectos temporales dinámicos del grafo
- Enfoque batch (no streaming en tiempo real)
- No incorpora homología persistente avanzada (trabajo futuro)

## 1.7 Organización del Documento

- **Capítulo 2**: Marco teórico sobre grafos dirigidos, métricas topológicas y detección de anomalías
- **Capítulo 3**: Metodología propuesta y pipeline de procesamiento
- **Capítulo 4**: Datos, experimentación y resultados
- **Capítulo 5**: Conclusiones y trabajo futuro