# Detección de Fraude Financiero - Documentación

## Introducción

Este proyecto implementa un sistema de detección de fraude financiero basado en análisis topológico de grafos dirigidos. El enfoque utiliza propiedades de teoría de grafos y aprendizaje automático para identificar patrones anómalos en redes de transacciones.

## Conceptos Clave

### Grafos Dirigidos

En este proyecto, las transacciones financieras se modelan como grafos dirigidos donde:

- **Nodos**: Representan cuentas bancarias o entidades financieras
- **Aristas dirigidas**: Representan transacciones del nodo origen al nodo destino
- **Pesos**: Indican el monto de la transacción

### Métricas Topológicas

#### Centralidad

1. **Degree Centrality**: Mide la conectividad directa de un nodo
   - En grafos dirigidos: in-degree (receptores) y out-degree (emisores)
   
2. **Betweenness Centrality**: Identifica nodos que actúan como intermediarios
   - Útil para detectar "mulas financieras"

3. **Closeness Centrality**: Mide qué tan cerca está un nodo de todos los demás
   
4. **PageRank**: Importancia de un nodo basada en la estructura de enlaces
   - Especialmente útil en grafos dirigidos

#### Detección de Comunidades

Identifica grupos de nodos densamente conectados:
- Útil para detectar redes de fraude coordinado
- Implementado con algoritmos de propagación de etiquetas

#### Ciclos

La detección de ciclos es crucial para identificar:
- Esquemas de lavado de dinero circular
- Transacciones cíclicas sospechosas

## Métodos de Detección de Anomalías

### 1. Isolation Forest

Algoritmo basado en árboles de decisión que:
- Aísla observaciones anómalas
- No requiere etiquetas (unsupervised)
- Eficiente para datasets grandes

### 2. Análisis Estadístico

Utiliza z-scores para identificar valores atípicos en:
- Características topológicas
- Montos de transacción
- Patrones temporales

### 3. Detección de Patrones

Identifica estructuras específicas sospechosas:
- **Patrones estrella**: Un nodo con muchas conexiones (posible "money mule")
- **Ciclos anómalos**: Transacciones circulares de corta longitud
- **Nodos aislados**: Cuentas dormidas que súbitamente activan

## Uso del Sistema

### Pipeline Básico

```python
# 1. Cargar/generar datos
from src.data_processing import generate_synthetic_transactions
transactions = generate_synthetic_transactions(n_accounts=100, n_transactions=1000)

# 2. Construir grafo
from src.graph_construction import TransactionGraph
tg = TransactionGraph(directed=True)
tg.load_data(transactions)
graph = tg.build()

# 3. Analizar topología
from src.topology_analysis import TopologyAnalyzer
analyzer = TopologyAnalyzer(graph)
centrality = analyzer.compute_centrality_measures()

# 4. Detectar anomalías
from src.anomaly_detection import AnomalyDetector
detector = AnomalyDetector(graph, contamination=0.05)
anomalies = detector.detect_node_anomalies()

# 5. Visualizar resultados
from src.visualization import GraphVisualizer
visualizer = GraphVisualizer(graph)
visualizer.plot_graph(node_colors=anomalies)
```

## Interpretación de Resultados

### Scores de Anomalía

- **Score bajo (más negativo)**: Más anómalo
- Los nodos con scores en el percentil 10 más bajo se consideran sospechosos

### Patrones Sospechosos

1. **Alto out-degree, bajo in-degree**: Posible dispersión de fondos
2. **Alto betweenness**: Intermediario en muchas transacciones
3. **Ciclos cortos**: Posible lavado de dinero
4. **Aislamiento seguido de alta actividad**: Comportamiento atípico

## Métricas de Evaluación

Para evaluar el desempeño del sistema:

- **Precisión**: Proporción de fraudes reales entre las detecciones
- **Recall**: Proporción de fraudes detectados del total
- **F1-Score**: Media armónica de precisión y recall

## Referencias Teóricas

- **Teoría de Grafos**: Newman, M. E. J. (2010). Networks: An Introduction
- **Detección de Anomalías**: Chandola, V., et al. (2009). Anomaly detection: A survey
- **Fraude Financiero**: Bolton, R. J., & Hand, D. J. (2002). Statistical fraud detection

## Futuras Mejoras

1. **Análisis Temporal**: Incorporar evolución de grafos en el tiempo
2. **Homología Persistente**: Aplicar topología algebraica avanzada
3. **Deep Learning**: Graph Neural Networks para detección
4. **Features Adicionales**: Incorporar más metadatos de transacciones
