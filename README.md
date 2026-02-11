# Detección de Fraude Financiero mediante Topología y Grafos Dirigidos

## Tesis de Maestría en Ciencia de Datos

### Descripción del Proyecto

Este repositorio contiene el código fuente, análisis y documentación para la tesis de maestría enfocada en la detección de patrones anómalos en transacciones financieras utilizando teoría de grafos y análisis topológico.

### Objetivo

Desarrollar un sistema de detección de fraude financiero que aproveche las propiedades topológicas de grafos dirigidos para identificar patrones anómalos en redes de transacciones.

### Metodología

1. **Construcción de Grafos Dirigidos**: Representar transacciones financieras como grafos dirigidos donde:
   - Los nodos representan cuentas/entidades
   - Las aristas dirigidas representan transacciones
   - Los pesos representan montos de transacción

2. **Análisis Topológico**: Aplicar métricas topológicas para caracterizar comportamientos:
   - Centralidad (degree, betweenness, closeness, PageRank)
   - Detección de comunidades
   - Patrones de flujo
   - Ciclos y caminos anómalos

3. **Detección de Anomalías**: Implementar algoritmos para identificar:
   - Transacciones fuera de lo común
   - Estructuras sospechosas de red
   - Comportamientos atípicos en series temporales

### Estructura del Proyecto

```
tesis-maestria/
├── data/               # Datos (no incluidos en el repositorio por privacidad)
│   ├── raw/           # Datos sin procesar
│   ├── processed/     # Datos procesados
│   └── examples/      # Datos de ejemplo generados sintéticamente
├── src/               # Código fuente principal
│   ├── data_processing/      # Procesamiento de datos
│   ├── graph_construction/   # Construcción de grafos
│   ├── topology_analysis/    # Análisis topológico
│   ├── anomaly_detection/    # Detección de anomalías
│   └── visualization/        # Visualización
├── notebooks/         # Jupyter notebooks para análisis exploratorio
├── tests/            # Tests unitarios
├── docs/             # Documentación adicional
├── results/          # Resultados de experimentos
└── config/           # Archivos de configuración
```

### Tecnologías Utilizadas

- **Python 3.9+**: Lenguaje principal
- **NetworkX**: Construcción y análisis de grafos
- **scikit-learn**: Algoritmos de machine learning
- **pandas & numpy**: Manipulación de datos
- **matplotlib & plotly**: Visualización
- **jupyterlab**: Análisis exploratorio
- **pytest**: Testing

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/mattsa-99/tesis-maestria.git
cd tesis-maestria

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete en modo desarrollo
pip install -e .
```

### Uso

```python
# Ejemplo básico de uso
from src.graph_construction import TransactionGraph
from src.anomaly_detection import AnomalyDetector

# Construir grafo desde datos de transacciones
graph = TransactionGraph()
graph.load_data('data/processed/transactions.csv')
graph.build()

# Detectar anomalías
detector = AnomalyDetector(graph)
anomalies = detector.detect()
```

### Roadmap

- [x] Configuración inicial del proyecto
- [ ] Implementación de procesamiento de datos
- [ ] Construcción de grafos dirigidos
- [ ] Métricas topológicas
- [ ] Algoritmos de detección de anomalías
- [ ] Validación experimental
- [ ] Documentación de tesis

### Contribuciones

Este es un proyecto académico personal para tesis de maestría. Sin embargo, sugerencias y comentarios son bienvenidos.

### Licencia

Este proyecto es de uso académico. Todos los derechos reservados.

### Contacto

Para preguntas o colaboraciones, favor contactar al autor del proyecto.

---
*Última actualización: Febrero 2026*