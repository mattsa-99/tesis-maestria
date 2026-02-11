# Detección de Fraude Financiero mediante Topología y Grafos Dirigidos

## Tesis de Maestría en Ciencia de Datos

### Descripción del Proyecto

Este repositorio contiene el código fuente, análisis y documentación para la tesis de maestría enfocada en la detección de patrones anómalos en transacciones financieras utilizando teoría de grafos y análisis topológico.

### Objetivo

Desarrollar un sistema de detección de fraude financiero que aproveche las propiedades topológicas de grafos dirigidos para identificar patrones anómalos en redes de transacciones.

---

##  Datos y Experimentación

### Fuente de Datos
- **Dataset**: [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) - Simulador de transacciones financieras móviles
- **Tamaño**: ~6.3M transacciones
- **Período**: 30 días de transacciones sintéticas basadas en datos reales
- **Tipos de transacción**: CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT

### Variable Objetivo
- **Variable (y)**: `isFraud` (binaria)
  - `0`: Transacción legítima
  - `1`: Transacción fraudulenta
- **Desbalance de clases**: ~0.13% fraude (clase minoritaria)

### Features Principales
1. **Features transaccionales**:
   - Monto de transacción
   - Tipo de transacción
   - Balance antes/después
   - Diferencia temporal

2. **Features topológicas (extraídas del grafo)**:
   - Degree centrality (in/out)
   - Betweenness centrality
   - PageRank
   - Clustering coefficient
   - Pertenencia a comunidades

---

## Baseline y Métricas

### Modelo Baseline
- **Algoritmo**: Random Forest con features transaccionales básicas (sin topología)
- **Features baseline**: monto, tipo, balance, tiempo
- **Objetivo**: Establecer desempeño mínimo antes de agregar información topológica

### Métrica Principal
- **F1-Score**: Métrica principal debido al alto desbalance de clases
  - Balance entre Precision y Recall
  - Crítico para detectar fraudes (clase minoritaria)

### Métricas Secundarias
- **Precision**: Qué proporción de alertas son fraudes reales
- **Recall**: Qué proporción de fraudes son detectados
- **AUC-ROC**: Capacidad de discriminación del modelo
- **Confusion Matrix**: Análisis detallado de FP/FN

### Validación
- **Estrategia**: Split temporal (80/20)
  - Train: Primeros 24 días
  - Test: Últimos 6 días
- **Justificación**: Simula deployment real (predecir fraudes futuros)
- **Cross-validation**: Estratificado por clase en conjunto de entrenamiento

---

##  Metodología

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

---

## Estructura del Proyecto

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
│   ├── capitulos/    # Capítulos de la tesis
│   └── bitacora/     # Bitácora semanal
├── results/          # Resultados de experimentos
└── config/           # Archivos de configuración
```

---

##  Tecnologías Utilizadas

- **Python 3.9+**: Lenguaje principal
- **NetworkX**: Construcción y análisis de grafos
- **scikit-learn**: Algoritmos de machine learning
- **pandas & numpy**: Manipulación de datos
- **matplotlib & plotly**: Visualización
- **jupyterlab**: Análisis exploratorio
- **pytest**: Testing

---

##  Instalación

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

---

##  Roadmap

- [x] Configuración inicial del proyecto
- [x] Definición de fuente de datos (PaySim)
- [x] Definición de baseline y métricas
- [ ] Descarga y exploración de datos
- [ ] Implementación de procesamiento de datos
- [ ] Construcción de grafos dirigidos
- [ ] Extracción de métricas topológicas
- [ ] Implementación del baseline
- [ ] Algoritmos de detección con features topológicas
- [ ] Validación experimental
- [ ] Documentación de tesis

---

##  Documentación de Tesis

La tesis se documenta en formato Markdown en `docs/capitulos/`:
- [01 - Introducción](docs/capitulos/01%20-%20introduccion.md)
- [02 - Marco Teórico](docs/capitulos/02%20-%20marco-teorico.md)

Bitácora semanal en `docs/bitacora/`

---

##  Licencia

Este proyecto es de uso académico. Todos los derechos reservados.

---

*Última actualización: 11 de febrero de 2026*