# Bitácora Semana 1
**Fecha**: 11 de febrero de 2026
**Autor**: Matías Sánchez

## Resumen Semanal

Primera semana de trabajo en la tesis de maestría. Se establecieron las bases del proyecto y se definieron los componentes fundamentales de la investigación.

## Actividades Realizadas

### 1. Configuración del Proyecto ✅
- ✅ Creación del repositorio GitHub: `mattsa-99/tesis-maestria`
- ✅ Estructura inicial de directorios
- ✅ Configuración de entorno Python
- ✅ Instalación de dependencias base (NetworkX, scikit-learn, pandas)

### 2. Definición de Datos ✅
- ✅ **Fuente seleccionada**: PaySim Dataset (Kaggle)
  - 6.3M transacciones sintéticas
  - Basado en datos reales de sistema de pagos móviles
  - Disponible públicamente
- ✅ **Variable objetivo definida**: `isFraud` (binaria: 0=legítimo, 1=fraude)
- ✅ **Características del dataset identificadas**:
  - Tipos de transacción: CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT
  - Desbalance de clases: ~0.13% fraude
  - Período: 30 días de transacciones

### 3. Establecimiento de Baseline ✅
- ✅ **Modelo baseline definido**: Random Forest con features transaccionales básicas
- ✅ **Features baseline**:
  - Monto de transacción
  - Tipo de transacción
  - Balance antes/después
  - Diferencia temporal
- ✅ **Objetivo**: Establecer desempeño de referencia antes de agregar topología

### 4. Definición de Métricas ✅
- ✅ **Métrica principal**: F1-Score
  - Justificación: Alto desbalance de clases (fraude es minoritario)
  - Balance entre Precision y Recall
- ✅ **Métricas secundarias**:
  - Precision (minimizar falsos positivos)
  - Recall (detectar máximo de fraudes)
  - AUC-ROC (capacidad discriminativa)
  - Confusion Matrix (análisis FP/FN)

### 5. Estrategia de Validación ✅
- ✅ **Enfoque**: Split temporal (80/20)
  - Train: Primeros 24 días
  - Test: Últimos 6 días
- ✅ **Justificación**: Simula deployment real (predecir fraudes futuros)
- ✅ Cross-validation estratificado en conjunto entrenamiento

### 6. Documentación ✅
- ✅ README.md actualizado con:
  - Descripción del proyecto
  - Fuente de datos y variable objetivo
  - Baseline y métricas
  - Estrategia de validación
- ✅ Capítulo 1 (Introducción) expandido con:
  - Problema de investigación
  - Hipótesis
  - Objetivos generales y específicos
  - Alcance y limitaciones

## Decisiones Técnicas

### Dataset: ¿Por qué PaySim?
1. **Disponibilidad pública**: No requiere acuerdos de confidencialidad
2. **Documentado**: Paper académico con descripción detallada
3. **Realista**: Basado en logs reales de sistema africano de pagos móviles
4. **Volumen adecuado**: 6M+ transacciones (suficiente para análisis topológico)
5. **Etiquetado**: Incluye ground truth de fraudes

### Métrica: ¿Por qué F1-Score?
- **Desbalance extremo**: 0.13% fraude vs 99.87% legítimo
- **Costo asimétrico**: No detectar fraude es más costoso que falsa alarma
- **Balance necesario**: Accuracy no es representativo en desbalance
- F1-Score pondera igualmente Precision y Recall

### Validación: ¿Por qué Split Temporal?
- **Realismo**: En producción, modelos predicen futuro (no pasado)
- **Evita data leakage**: Train/test temporal estricto
- **Detecta concept drift**: Si desempeño baja, puede haber cambio de patrones

## Próximos Pasos (Semana 2)

### Prioridad Alta
1. 🔄 Descargar dataset PaySim desde Kaggle
2. 🔄 Análisis exploratorio de datos (EDA)
   - Distribución de clases
   - Estadísticas descriptivas
   - Visualizaciones básicas
3. 🔄 Implementar pipeline de procesamiento
   - Limpieza de datos
   - Feature engineering básico
   - Split temporal

### Prioridad Media
4. 🔄 Construir grafos dirigidos iniciales
   - NetworkX implementation
   - Visualización de subgrafos pequeños
5. 🔄 Implementar modelo baseline
   - Random Forest con features transaccionales
   - Evaluación con métricas definidas

### Prioridad Baja
6. 🔄 Comenzar extracción de métricas topológicas
   - Degree centrality
   - PageRank básico

## Aprendizajes de la Semana

1. **Importancia de dataset público**: Facilita reproducibilidad y evita trabas legales
2. **Necesidad de baseline claro**: Comparación cuantitativa requiere punto de referencia
3. **Desbalance de clases es crítico**: Debe guiar elección de métricas y estrategia
4. **Documentación temprana**: Escribir decisiones mientras están frescas

## Bloqueadores y Riesgos

### Bloqueadores Actuales
- Ninguno 🎉

### Riesgos Identificados
1. **Riesgo computacional**: 6M transacciones → grafo potencialmente muy grande
   - Mitigación: Sampling inicial, análisis por ventanas temporales
2. **Riesgo de desbalance**: 0.13% fraude → modelos pueden ignorar clase minoritaria
   - Mitigación: SMOTE, class weights, métricas adecuadas
3. **Riesgo de overfitting en topología**: Features topológicas pueden ser muy específicas
   - Mitigación: Validación estricta, regularización

## Métricas de Progreso

- ✅ Repositorio creado y configurado: **100%**
- ✅ Dataset definido: **100%**
- ✅ Baseline definido: **100%**
- ✅ Métricas definidas: **100%**
- ✅ Validación definida: **100%**
- ✅ Documentación v0: **100%**
- 🔄 Código implementado: **5%** (solo estructura inicial)

## Notas Adicionales

- Repositorio público en: https://github.com/mattsa-99/tesis-maestria
- Todos los commits están documentados
- Estructura de proyecto sigue best practices de ML

---

**Horas invertidas esta semana**: ~8 horas
**Estado general**: ✅ En tiempo, objetivos cumplidos