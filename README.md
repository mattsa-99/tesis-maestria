# Detección de Fraude Financiero mediante Topología y Grafos Dirigidos

**Tesis de Maestría en Ciencia de Datos** · Universidad Autónoma del Estado de México · Mateo Soto Arango

## De qué trata

La tesis estudia **bajo qué condiciones** las variables topológicas de grafos dirigidos de transacciones mejoran la detección de fraude y lavado de dinero frente a un modelo que usa solo variables transaccionales. Se evalúa en tres conjuntos de datos: PaySim, IBM AML (contraste principal) y Elliptic. Además se hace un experimento de inyección controlada de fraude estructurado.

## Resultados principales

| Conjunto | F1 sin topología | F1 con topología | ΔF1 (IC 95 %) | Criterio de H1 |
|---|---|---|---|---|
| **IBM AML HI-Small** (contraste principal) | 0.136 | **0.614** | [+0.448, +0.504] | ✅ cumplido |
| IBM AML LI-Small (réplica) | 0.044 | 0.266 | [+0.196, +0.258] | ✅ cumplido |
| Elliptic (réplica) | 0.751 | 0.766 | cambia de signo según la semilla | ❌ no robusto |
| PaySim (Estudio 1) | 0.998 | 0.995 | [−0.006, −0.002] | ❌ no aplicable |

- La topología mejora la detección **cuando el fraude tiene estructura de red**: en IBM AML la exhaustividad por patrón sube de 13–28 % a 76–93 %. El lavado sin patrón (38 %) sigue sin detectarse.
- La ganancia viene de **contadores causales locales**, tanto estructurales (grado, *fan-in*/*fan-out*, repetición del par) como temporales (tiempo entre transacciones), y no de métricas globales (PageRank, intermediación, comunidades).
- PaySim no sirve para evaluar métodos de grafos: su grafo es un bosque sin ciclos y su fraude se delata por un artefacto de saldos del simulador.

Detalle en los capítulos 4 (Resultados) y 5 (Discusión).

## Documento

Los capítulos están en Markdown en [`docs/capitulos/`](docs/capitulos/):

- [Portada](docs/capitulos/00%20-%20portada.md) · [Resumen / Abstract](docs/capitulos/00%20-%20resumen.md)
- [1. Introducción](docs/capitulos/01%20-%20introduccion.md)
- [2. Marco teórico](docs/capitulos/02%20-%20marco-teorico.md)
- [3. Metodología](docs/capitulos/03%20-%20metodologia.md)
- [4. Resultados](docs/capitulos/04%20-%20resultados.md)
- [5. Discusión](docs/capitulos/05%20-%20discusion.md)
- [6. Conclusiones](docs/capitulos/06%20-%20conclusiones.md)
- [Anexos](docs/capitulos/07%20-%20anexos.md): glosario, catálogo de variables, y código y reproducibilidad
- [Referencias](docs/capitulos/08%20-%20referencias.md): se generan desde `referencias/bibliografia.bib` con `python docs/herramientas/generar_referencias.py`

La bitácora del proceso está en [`docs/bitacora/`](docs/bitacora/).

## Estructura del repositorio

```
tesis-maestria/
├── docs/
│   ├── capitulos/      # Tesis (Markdown) e imágenes
│   ├── bitacora/       # Bitácora semanal
│   └── herramientas/   # Figuras del Cap. 2, referencias y formato de fórmulas para GitHub
├── experiments/        # Variables causales, experimentos y figuras
├── tests/              # Pruebas de causalidad de las variables
├── results/            # Métricas, intervalos de confianza y figuras
├── referencias/        # bibliografia.bib
├── notebooks/          # Material del proceso (semana 2)
└── data/               # raw/ y processed/ (no versionados)
```

## Reproducir

```bash
git clone https://github.com/mattsa-99/tesis-maestria.git
cd tesis-maestria
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests                        # pruebas de causalidad
```

Los conjuntos de datos se descargan con `kagglehub` en `data/raw/_kagglehub/`. Los identificadores, el entorno exacto y el script que produce cada tabla y figura están en el **Anexo C** de la tesis. Los comandos principales:

```bash
python -m experiments.features                # variables de PaySim
python -m experiments.run_experiments         # Estudio 1 (PaySim)
python -m experiments.run_t3_vs_b1            # contraste de H1 en PaySim
python -m experiments.run_no_shortcut         # PaySim sin atajo (post hoc)
python -m experiments.verify_paysim_by_type   # exhaustividad por tipo en PaySim sin atajo
python -m experiments.ibm_features HI-Small   # variables de IBM (también LI-Small, --full)
python -m experiments.run_ibm HI-Small        # contraste confirmatorio
python -m experiments.run_ibm_struct_vs_time  # estructurales frente a temporales (post hoc)
python -m experiments.run_elliptic            # réplica en Elliptic
python -m experiments.injection               # inyección controlada
python experiments/make_figures.py            # figuras de resultados
python docs/herramientas/figuras_marco.py     # figuras y valores del marco teórico (Cap. 2)
```

Si se editan fórmulas, `python docs/herramientas/compatibilizar_github.py --aplicar` las pasa a la sintaxis que GitHub renderiza (`--revertir` vuelve a `$...$`).

## Pendiente

- [ ] Verificar contra la fuente las referencias marcadas en `bibliografia.bib`
- [ ] Revisión con el director

## Licencia

Uso académico. Todos los derechos reservados.

*Última actualización: 17 de noviembre de 2026*
