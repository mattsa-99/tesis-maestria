# Bitácora Semana 3
**Fecha**: 16 de noviembre de 2026
**Autor**: Mateo Soto Arango

## Resumen

Se descargó PaySim completo, se construyó el pipeline causal de features topológicas, se corrieron
los experimentos con ablación e inferencia y se redactaron los capítulos 2-5.

## Hallazgos

- **El baseline de la semana 2 no es válido.** Se entrenó sobre un proxy aleatorio (F1 0.476). Con PaySim
  real el baseline alcanza F1 ≈ 0.997.
- **Artefacto del simulador**: los fraudes vacían la cuenta de origen (`amount ≈ oldbalanceOrg`);
  `amount_to_orig_ratio` e `insufficient_funds` suman ~58 % de la importancia.
- **Grafo acíclico y casi bipartito**: 0 ciclos, 1 769 cuentas origen+destino.
- **Contraste**: ΔF1(T3 - B1) en [-0.0057, -0.0016] → H0.
- **Análisis post hoc sin atajo**: ΔF1 ≈ +0.07, solo precisión; aporta el grado del destino, no PageRank/betweenness.

## Decisiones

| Decisión | Justificación |
|---|---|
| Features causales (solo pasado) | Evitar leakage temporal en grafo |
| Instantáneas diarias con rezago de 24 h | Costo computacional de 6.3 M aristas |
| Betweenness exacta solo en componentes con nodos puente | Equivalente al cálculo global, viable |
| Comparar contra B1 (no B0) | Aislar efecto topológico del de `dest_is_merchant` |
| Reportar análisis sin atajo como exploratorio | Se diseñó después de ver los resultados |

## Incidencias

- Bug detectado por tests: `snapshot_features` no creaba las columnas de ciclo tras excluirlas de `SNAPSHOT`.
- El bootstrap inicial comparó contra B0 en lugar de B1; se corrigió con `run_t3_vs_b1.py`.
- `tests/test_data_processing.py::test_deterministic_generation` falla (preexistente: `timestamp` usa `datetime.now()`).

## Pendiente

- [ ] Verificar la bibliografía contra las fuentes originales.
- [ ] Revisión con el director.
- [ ] Decidir si se añade TDA con homología persistente.

---

## Extensión (16–17 de noviembre de 2026): contraste en datos con estructura

Plan aprobado: Línea 1 (IBM AML como principal, LI-Small y Elliptic como réplicas) y Línea 2 (inyección controlada en PaySim). El criterio de H1 se escribió en el Capítulo 3, sección 3.8, antes de correr los experimentos.

### Resultados
| Dataset | F1 B | F1 B+topología | ΔF1 IC 95 % | Criterio |
|---|---|---|---|---|
| IBM HI-Small (principal) | 0.136 | 0.614 | [+0.448, +0.504] | ✅ H1 |
| IBM LI-Small | 0.044 | 0.266 | [+0.196, +0.258] | ✅ |
| HI-Small sin recorte | 0.238 | 0.717 | [+0.454, +0.499] | ✅ |
| Elliptic | 0.751 | 0.766 | cambia de signo según semilla | ❌ no robusto |
| Inyección (9 escenarios) | ≤ 0.07 | 0.59–0.97 | todos > 0 | ✅ prueba de capacidad |

### Hallazgos
- Recall por patrón en IBM: de 13–28 % a 76–93 %; el lavado sin patrón (38 %) no se detecta.
- La señal viene sobre todo de contadores causales (`pair_prev`, tiempos, fan-out), no de PageRank, intermediación o comunidades.
- Atajo temporal en IBM tras el 2022-09-10: se recortó (el baseline sube de 0.136 a 0.238 sin el recorte).

### Incidencias
- LI-Small: faltaba el archivo de patrones; el recall por patrón no se calculó.
- Elliptic: el bootstrap con una sola semilla era engañoso; se pasó a un bootstrap por semilla.
- Inyección v1: el inicio uniforme creaba un atajo temporal (baseline con F1 0.41); se corrigió en la v2.

### Pendiente
- [ ] Verificar la bibliografía (incluida Altman et al., 2023).
- [ ] Revisión con el director.
- [ ] Commit del trabajo.

---

## Revisión con la skill thesis-writing (17 de noviembre de 2026)

### Cambios de estructura
- Capítulos reorganizados en 0 Resumen/Abstract, 1 Introducción, 2 Marco teórico, 3 Metodología, 4 Resultados, 5 Discusión, 6 Conclusiones y 7 Anexos (glosario y catálogo de variables).
- La pregunta de investigación se reformula como "¿bajo qué condiciones…?", con sub-preguntas P1–P4.
- Marco teórico:
  - preliminares matemáticos (22 definiciones, 3 proposiciones con demostración, 1 corolario);
  - estado del conocimiento organizado por ideas;
  - vacío y modelo conceptual;
  - cinco hipótesis rivales.
- La metodología se integra en dos estudios, con definición operacional de las variables topológicas, validez interna y externa, y *bootstrap* formalizado.
- Los resultados quedan sin interpretación (14 tablas y 7 figuras numeradas); la interpretación pasa al capítulo de Discusión.
- Terminología unificada en español (variable, modelo de referencia, partición, exhaustividad, intermediación, etc.).

### Nuevo experimento (post hoc, hipótesis rival R1)
| Configuración IBM HI-Small | F1 | PR-AUC |
|---|---|---|
| B | 0.136 | 0.062 |
| B + estructurales | 0.432 | 0.409 |
| B + temporales | 0.450 | 0.373 |
| B + estructurales + instantáneas | 0.459 | 0.445 |
| T3 completo | 0.602 | 0.563 |

R1 queda parcialmente apoyada: estructura y tiempo aportan cada uno una mejora similar y son complementarios. La estructura sola también cumple el criterio de H1.

### Correcciones
- Error conceptual: los ciclos dirigidos no son el número de Betti β1. Se calcularon β0 y β1: PaySim es un bosque (β1 = 0) e IBM HI-Small tiene β1 = 242 796.
- Proporción diaria de ilícitas en la cola de IBM: 57–73 % (no 30–80 %).
- Valor de la ablación de LI-Small (T3, semilla 42): 0.270 / 0.227.

### Pendiente (autor)
- [ ] Completar 2.5.1 con 3–5 trabajos adicionales y las cifras de Altman et al. (2023) y Egressy et al. (2024).
- [ ] Verificar las 33 referencias del .bib contra las fuentes originales.
- [ ] Revisión con el director.

### Actualización (17-nov-2026, tarde)
- [x] Sección 2.5 completada con literatura verificada contra las fuentes: Altman et al. (2023) y Egressy et al. (2024), con sus cifras en la Tabla 2.1; Van Vlasselaer et al. (2015); Pourhabibi et al. (2020); Starnini et al. (2021); Cardoso et al. (2022).
- Corrección: los trabajos sobre IBM AML **no** reportan un modelo solo transaccional (antes se afirmaba lo contrario). Ese vacío lo llena esta tesis.
- Egressy et al. (2024) confirman de forma independiente que el 38 % del lavado de HI-Small no tiene patrón y que su exhaustividad es cercana a 0.
- Convención numérica: se probó la latinoamericana (coma decimal, punto de miles) y se revirtió a la de la RAE (2010): punto decimal y espacio para los miles. Se verificó la ida y vuelta sin pérdidas.
- [ ] Pendiente: verificar las 31 referencias restantes del .bib (las 6 de esta actualización ya están verificadas).
