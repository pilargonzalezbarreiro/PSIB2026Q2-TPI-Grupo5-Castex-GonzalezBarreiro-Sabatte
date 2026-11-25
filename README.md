# TPI PSIB 2026 Q2 – Biomarcadores de imagen de nódulos pulmonares sólidos

Detección temprana de cáncer de pulmón: medidas del **interior** y del **entorno** de nódulos sólidos en tomografía (LIDC-IDRI), con técnicas clásicas de procesamiento de imágenes y validación de la segmentación contra la variabilidad entre radiólogos.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No hace falta instalar un numpy viejo para pylidc: `tpi/compat.py` recrea los alias (`np.int`, `np.bool`, `np.float`) antes de importarlo.

## Dos partes

- **`scripts/` — pipeline principal.** Lo que produce los resultados del TP, en orden. No filtra ni comprime nada: mide sobre la imagen original.
- **`comparaciones/` — preguntas de "¿qué método conviene?".** Cada archivo responde una (reconstrucción, filtros, compresión, robustez). No alimentan al pipeline; se corren cuando querés probar algo.

## Pipeline principal (en orden)

| Paso | Script | Qué hace | Necesita imágenes |
|---|---|---|---|
| 0 | `python tests/test_fantoma.py` | Prueba todo con un fantoma sintético | No |
| 1 | `python scripts/01_seleccionar_nodulos.py --max-pacientes 80` | Contextualización + selección de nódulos con las anotaciones de pylidc (~4 min la primera vez) | No |
| 2 | `python scripts/02_descargar_dicom.py --limite 15` | Descarga solo las series seleccionadas (todas: ~11 GB; 15 series: ~2 GB) | — |
| 3 | `python scripts/03_extraer_cortes.py` | Corte de mayor área, consenso, semilla, remuestreo 0,7 mm, ROI 64 mm | Sí |
| 4 | `python scripts/04_caracteristicas.py --ajustar` | Análisis de histogramas, segmentación, validación (umbral de aceptabilidad), medidas de interior y entorno | Sí |
| 1b | `python scripts/01_seleccionar_nodulos.py --ampliar-tamano 8 16` | Agrega a la muestra los nódulos elegibles de 8 a 16 mm (control por tamaño) | No |
| 2–3 | `python scripts/02_descargar_dicom.py` y `python scripts/03_extraer_cortes.py --borrar-dicom` | Bajan solo lo que falta extraer; `--borrar-dicom` borra cada tomografía apenas se extraen sus nódulos | — |
| 3b | `python scripts/03b_encabezados_dicom.py [--descargar --borrar-dicom]` | Parámetros de adquisición de cada corte (fabricante, kVp, mA, mAs, kernel, espesor, CTDIvol): describen la muestra y anclan la simulación de dosis. Solo lee encabezados; `--descargar` vuelve a bajar las series ya borradas | — |
| 5 | `python scripts/05_estadistica.py` | Medida por medida: boxplots, Mann-Whitney + AUC + Holm, control por tamaño (estratos y Spearman parcial). Por defecto solo entrenamiento | Sí |
| 5b / 6b | `... 05_estadistica.py --rango 8 16` y `... 06_puntaje.py --rango 8 16` | Lo mismo, solo con nódulos de tamaño parecido (8–16 mm) | Sí |
| 6 | `python scripts/06_puntaje.py` | Puntaje que combina las mejores medidas: diámetro solo vs interior vs interior + entorno, con validación cruzada por paciente y curvas ROC. `--evaluar` usa el 20 % de evaluación (una sola vez, al final) | Sí |
| 6c | `... 05_estadistica.py --sin-calcio` y `... 06_puntaje.py --sin-calcio` | Solo nódulos sin calcio: la pregunta clínica abierta (el calcio ya lo usan los radiólogos) | Sí |
| 7a | `python scripts/07a_calibrar_dosis.py` | Calibra la constante del tubo (fotones por mAs) con el ruido de los propios estudios de LIDC y la **valida** prediciendo el ruido de los estudios de LIDC que ya son de baja dosis, con y sin ruido electrónico | Sí |
| 7 | `python scripts/07_baja_dosis.py --anclaje mas`, después `--anclaje ruido` y `--anclaje hibrido` | ¿Los biomarcadores sobreviven a la baja dosis (tamizaje) y a la compresión? Lleva cada corte a ½, ¼, 1/10 y 1/20 de la dosis **de su estudio** con el método de Naziroglu et al. (2017): anclaje al mAs del DICOM, compensación del ruido que la imagen ya trae, ruido electrónico, FBP solo del ruido con la ventana del kernel del equipo. Corre todo el pipeline y mide segmentación, reproducibilidad (CCC) y AUC re-armado vs transferido. Lento; después `--solo-analisis` | Sí |
| 8 (plus) | `python scripts/08_propuesta.py [--anclaje hibrido --sigma-e2 lidc] [--dosis-uso 4\|10]` | Propuesta: cuánto ruido aguanta cada medida (CCC e ICC), un puntaje que solo usa medidas robustas y que separan, y un calcio robusto al ruido (mediana + etiquetado + área mínima). Lee los resultados del 07, no cambia nada del pipeline | Sí |
| 9 | `python scripts/09_sensibilidad_umbral_calcio.py` | Calcio: ¿cambia algo con umbrales de 130, 200 y 300 HU? | Cortes |
| 10 | `python scripts/10_sin_pixeles_calcio.py` | Calcio: textura e histograma recalculados SIN los píxeles calcificados (sin descartar nódulos) | Cortes |
| 11 | `python scripts/11_parametros_dicom.py` | Tabla de parámetros de adquisición (kVp, corriente, kernel, espesor, equipo) en total y por grupo, desde `encabezados_dicom.csv` | No |
| 12 | `python scripts/12_dice_vs_cada_radiologo.py` | Dice contra CADA radiólogo (comparación justa) y cuántos nódulos son aceptables con ese criterio | Cortes |
| 13 | `python scripts/13_puntaje_robusto_estricto.py [--anclaje hibrido --sigma-e2 lidc]` | Puntaje robusto con TODA la selección (también el CCC) dentro de cada pliegue | No |
| 14 | `python scripts/14_figura_nitidez_borde.py` | Figura que explica cómo se mide la nitidez del borde | Cortes |
| 15 | `python scripts/15_calcio_informe.py` | Calcio para el informe: especificidad y sensibilidad, AUC, boxplot de tres grupos con Holm, AUC del diámetro por estrato (A5), Spearman parcial con tamaño y calcio como covariables (A6) y umbral de fracción 0-10 % (A7) | No |

## Interfaz gráfica

Abrir `abrir_interfaz.py` y apretar ▶ (la primera vez instala Streamlit). Se abre en el navegador una página que recorre el trabajo completo sobre el nódulo que se elija, con las mismas funciones del pipeline: **1** imagen, histograma y ruido; **2** reconstrucción (sinograma, SBP, FBP rampa/Hann, SART, con menos proyecciones o menos fotones); **3** filtros (mediana, gaussiano, Wiener) sobre una imagen de baja dosis; **4** segmentación paso a paso, con Dice contra el consenso y contra cada radiólogo y Otsu; **5** características (forma, borde, calcio, histograma, GLCM, entorno) comparadas con los dos grupos; **6** el puntaje del 06 armado sin el paciente de ese nódulo; **7** baja dosis (Naziroglu, con los tres escenarios de ruido electrónico) y JPEG, con el cambio de cada medida y del puntaje; **8** las figuras globales. Necesita `datos/cortes/` y `resultados/`. El código está en `interfaz/app.py`.

## Baja dosis, todo junto (sin comandos)

Abrir `correr_baja_dosis.py`, elegir `HASTA` y apretar ▶. Todo queda en `resultados/log_baja_dosis.txt`:

| `HASTA` | Qué corre |
|---|---|
| `"07a"` | 03b, el test en el fantoma y la calibración 07a |
| `"todo"` | 07 con los tres anclajes, comparación de anclajes, 08 e incertidumbre (varias horas; retoma si se corta) |
| `"resto"` | 08 con dosis de uso ×1/10 e incertidumbre acotada |
| `"electronico"` | 07a con la estimación del ruido electrónico en LIDC y el 07 híbrido con ese valor (σe² = 6) y con el límite superior de su intervalo (34) |
| `"final"` | 08, 11 y 13 con el resultado principal (híbrido, σe² de LIDC) |
| `"control"` | AUC con los 112 nódulos aceptables (criterio original) y con los 92 del criterio justo (script 12) |

## Comparaciones (independientes, en cualquier orden salvo la primera)

| Script | Pregunta | Tiempo |
|---|---|---|
| `python comparaciones/reconstruccion.py --casos 3` | ¿Qué método de reconstrucción conviene con mucho ruido o pocos ángulos? (SBP, FBP rampa/Hann, SART). Acá los I0 son nominales: compara MÉTODOS sobre la misma adquisición simulada, no simula fracciones de dosis (eso es el 07). **Guarda las reconstrucciones** para los dos de abajo | lento (~30 s por caso) |
| `python comparaciones/sensibilidad_dosis.py [--carpeta datos/cortes --casos 3]` | ¿Cuánto error mete elegir mal la constante del tubo (Φ) y el bowtie? Fórmula + simulación para Φ; barrido de 49 bowties calibrando como el pipeline; estimación del bowtie desde el perfil del ruido. Sin `--carpeta` usa el fantoma (verdad conocida) | ~3-4 min por caso |
| `python comparaciones/incertidumbre_dosis.py [--carpeta datos/cortes --casos 3]` (por defecto, los tres anclajes) | Intervalo de confianza de la dosis real de cada etiqueta, propagando el rango plausible de TODOS los parámetros que no se pueden medir (bowtie de Kramer, ruido electrónico, kernel, Φ, geometría...). Tornado (uno por vez) + Monte Carlo (todos a la vez) + % de la variabilidad que explica cada uno | ~5 min por caso |
| `python comparaciones/sensibilidad_aceptables.py` | ¿Cambian los AUC principales si se usa el criterio justo de aceptabilidad (92 nódulos en vez de 112)? | ~1 min |
| `python comparaciones/comparar_anclajes.py` | Después de correr el 07 con y sin `--anclaje ruido`: ¿cambian las conclusiones (CCC, AUC) según cómo se anclan los fotones? | segundos |
| `python comparaciones/filtros.py --casos 3` | ¿Qué filtro limpia mejor la baja dosis? Mediana, gaussiano y Wiener en varios tamaños: ruido que queda vs nitidez del borde | segundos |
| `python comparaciones/robustez.py --casos 3` | ¿Cuánto cambian la intensidad y la textura del nódulo con cada reconstrucción, filtro y JPEG? | segundos |
| `python comparaciones/compresion.py` | ¿Cómo guardar o enviar? Huffman, RLE, PNG, JPEG: tamaño y pérdida | segundos |

Resultados en `resultados/comparaciones/` y `figuras/comparaciones/`.

**Decisiones frente a la materia:** `DECISIONES.md` (qué es de la materia, qué hicimos distinto y por qué).

**Recorrido visual:** `notebooks/recorrido_completo.ipynb` muestra cada paso con imágenes sobre un nódulo (abrir en VS Code y ejecutar celda por celda).

**Probar sin imágenes reales** (con el fantoma): agregar `--carpeta tests/salida/cortes` a cualquier script de `comparaciones/`, y para el pipeline:
```bash
python scripts/04_caracteristicas.py --carpeta tests/salida/cortes --meta tests/salida/meta.csv
```

Los resultados quedan en `resultados/` (CSV) y `figuras/` (PNG). Las imágenes DICOM, los cortes `.npz` y las reconstrucciones guardadas no se suben al repo.

## Estructura

```
tpi/                    módulos (cada uno explica en su docstring el porqué de cada decisión)
  config.py             todos los parámetros en un solo lugar
  compat.py             compatibilidad pylidc/numpy y ~/.pylidcrc
  datos.py              selección de nódulos, división por paciente, corte de trabajo
  segmentacion.py       crecimiento de regiones desde semilla (+ Otsu como comparación)
  validacion.py         Dice, Jaccard, concordancia entre radiólogos, aceptabilidad
  caracteristicas.py    tamaño, forma, espículas (apertura), borde (Sobel), intensidad, composición, GLCM
  entorno.py            pulmón, anillos 0–3 / 3–6 mm, estructuras convergentes (esqueleto)
  mejoramiento.py       mediana, gaussiano, Wiener
  reconstruccion.py     HU→μ, Radon, ruido Poisson, SBP, FBP (rampa/Hann), SART
  compresion.py         ventana a 8 bits, Huffman, RLE, PNG, JPEG, RMSE/SSIM
  estadistica.py        Mann-Whitney, AUC, Holm, estratos por tamaño, Spearman parcial
  puntaje.py            puntaje combinado, validación cruzada por paciente, ROC
  visualizacion.py      figuras
  baja_dosis.py         simulación de baja dosis (Naziroglu et al. 2017): sinograma virtual, bowtie, compensación, ruido electrónico, FBP del ruido
  encabezados.py        parámetros de adquisición del DICOM, familia de kernel, bowtie por equipo
scripts/                pipeline principal, 01 … 08, y del 09 al 15 (calcio, adquisición, Dice justo, nitidez del borde, tablas del calcio)
comparaciones/          una pregunta de "¿qué método conviene?" o "¿cuánto error mete?" por archivo
interfaz/app.py         interfaz gráfica (se abre con abrir_interfaz.py)
correr_baja_dosis.py    corre la parte de baja dosis con ▶, sin comandos
tests/test_fantoma.py   fantoma sintético para verificar el código
tests/test_baja_dosis.py  verificación de la simulación de dosis en un fantoma
```

## Dataset

LIDC-IDRI (TCIA, CC BY 3.0). Armato SG 3rd et al., *Medical Physics* 38:915–931, 2011. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
