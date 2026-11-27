# Decisiones del pipeline frente a la materia

Criterio general:
1. **La base es la materia.** No reemplazamos un método de la materia por uno equivalente de afuera.
2. **Construimos encima.** Donde el problema pide más (clasificar nódulos), agregamos medidas y pasos nuevos, preferentemente armados con herramientas de la materia usadas de otra forma (lo que borra la apertura, el esqueleto del entorno, el gradiente del borde).
3. **Si el problema real obliga a hacer algo distinto** (nódulos chicos, distinto píxel, HU), se hace y se justifica acá.
4. **Lo que agregamos no se da por bueno de antemano:** la estadística (05) y el puntaje (06) dicen si sirvió.

## 1. Igual que en la materia

| Etapa | Qué usamos | Sección del resumen |
|---|---|---|
| Análisis | Mirar histogramas y tipo de ruido antes de elegir métodos | 2 |
| Reconstrucción | Beer-Lambert, Radon, sinograma, SBP, FBP con rampa (Ram-Lak) y con Hann, SART | 1.3–1.10 |
| Mejoramiento | Mediana (impulsivo), gaussiano (granulado) y Wiener (adaptativo), elegidos según el tipo de ruido | 3 |
| Segmentación | Crecimiento de regiones desde semilla (conectividad 8); Otsu como comparación | 4.3, 4.5, 4.10 |
| Posprocesamiento | Apertura (cortar vasos pegados), etiquetado, eliminar objetos chicos por área, cierre | 5.2–5.5 |
| Evaluación | Dice y Jaccard | 4.11 |
| Características | Área, intensidad media y desvío (máscara × original), GLCM | 6.6, 6.7 |
| GLCM | 16 niveles, distancia 1, 4 direcciones promediadas, simetrizada y normalizada, solo dentro de la región; energía, entropía, contraste, homogeneidad, correlación, cluster shade y prominence | 6.2–6.4 |
| Entorno | Anillos por dilatación ("margen alrededor de un tumor"), calculada con la transformada de distancia (dilatación con disco euclídeo exacto; el disco de píxeles cambiaba los anillos un 11–17 %) | 5.3 |
| Compresión | Huffman, RLE, PNG (sin pérdida), JPEG por calidades (con pérdida) | 7 |

## 2. Construido encima de la materia

| Qué agregamos | Herramienta de la materia en la que se apoya | Para qué |
|---|---|---|
| **Circularidad, solidez, excentricidad** | Propiedades de la región (área, perímetro, envolvente, momentos) | La materia mide área y bounding box; el signo de malignidad es el **borde** (lobulado, espiculado) y la forma. Excentricidad mide alargamiento, algo que la circularidad no ve (un óvalo liso) |
| **Fracción fina** (`frac_fina`) | Apertura: "elimina estructuras delgadas" | En vez de usar la apertura solo para limpiar, se mide **lo que borra**: la parte del nódulo de menos de ~3 mm de ancho = espículas |
| **Nitidez del borde** (HU/mm) | Detección de bordes (discontinuidad), gradiente de Sobel | Versión objetiva del "margen" que califican los radiólogos: borde nítido vs difuso |
| **Estructuras convergentes** (cuántas llegan al nódulo, longitud por mm de borde) | Esqueletización ("vasos: esqueleto → largo, ramificaciones") + etiquetado | Signo de convergencia vascular y espículas que se extienden: el nódulo maligno tracciona el tejido |
| **Forma con y sin apertura** (`*_cruda`) | Apertura | La apertura corta vasos pero también afeita espículas; medir las dos versiones muestra cuánta información se pierde |
| **Asimetría y curtosis** de las HU | Histograma de la región | Forma del histograma (colas por necrosis, calcio, burbujas) que la media y el desvío no capturan. Riesgo: con ~60 píxeles pueden ser ruidosas |
| **Fracción de calcio y de aire** | Umbral sobre la región | Específicas de TC: calcio → suele ser benigno; aire/burbujas → puede ser sospechoso |
| **Puntaje combinado** (script 06) | "Vector de características para clasificación" | Combina las k medidas más discriminativas (no repetidas) en un número; se compara diámetro solo vs interior vs interior + entorno con validación cruzada por paciente y curvas ROC |
| **Biomarcadores bajo baja dosis y compresión** (script 07) | Reconstrucción (Radon, Beer-Lambert, ruido de Poisson, FBP) + compresión JPEG + todo el pipeline | El tamizaje se hace con baja dosis. Cada corte real se lleva a una fracción de la dosis de su propio estudio con el método de Naziroglu et al. (2017), ver sección 5 (y se comprime), se corre el pipeline completo y se mide: segmentación, reproducibilidad de cada medida (CCC de Lin) y AUC del puntaje **re-armado** con esa dosis y **transferido** (armado con dosis normal, aplicado a baja dosis: lo que pasa cuando un modelo publicado se usa en tamizaje) |
| **Propuesta (script 08, plus)** | Mediana (valores aislados), etiquetado y eliminación de objetos chicos por área, el mismo puntaje del 06 | Responde a los dos problemas del 07: un puntaje que solo elige medidas reproducibles a la dosis de un tamizaje (CCC ≥ 0,9 a ×1/4) y que separan (AUC ≥ 0,7), y un calcio que cuenta solo manchas de calcio y no píxeles sueltos de ruido. También reporta el ICC prometido en el anteproyecto (da casi igual que el CCC) |
| **Robustez** (`comparaciones/robustez.py`) | "El filtrado cambia la textura; JPEG altera la GLCM" | Se cuantifica: misma máscara, medidas sobre cada reconstrucción, filtro y JPEG |
| **Evaluación de filtros** (`comparaciones/filtros.py`) | Elegir filtro según el ruido | Además de RMSE/PSNR/SSIM: ruido residual en el parénquima y nitidez del borde del nódulo, con varios tamaños de cada filtro |

## 3. Distinto de la materia, porque el problema lo pide

| Decisión | Qué dice la materia | Qué hicimos y por qué |
|---|---|---|
| Nódulos pegados a la pared | Region growing "se escapa con bordes débiles"; "de lo grande a lo chico" (4.10) | Con datos reales, ~30 % de los nódulos (yuxtapleurales) se escapaban por toda la pared del tórax (Dice < 0,3, región de 40–48 mm). **Se segmenta primero el pulmón y se le aplica un cierre con disco grande (rolling ball)** que rellena la muesca del nódulo sin entrar en la pared; el nódulo solo crece adentro. El radio del cierre (10/15/20/30 mm) se ajusta en entrenamiento junto con T. El cierre se calcula con la transformada de distancia (mismo resultado, disco euclídeo exacto, instantáneo) |
| Criterio del crecimiento de regiones | "Diferencia con la semilla ≤ tolerancia" | **HU > T.** No depende del valor de un único píxel (la semilla puede caer sobre calcio o ruido). Es "parecido al tejido" con tolerancia fija. T se ajusta con el conjunto de entrenamiento |
| Huecos internos | Cierre | **Relleno de huecos.** El cierre también suaviza el borde y tapa los entrantes entre lóbulos, que es justo lo que mide la forma. Las cavidades internas son parte del nódulo |
| Rango de cuantización de la GLCM | Reducir a 8–16 niveles | 16 niveles, pero **repartidos en [−500, 300] HU para el nódulo y [−1000, 0] HU para el entorno**. Con [−1000, 400], cada nivel vale 87 HU y el interior del nódulo cae en 2–3 niveles. El rango es fijo e igual para todos los nódulos (condición de la materia) |
| Iteraciones de SART | "Suele bastar una" | Con 180 vistas, sí. **Con 60 vistas, una iteración subestima las HU** (no convergió): se muestran 1 y 3 iteraciones |
| ART | Método de la materia | **No implementado.** Requiere armar A (rayos × píxeles): ~92.000 × 262.144 para 512×512. SART resuelve el mismo sistema sin armar A |
| Perímetro para circularidad y solidez | Área y perímetro de la región | Contar píxeles de borde subestima el perímetro de objetos chicos: un círculo perfecto de 5 mm daba circularidad 1,25 y uno de 20 mm 0,91, y en datos reales los benignos (chicos) salían con circularidad > 1. **Área y perímetro del mismo contorno sub-píxel** (isolínea 0,5 de la máscara apenas suavizada): el círculo da 0,96–0,99 en todos los tamaños y la circularidad no puede pasar de 1 |
| Calcio como confundidor | — (no está en la materia) | Con 217 nódulos, el 57 % de los benignos tiene calcio (> 200 HU y más del 2 % del área) contra el 17 % de los sospechosos (en los 112 nódulos de desarrollo con segmentación aceptable: 61 % y 13 %). Las medidas que "separan más allá del tamaño" (contraste GLCM, asimetría, desvío, nitidez del borde) están asociadas al calcio: la separación se explica **en gran parte, no por completo**, por el calcio. Pero el calcio es justamente lo que el radiólogo ya mira (y Lung-RADS ya clasifica como benigno): encontrarlo no es un hallazgo. **Se agrega el análisis `--sin-calcio` (05 y 06)**, que responde la pregunta clínica abierta: en nódulos sólidos SIN calcio, ¿algo separa más que el diámetro? |
| Umbrales del calcio (200 HU y 2 %) | — | Son una elección operativa, inspirada en el puntaje de Agatston (130 HU, área mínima de 1 mm²), no validada contra radiólogos. **Se varían**: el 2 % entre 0 y 10 % (script 15, Tabla A7: AUC del diámetro sin calcio de 0,938 a 0,941) y los 200 HU entre 130 y 300 HU (script 09) (0,931 a 0,942; nitidez del borde 0,51 a 0,55). Las conclusiones no cambian |
| Calcio: tablas y figura del informe (script 15) | — | Todo lo que el informe dice del calcio sale de un script, sin descartar nódulos: especificidad (88 %) y sensibilidad (61 %) del calcio, su AUC (0,74 presencia, 0,77 fracción), boxplot de tres grupos con Mann-Whitney y Holm (contraste: con vs sin calcio p < 0,001, sin calcio vs sospechosos p = 0,051; asimetría: p = 0,003 y p < 0,001), AUC del diámetro por estrato de calcio y tamaño (Tabla A5), Spearman parcial controlando tamaño, calcio y ambos (Tabla A6, el calcio como covariable que propuso la devolución: solo la asimetría conserva algo, −0,21, p = 0,028, que con Bonferroni o Holm pasa a 0,23) y el umbral de fracción de 0 a 10 % (Tabla A7) |
| Textura sin los píxeles de calcio | — | Se recalculan las medidas del interior sacando de la máscara el calcio (script 10), sin descartar nódulos. En los 34 benignos calcificados el calcio no son "unos pocos píxeles": ocupa una mediana del 53 % del área (más de la mitad en 20 de 34). Sacando solo los píxeles > 200 HU queda el borde de transición (volumen parcial: el desenfoque del equipo deja un anillo de ~100-200 HU) y el contraste GLCM de esos nódulos **sube** (mediana 20,5 → 24,0). Sacando también **1 píxel alrededor**, cae a 9,6 (benignos sin calcio: 7,2; sospechosos: 5,8) y el AUC del contraste baja de 0,85 a 0,71, igual que en el subgrupo sin calcio (0,69); con 2 píxeles, 9,3 y 0,69 (estable). **El contraste separaba en gran parte porque detectaba el calcio y su volumen parcial**, como señaló la devolución. La **asimetría** no se explica así: su AUC se mantiene (0,86 → 0,81) y los benignos calcificados siguen con asimetría positiva (0,29), coherente con la Tabla A6. Sin el calcio y su anillo, el desvío de las HU pasa a ser mayor en los sospechosos (AUC 0,77): se reporta con cautela, porque las regiones que quedan son más chicas |
| Dice contra cada radiólogo | Dice y Jaccard (4.11) | Contra el consenso el algoritmo parte con ventaja. El script 12 calcula el Dice contra cada radiólogo y lo compara con la concordancia entre radiólogos: **0,80 contra 0,835 entre radiólogos** (0,83 contra el consenso). El algoritmo queda cerca de los radiólogos, algo por debajo. Con el criterio justo los aceptables bajan de 112 a 92 (desarrollo) y de 25 a 18 (evaluación); el análisis principal se mantiene con el criterio original y se repite como control con los 92: las conclusiones sobre la baja dosis no cambian, pero la ventaja de la textura sobre el diámetro en la imagen original se achica de 0,09 a 0,04 de AUC (ver “Control: criterio de aceptabilidad” en la sección 6) |
| Figura de la nitidez del borde | Sobel | El script 14 dibuja, para el nódulo de mayor nitidez (LIDC-IDRI-0822_s833_n6, 1042 HU/mm) y el de menor (LIDC-IDRI-0031_s42_n4, 184 HU/mm), el contorno, la banda de ~3 px donde se mide y el gradiente |
| "Entorno solo" en el puntaje | — | El puntaje toma las k mejores medidas; si las del entorno son más débiles, "interior + entorno" sale idéntico a "interior" y no se sabe si el entorno tiene señal. Se agrega un puntaje solo con el entorno |
| Radio del cierre del pulmón | — | Con 20/30/40 mm el Dice bajaba al crecer el radio y el mejor quedó en el borde de la grilla: se prueba 10/15/20/30 mm. Las fugas que quedan son de nódulos pegados al mediastino o al hilio: un disco grande rellena también esa concavidad |
| Control por tamaño | Estratos de diámetro (anteproyecto) | En LIDC los benignos son casi todos < 10 mm y los sospechosos > 10 mm: con la muestra original casi no hay nódulos de tamaño parecido en los dos grupos y los estratos quedan vacíos. **Se amplía la muestra con todos los nódulos elegibles de 8 a 16 mm** (81 benignos y 75 sospechosos en total) y se compara dentro de ese rango (`--rango 8 16` en 05 y 06), con el diámetro de los radiólogos para no depender de la segmentación |
| Ecualización | Para bajo contraste | **No se usa.** Cambia los valores de cada imagen de forma distinta y las HU dejarían de ser comparables. En su lugar, ventana de HU fija para visualizar y para pasar a 8 bits |
| Filtrado antes de medir | "Si no hace falta, no filtres" | **No se filtra antes de segmentar ni de medir.** El ruido (~15–30 HU) es chico frente al salto pulmón–tejido (~850 HU) |
| Organización | — | **Pipeline (`scripts/`) separado de comparaciones (`comparaciones/`).** El pipeline mide sobre la imagen original; las comparaciones responden "¿qué método conviene?" sin alimentar los resultados |

## 4. Fuera de la materia (imprescindible para el problema)

| Qué | Por qué |
|---|---|
| Remuestreo a 0,7 mm | Cada tomógrafo tiene otro tamaño de píxel. Sin esto, las medidas en píxeles no son comparables |
| Consenso de radiólogos (≥ 50 %) y umbral de aceptabilidad | Referencia para validar: aceptable si Dice(algoritmo, consenso) ≥ Dice medio entre radiólogos. **Esta comparación favorece al algoritmo**, porque el consenso se arma con los mismos radiólogos. Por eso el script 12 compara también contra cada radiólogo por separado (ver sección 3) |
| División entrenamiento / evaluación por paciente | Ajustar con unos nódulos y evaluar con otros, sin que el mismo paciente quede en los dos |
| Métricas de calidad de imagen | RMSE (en HU), PSNR (estándar en compresión, figura en el anteproyecto) y SSIM (distingue ruido de borroneo, que RMSE mezcla) |
| Estadística (Mann-Whitney, AUC, Holm, estratos por tamaño, Spearman parcial) | El paso "comparación" con ~100 nódulos y ~40 medidas necesita una prueba que no suponga normalidad, corrección por comparaciones múltiples y dos controles del tamaño (confundidor), como promete el anteproyecto |
| Validación cruzada por paciente y bootstrap | Para evaluar el puntaje con nódulos que no se usaron para armarlo, y dar un intervalo de confianza honesto con pocos casos |

## 5. Simulación de baja dosis (rehecha para el recuperatorio)

### De qué método parte

LIDC trae imágenes reconstruidas, no las proyecciones crudas del tomógrafo. Eso descarta los métodos que agregan ruido sobre los datos crudos, como **Zeng et al. (2015)** y **Yu et al. (2012)**. El método que aplica es el de **Naziroglu et al. (2017)**, pensado justamente para simular baja dosis a partir de imágenes ya reconstruidas. La versión de la primera entrega tenía su esqueleto (Radon, Beer-Lambert, Poisson, FBP) pero no lo citaba y le faltaban las piezas que convierten "agregar ruido" en "bajar la dosis". **Byl et al. (2023)** compararon bajar la corriente del tubo con bajar la cantidad de proyecciones a igual dosis y concluyen que conviene bajar la corriente (con menos proyecciones aparecen artefactos de submuestreo que no se corrigen bien): por eso simulamos menos fotones y no menos vistas.

### Qué se cambió

| Pieza | Antes | Ahora | Por qué |
|---|---|---|---|
| Anclaje a la dosis del estudio | I0 = 10⁵ fijo para todos | Fotones por rayo = Φ · corriente (mA) · (kVp/120)^2,5 · Δs · espesor / vistas · bowtie, con la corriente, el kVp y el espesor del DICOM del corte (`03b`). Por qué la corriente y no los mAs: ver más abajo | Con un I0 fijo, cada etiqueta era "una adquisición con I0 fotones", no una fracción de la dosis original. LIDC va de ~40 a ~600 mA: el mismo I0 era otra fracción en cada estudio |
| Constante del tubo Φ | — | Calibrada con la propia LIDC (`07a`): el ruido medido de cada estudio y su corriente dan un Φ por estudio; se usa la mediana de los de más corriente (≥ 301 mA), por fabricante. Φ absorbe el tiempo de rotación y el pitch, que no están de forma confiable en LIDC | Naziroglu la calibra con un cilindro de agua en el mismo equipo; acá no hay esos barridos. Si el modelo es correcto, Φ tiene que salir parecido en todos los estudios de un mismo fabricante: la dispersión mide el error del modelo |
| Compensación del ruido que la imagen ya trae | Se sumaba el ruido de la dosis objetivo | Se suma el de la dosis equivalente N·f/(1−f) (200 → 50 mAs = agregar el ruido de 66,7 mAs) | La varianza del ruido cuántico es ∝ 1/fotones; las varianzas se suman. Sin compensar, "×1" agregaba ruido |
| Ruido electrónico | No | Gaussiano en las cuentas, compensado: σ² agregado = σe² (1+f)/(1−f) | No escala con la dosis: pesa cuando quedan pocos fotones (rayos largos, dosis muy baja) |
| Qué se reconstruye | El sinograma completo (la imagen pasaba dos veces por Radon + FBP y perdía resolución: la condición "FBP sin ruido" ya cambiaba la imagen) | Solo el sinograma de ruido; la imagen de ruido se suma a la original | Con f = 1 la imagen sale idéntica, así que todo cambio en las medidas es efecto del ruido |
| Kernel del equipo | Rampa para todos | Ventana por familia de kernel del DICOM: duro (LUNG, BONE, B60+) → rampa; intermedio (B40-B59) → coseno; suave (STANDARD, B30f) → Hann | La ventana define la textura del ruido (su espectro), y la GLCM mide textura. Los kernels duros reales refuerzan frecuencias altas más que la rampa: es la aproximación más dura que tenemos. Para el intermedio se probó primero Shepp-Logan y el 07a mostró que estaba mal: el Φ de los estudios con kernel intermedio daba 8,6·10⁵, más del doble que el de los duros y suaves. Si la ventana fuera correcta, Φ no debería depender del kernel (es una propiedad del tubo). Con coseno, los tres dan lo mismo: duro 3,70·10⁵, intermedio 3,69·10⁵, suave 3,71·10⁵ |
| Ruido de la imagen original | MAD del residuo en el parénquima (~28 HU) | Ruido global de Christianson et al. (2015): moda del desvío local (6 mm) en tejido blando | En el parénquima la textura del pulmón se confunde con ruido. El mismo estimador se aplica a la imagen real y al ruido simulado |
| Cómo se reportan los resultados | Por etiqueta de dosis | Por fracción de la dosis del estudio **y** por ruido agregado en HU (medido exacto sobre la imagen de ruido) y corriente equivalente (la columna `mas_equivalente` conserva el nombre, pero está en mA) | La devolución tenía razón en que las etiquetas viejas subestimaban la reducción |

### Por qué se ancla con la corriente (mA) y no con los mAs

La devolución sugería usar los mAs del encabezado, como hace Naziroglu. Al leer los encabezados de los 217 cortes (`03b`) aparece que en LIDC esos campos no significan lo mismo en cada fabricante:

| Fabricante | Estudios | Exposure (mAs) | ExposureTime (ms) | XRayTubeCurrent (mA) |
|---|---|---|---|---|
| GE | 123 | 1 – 7200 (mediana 880) | 400 – 1160 | 40 – 570 (mediana 265) |
| Philips | 20 | 154 – 350 | no lo informa | 240 – 422 |
| Siemens | 34 | 25 – 324 (mAs efectivos, ya divididos por el pitch) | 500 – 600, en 26 de 34 | 50 – 561 |
| Toshiba | 1 | 130 | 500 | 260 |

Un mAs de 7200 no es físico para un corte, Philips no informa el tiempo y Siemens guarda mAs efectivos. Comparar esos números entre equipos mezclaría definiciones distintas. La corriente es la misma magnitud en todos, así que es el índice de dosis. Lo que la corriente no incluye (tiempo de rotación, pitch) queda dentro de Φ, que se calibra por fabricante. Para la fracción de dosis da lo mismo: si un estudio se adquirió con 240 mA, "×1/4" es adquirirlo con 60 mA y el resto igual.

### Lo que no se puede replicar exacto (y cómo se trata)

- **Calibración por equipo.** Sin cilindro de agua por equipo, Φ es uno por fabricante. Los equipos de un mismo fabricante difieren entre sí: queda como dispersión, que se reporta.
- **Bowtie y constante del tubo.** Ver sección 6.
- **Ruido electrónico.** Sin medirlo en cada equipo, se usa un valor de literatura (Elhamiasl y Nuyts 2020 reportan ~29 fotones equivalentes en un equipo moderno). El 07 se puede correr `--sin-electronico` para ver cuánto cambia; el 07a lo pone a prueba en los estudios de LIDC que ya son de baja dosis, y además estima su valor con los propios datos de LIDC (ver “Resultados con LIDC”).
- **Geometría.** El sinograma virtual es 2D, de haz paralelo, de un solo corte. El equipo real es de haz en abanico, helicoidal, con reconstrucción de cortes que mezclan filas. La cantidad de fotones se escala para que el ruido por píxel sea el mismo, pero la forma espacial del ruido es una aproximación.
- **Lo que el ruido original tiene de no cuántico** (electrónico, artefactos) se trata como cuántico al anclar. Por eso conviene simular fracciones moderadas (½, ¼) con más confianza que 1/20.

### Cómo se valida

1. `tests/test_baja_dosis.py`, en un fantoma donde la verdad se conoce: "×1" devuelve la imagen idéntica; original + ruido agregado tiene el mismo desvío que adquirir directamente con f·N fotones (dentro del 8 %), y sin compensar no; el ruido total sigue σ0/√f cuando no hay ruido electrónico.
2. `07a`: la dispersión de Φ entre estudios del mismo fabricante, y la predicción del ruido de los estudios de LIDC que ya son de baja dosis (≤ 160 mA), con Φ calibrado solo con los de alta corriente (≥ 301 mA).
3. Con el anclaje por ruido, que no agrega electrónico, el ruido total tiene que seguir σ0/√f exacto en los nódulos reales.

### Referencias

- Naziroglu RE, van Ravesteijn VF, van Vliet LJ, Streekstra GJ, Vos FM. Simulation of scanner- and patient-specific low-dose CT imaging from existing CT images. *Phys Med.* 2017;36:12-23. doi:10.1016/j.ejmp.2017.02.009
- Zeng D, Huang J, Bian Z, et al. A simple low-dose X-ray CT simulation from high-dose scan. *IEEE Trans Nucl Sci.* 2015;62(5):2226-2233. doi:10.1109/TNS.2015.2467219
- Yu L, Shiung M, Jondal D, McCollough CH. Development and validation of a practical lower-dose-simulation tool for optimizing computed tomography scan protocols. *J Comput Assist Tomogr.* 2012;36(4):477-487.
- Byl A, Sawall S, Rafecas M, Hoeschen C, Kachelrieß M. Low-dose CT: reducing tube current, number of projections, or both? IEEE NSS/MIC 2023.
- Elhamiasl M, Nuyts J. Low-dose x-ray CT simulation from an available higher-dose scan. *Phys Med Biol.* 2020;65:135010.
- Christianson O, Winslow J, Frush DP, Samei E. Automated technique to measure noise in clinical CT examinations. *AJR Am J Roentgenol.* 2015;205(1):W93-W99.

## 6. Parámetros que no se pueden medir en LIDC: qué elegimos y cuánta incertidumbre meten

Naziroglu et al. (2017) obtienen los parámetros del equipo escaneando un cilindro de agua en ese mismo tomógrafo. LIDC no trae esos barridos, y el perfil del bowtie, el rendimiento del tubo y el ruido de lectura son información que los fabricantes no publican. Whiting et al. (2014) midieron el bowtie de cinco equipos de tres fabricantes, encontraron perfiles bastante distintos, y concluyen que un bowtie genérico no alcanza para investigación sobre el sistema.

Por eso el criterio es: **para cada parámetro que no podemos medir, elegimos el valor mejor fundado que haya en la literatura, definimos un rango plausible con su fuente, y propagamos ese rango hasta el resultado**. El resultado deja de ser "esto es ×1/4" y pasa a ser "esto es ×1/4, y con lo que no sabemos de los equipos corresponde a entre 1/a y 1/b de la dosis (intervalo del 95 %)".

### El bowtie: cómo se mide y qué hacemos sin el equipo

Todos los métodos para medir un bowtie necesitan el tomógrafo físico: medir con el tubo quieto en modo servicio, un dosímetro de tiempo real con el tubo girando (COBRA: Boone 2010; McKenney et al. 2011; Whiting et al. 2014), película radiocrómica (Whiting et al. 2015) o un detector lineal dentro del gantry (Yang et al. 2017, 2019). Para LIDC no es posible.

El único trabajo que encontramos que estimó bowties **sin datos del fabricante ni mediciones propias** es el de Kramer et al. (2017). Proponen un filtro de PMMA cuyo espesor sigue una función sigmoide del ángulo del abanico (diseñado para compensar un fantoma de cuerpo de 32 cm hasta ~12° y que después se aplana en ~8 cm), y lo adaptan a cada tomógrafo con un único factor (SFF). El SFF se ajusta hasta que el CTDI calculado por Monte Carlo coincide, dentro del 5 %, con el CTDI publicado para ese equipo. Ajustaron 11 tomógrafos de la época de LIDC (GE LightSpeed, Philips Mx8000/Brilliance, Siemens Sensation, Toshiba Aquilion), con SFF de cuerpo entre 0,66 y 1,76.

Lo que hacemos:
- Usamos el modelo de Kramer. Si el modelo del tomógrafo del estudio (tag DICOM ManufacturerModelName) está en su tabla, usamos su SFF; si no, la mediana de los 11 (1,01).
- La incertidumbre del bowtie se propaga con el rango completo de los 11 equipos (0,66-1,76), más la del μ efectivo del PMMA (no modelamos el endurecimiento del haz) y la de la distancia fuente-isocentro.

### Todos los parámetros, con su rango y su fuente

| Parámetro | Valor por defecto | Rango plausible | De dónde sale el rango |
|---|---|---|---|
| Factor del bowtie (SFF) | el del modelo en Kramer, o 1,01 | 0,66 – 1,76 | Mínimo y máximo de los 11 equipos de Kramer et al. 2017 |
| μ efectivo del PMMA | 0,0229 /mm | 0,021 – 0,025 /mm | Tablas NIST, energía efectiva entre 80 y 50 keV (endurecimiento) |
| Distancia fuente-isocentro | 570 mm | 540 – 600 mm | Rango de los equipos clínicos |
| Ruido electrónico por lectura | 30 | 5 – 150 (escala log) | ~29 en un equipo moderno (Elhamiasl y Nuyts 2020); rango amplio porque los equipos de LIDC son de 1999-2004 |
| Vistas por rotación | 1000 | 900 – 2400 | Equipos clínicos |
| Ancho de canal en el isocentro | 0,6 mm | 0,5 – 0,75 mm | Detectores clínicos |
| Ancho de fila | 0,625 mm | 0,5 – 1,25 mm | Equipos de 4 a 64 filas |
| Ventana del kernel | según familia | la familia vecina, más dura o más suave | El kernel real no es público |
| Vistas de la simulación | 360 | 180 – 720 | Discretización numérica |
| Constante del tubo Φ (anclaje por corriente) | calibrada por fabricante (07a) | dispersión real k = Φ_fabricante/Φ_estudio del 07a: mediana 0,73, P25 0,54, P75 1,00 | La propia LIDC |
| Error al medir el ruido original (anclajes ruido e híbrido) | 0 | Distribución medida en cada imagen (07a): mediana 4,5 %, P90 18,8 % | Mitades del tejido blando; provisorio: Christianson et al. 2015 (3,4-4,7 %) |
| Parte del ruido medido que no es cuántica | 0 | IC 95 % del ajuste σ0² = cuántico/mA + b (07a): 22 – 56 % (mediana 38 %), con tope en 80 % | La propia LIDC |
| Exponente del kVp | 2,5 | 2 – 3 | Solo pesa en estudios que no son de 120 kVp |

### Cómo se propaga (`comparaciones/incertidumbre_dosis.py`)

La pregunta es: si los parámetros verdaderos del equipo fueran otros, ¿a qué fracción de dosis corresponde de verdad lo que el pipeline etiqueta "×1/4"?

1. **Mundo verdadero.** Se simula el equipo con los parámetros sorteados. Los fotones se ajustan para que el ruido del tejido blando dé el ruido medido en la imagen (es un dato). Se obtiene la varianza cuántica (A) y electrónica (E) que la imagen ya trae cerca del nódulo.
2. **Lo que hace el pipeline.** Con los valores por defecto, agrega cerca del nódulo una varianza σ²(f).
3. **Fracción real.** En el mundo verdadero, esa varianza equivale a bajar la dosis a la fracción f_real que cumple A(1/f_real − 1) + E(1/f_real² − 1) = σ²(f).

Se hacen dos análisis:
- **Uno por vez (tornado).** Cada parámetro en su mínimo y su máximo, el resto por defecto. Dice qué parámetro mueve más el resultado.
- **Todos a la vez (Monte Carlo, 200 sorteos).** Se sortean todos dentro de sus rangos. Da la mediana y el intervalo del 95 % de la dosis real de cada etiqueta, y qué porcentaje de la variabilidad explica cada parámetro (correlación de rangos de Spearman al cuadrado, normalizada). Con 200 sorteos, los porcentajes menores a ~2 % no se distinguen de cero.

Se usan siempre las mismas semillas de ruido, así que las diferencias entre sorteos son por los parámetros y no por el azar.

### Resultados en el fantoma (históricos: rangos PROVISORIOS, antes de correr el 07a; los que valen son los de “Resultados con LIDC”)

Dosis real de cada etiqueta, mediana e IC 95 % (`incertidumbre_dosis.py --anclaje todos`):

| Etiqueta | mAs × Φ (paper) | Ruido de la imagen | Híbrido |
|---|---|---|---|
| ×1/2 | 1/2,3 (1/1,5 – 1/4,8) | 1/2,2 (1/1,5 – 1/3,4) | 1/2,3 (1/1,6 – 1/3,6) |
| ×1/4 | 1/4,9 (1/2,5 – 1/11,7) | 1/4,5 (1/2,4 – 1/8,1) | 1/4,9 (1/2,6 – 1/9,0) |
| ×1/10 | 1/12,5 (1/5,0 – 1/33) | 1/10,5 (1/4,4 – 1/21) | 1/12,3 (1/5,1 – 1/27) |
| ×1/20 | 1/25 (1/8,7 – 1/77) | 1/18 (1/6,7 – 1/41) | 1/24 (1/8,6 – 1/57) |

- **De dónde viene la incertidumbre de ×1/4.** Con mAs: Φ 53 %, bowtie 29 %, ruido electrónico 6 %. Con ruido o híbrido: bowtie ~70 %, ruido electrónico ~7 %, ruido no cuántico ~6 %, kernel ~6 %.
- **El intervalo es más angosto anclando por el ruido de la imagen** (de ×1/4: cociente máx/mín 3,3-3,5 contra 4,8 con mAs). Pero eso depende de que la dispersión real de Φ sea tan grande como el rango provisorio: hay que repetirlo con el 07a corrido.
- **La mediana queda 15-25 % por encima de la etiqueta en los tres modos.** Viene del ruido no cuántico: si una parte del ruido medido no es cuántica, el pipeline cree que la imagen tenía menos fotones y agrega de más. Con el 07a corrido, ese sesgo se mide y se puede corregir. El modo ruido queda más cerca de la etiqueta en 1/20 porque no agrega ruido electrónico, y las dos omisiones se compensan por casualidad.

### Tres formas de anclar los fotones, y el error de cada una medido con LIDC

No elegimos un anclaje a priori: se corren los tres y se reporta el intervalo de cada uno.

| Anclaje (`07 --anclaje`) | De dónde salen los fotones de la imagen original | Su error principal | Cómo se mide ese error |
|---|---|---|---|
| `mas` (lo que hace el paper, con la corriente en lugar de los mAs) | Φ del fabricante × corriente del DICOM | Un Φ por fabricante no es el Φ de cada equipo | Dispersión real de Φ_estudio/Φ_fabricante (`07a_estudios.csv`) |
| `ruido` | El ruido medido en el tejido blando de la imagen | Error de medir el ruido; no agrega ruido electrónico | Error de medición por imagen (mitad izquierda vs derecha del tejido) |
| `hibrido` | El ruido de la imagen, más ruido electrónico | Error de medir el ruido | Igual que `ruido` |

Además, los tres comparten un error sistemático: si parte del ruido medido no es cuántico (estructura del tejido, artefactos), se sobreestima el ruido original. El 07a lo mide con LIDC: con y = (σ0/σ_ref)², que es el ruido medido relativo al que daría la simulación para ese mismo paciente (así se cancela el tamaño del paciente), y x = 1/(corriente × geometría), la física dice y = pendiente·x + b. La pendiente es la parte cuántica y b la que no baja con la dosis. Se ajusta con bootstrap por estudio, y su intervalo del 95 % entra al análisis de incertidumbre. En el fantoma con corrientes inventadas, donde el ruido no depende de la corriente, este ajuste lo detecta y da ~100 % no cuántico, que es lo correcto.

El 07a también compara, dentro de cada modelo de tomógrafo, la dispersión de Φ con la que daría solo el error de medir el ruido (fotones ∝ 1/σ², así que es ~2 × el error de σ). Si son parecidas, los equipos de ese modelo son iguales y promediar (modo `mas`) no pierde nada. Si la dispersión es mayor, los equipos difieren y conviene anclar estudio por estudio.

`comparaciones/incertidumbre_dosis.py --anclaje todos` toma estos tres errores del 07a (cuando está corrido) y da, para cada anclaje, el intervalo del 95 % de la dosis real de cada etiqueta, en un mismo gráfico de barras (`figuras/comparaciones/incertidumbre_anclajes.png`). `comparaciones/comparar_anclajes.py` muestra si las conclusiones del 07 (medidas robustas, AUC) cambian entre anclajes.

### Resultados con LIDC (corrida del 25/11/2026)

#### 07a: calibración y errores, medidos en los 178 estudios con corriente y ruido medible

- **Φ.** Mediana de los estudios de alta corriente (≥ 301 mA): 2,83·10⁵ fotones/(mA·mm²) en total, 2,4·10⁵ en GE y 6,05·10⁵ en Philips. Entre estudios del mismo fabricante, P75/P25 ≈ 2.
- **La ventana del kernel está bien elegida.** El Φ de cada estudio no depende de la familia de kernel (duro 3,70·10⁵, intermedio 3,69·10⁵, suave 3,71·10⁵). Si una ventana estuviera mal, esa familia tendría otro Φ, que es lo que pasaba con Shepp-Logan.
- **Error al medir el ruido de una imagen:** mediana 4,5 %, P90 18,8 % (Christianson et al. 2015 reportan 3,4-4,7 %).
- **Los equipos de un mismo modelo difieren de verdad.** Dentro de cada modelo de tomógrafo, la dispersión de Φ es entre 1,7 y 11 veces (mediana 4,4) la que daría solamente el error de medir el ruido. Un Φ por fabricante (lo que hace el anclaje `mas`) no alcanza a describir cada estudio; anclar con el ruido de cada imagen evita ese error.
- **Parte del ruido que no baja con la dosis:** mediana 38 % (IC 95 % 22-56 %). Lo tomamos como **cota superior**: las diferencias de Φ entre equipos del punto anterior funcionan como error en la variable x del ajuste, y un error en x aplana la pendiente y agranda la ordenada b (dilución por regresión). El valor real es probablemente menor.
- **Validación con los estudios que ya son de baja corriente** (93 cortes de ≤ 160 mA, con Φ calibrado solo con los de ≥ 301 mA). Error relativo del ruido predicho, (predicho − medido)/medido: **+24 % sin ruido electrónico** (P25 +11 %, P75 +42 %) y **+52 % con σe² = 30** (P25 +31 %, P75 +73 %). El modelo predice más ruido del que tienen las imágenes reales, y agregarle el ruido electrónico de literatura duplica el error. Estos estudios están a ~la mitad de la corriente de calibración, donde el ruido electrónico todavía debería pesar poco: que ya ahí empeore la predicción sugiere que σe² = 30 es alto para estos equipos. Pero esta validación sola no alcanza para afirmarlo: sin electrónico ya se pasa un 24 %, en parte por el ruido no cuántico de los estudios de calibración (que hace que Φ salga más bajo), y mientras ese error no se separe del electrónico no se puede decir cuánto es cada uno. Para eso está la estimación conjunta del punto siguiente.

- **Estudios de calibración y de validación** (tomografías reales de LIDC, separadas solo por la corriente con que se adquirieron): calibración, 60 estudios / 77 cortes (GE 40, Philips 13, Siemens 7), ≥ 301 mA (medianas 360, 412 y 440 mA); validación, 80 estudios / 93 cortes (GE 57, Siemens 23), ≤ 160 mA (medianas 160 y 75 mA). No hay Philips en la validación: todos los de la muestra son de ≥ 240 mA.
- **Ruido electrónico estimado con LIDC.** Cada fuente de ruido cambia distinto con la corriente I: la cuántica como 1/I, la electrónica como 1/I² y la no cuántica no cambia. Con los 177 estudios (de 40 a 570 mA) se ajustan las tres a la vez, con Φ por fabricante, en escala logarítmica y con IC 95 % por bootstrap de estudios (500 remuestreos). Resultado: **σe² = 6 por lectura (IC 95 % 0 – 34)**; ruido no cuántico 9 HU de desvío (6,7 – 11,3). El valor de literatura (30) queda dentro del intervalo, cerca del borde: los datos no permiten descartarlo, pero el más probable es bastante menor. El intervalo es ancho porque el ajuste tiene un error de ±50 % por estudio (las diferencias de Φ entre equipos) y porque el electrónico pesa poco en casi todos los estudios (mediana 2,6 % del ruido; máximo 29 %, en el de menor corriente).
  - *Validación del método antes de usarlo:* con datos sintéticos de verdad conocida y la misma dispersión de Φ entre estudios que mide el 07a, el ajuste recuperó σe² = 0 → 0,0 (0 – 3), 5 → 6,8 (0,2 – 24,5) y 30 → 40 (15 – 175): el intervalo contiene el valor verdadero en los tres casos. Una primera versión con residuos relativos subestimaba σe² cuando Φ varía entre equipos; la versión en escala logarítmica no tiene ese sesgo.
  - *Limitación:* se ancla con mA y no con mAs. Si los protocolos de baja corriente compensaran sistemáticamente con rotaciones más lentas, parte del término 1/I² quedaría disimulado. No se puede descartar con LIDC.

#### 07: los tres anclajes en los 217 nódulos (112 en la población de entrenamiento)

Ruido total en el pulmón (mediana, HU). La fila "original" es el ruido que el modelo le asigna a la imagen original, no una medición: por eso cambia con el anclaje.

| | `mas` | `ruido` | `hibrido` | σ0/√f esperado sin electrónico (`ruido`) |
|---|---|---|---|---|
| original | 19,8 | 15,7 | 16,9 | 15,7 |
| ×1/2 | 30,7 | 22,1 | 25,5 | 22,2 |
| ×1/4 | 52,2 | 31,4 | 42,1 | 31,3 |
| ×1/10 | 113,3 | 49,9 | 98,4 | 49,5 |
| ×1/20 | 245,1 | 71,4 | 184,4 | 70,1 |

El anclaje por ruido sigue σ0/√f, que es lo que tiene que pasar sin ruido electrónico: es la verificación de la compensación en datos reales. El pequeño exceso en ×1/20 viene del piso de una cuenta por rayo.

AUC transferido (puntaje "interior", armado con la original):

| | `mas` | `ruido` | `hibrido` | diámetro solo (`ruido` / `hibrido`) |
|---|---|---|---|---|
| original | 0,938 | 0,938 | 0,938 | 0,846 |
| ×1/2 | 0,913 | 0,934 | 0,933 | 0,842 / 0,844 |
| ×1/4 | 0,879 | 0,908 | 0,879 | 0,839 / 0,823 |
| ×1/10 | 0,750 | 0,862 | 0,758 | 0,840 / 0,793 |
| ×1/20 | 0,704 | 0,819 | 0,710 | 0,823 / 0,764 |

#### De dónde viene la diferencia entre anclajes (`comparar_anclajes.py`)

- **`mas` contra `hibrido`** (los dos con ruido electrónico; cambia solo cómo se estiman los fotones): el cociente del ruido agregado es ~1,3 en todas las dosis y el AUC difiere como máximo 0,026. Elegir entre Φ y el ruido medido casi no cambia las conclusiones.
- **`mas` contra `ruido`** (cambia además si hay ruido electrónico): el cociente crece con la reducción (mediana 1,5 en ×1/2, 1,8 en ×1/4, 2,3 en ×1/10, 3,0 en ×1/20) y el AUC llega a diferir 0,165.

La mayor fuente de incertidumbre de la simulación no es Φ ni el bowtie: es cuánto ruido electrónico tenían estos equipos, y pesa en ×1/10 y ×1/20. Coincide con lo que señalaba la devolución. Ese valor no está en el DICOM; lo acotamos con dos escenarios.

#### Resultado principal: anclaje `hibrido` con el σe² estimado en LIDC (6), con 0 y 34 como extremos

- Los fotones se anclan con el ruido medido en cada estudio, no con un Φ por fabricante: el 07a mostró que un Φ por fabricante no describe a cada equipo (dispersión 4,4 veces el error de medición), y la propagación de errores confirma que el anclaje por corriente × Φ exagera la reducción al doble (mediana de "×1/4" = 1/8,2) con un intervalo casi el doble de ancho que anclar por el ruido.
- Incluye el ruido electrónico, como el método de Naziroglu y como pedía la devolución, con el valor que dan nuestros datos (σe² = 6) en lugar del de otro equipo.
- Los extremos del rango compatible con los datos se reportan al lado: σe² = 0 (es el anclaje `ruido`, que además pasa la verificación exacta σ0/√f) y σe² = 33,7 (el límite superior del IC; prácticamente igual al híbrido con 30 de la literatura).
- `mas` queda como referencia del método del paper sin calibración por equipo.

AUC transferido del puntaje "interior" / diámetro solo, en los tres escenarios de ruido electrónico:

| σe² | ×1/2 | ×1/4 | ×1/10 | ×1/20 |
|---|---|---|---|---|
| 0 (`ruido`) | 0,934 / 0,842 | 0,908 / 0,839 | 0,862 / 0,840 | 0,819 / 0,823 |
| **6 (principal)** | **0,936 / 0,842** | **0,915 / 0,841** | **0,824 / 0,822** | **0,760 / 0,822** |
| 33,7 | 0,932 / 0,841 | 0,874 / 0,823 | 0,760 / 0,798 | 0,701 / 0,759 |

Ruido agregado en el pulmón con el principal (mediana): 16, 30, 58 y 100 HU a ×1/2, ×1/4, ×1/10 y ×1/20. Dice medio 0,89 / 0,89 / 0,87 / 0,85; aceptables 96 / 94 / 84 / 76 %.

**Conclusión, válida en todo el rango compatible con LIDC:** la ventaja de las medidas de textura sobre el diámetro se mantiene hasta ×1/4 y desaparece hacia ×1/10 (con el valor más probable quedan empatados, 0,824 contra 0,822); a ×1/20 el diámetro solo es igual o mejor.

*Nota sobre el CCC del tamaño:* con el principal, el CCC del diámetro da 0,65 a ×1/10 y 0,83 a ×1/20. No es que el tamaño mejore con más ruido: el CCC del tamaño lo dominan unas pocas segmentaciones que se escapan al tejido vecino (dos o tres nódulos cambian el valor). El efecto del ruido sobre la segmentación se reporta con el Dice y la proporción de aceptables, que bajan de forma pareja.

#### Qué conclusiones dependen del anclaje

- **No dependen:** a ×1/2 no se pierde nada (AUC 0,91-0,93) y a ×1/4 la pérdida es chica (0,88-0,91). El tamaño y las HU medias son robustos; la textura de los anillos es lo más frágil; el ruido genera calcio espurio.
- **Dependen del ruido electrónico:** el resultado exacto a ×1/10. Sin electrónico el puntaje de textura apenas le gana al diámetro (0,862 contra 0,840), con el valor más probable empatan (0,824 contra 0,822) y con el máximo pierde (0,760 contra 0,798). Se reporta como rango.
- Con 112 nódulos, los IC 95 % por bootstrap del AUC van de ±0,04 a ±0,09, así que diferencias de 0,02-0,03 entre condiciones no son significativas.

#### 08: puntaje robusto con dos criterios de dosis

**Resultado principal (híbrido, σe² = 6).**

- *Criterio ×1/4:* pasan 20 medidas del interior, prácticamente todas, y el puntaje robusto da **idéntico** al del 06 (AUC 0,938 / 0,936 / 0,915 / 0,824 / 0,760 en la original, ×1/2, ×1/4, ×1/10 y ×1/20). A un cuarto de la dosis el modelo ya es robusto y no hace falta recortarlo.
- *Criterio ×1/10:* pasan solo 7 (nitidez del borde, HU media, desvío, calcio, energía, cluster shade y prominence) y el puntaje cae a 0,764 en la original y 0,751 a ×1/10, por debajo del diámetro (0,846 y 0,822) en todas las dosis. La causa: a ×1/10 el CCC del área (0,23) y del diámetro (0,65) sale bajo por unas pocas segmentaciones que se escapan al tejido vecino, y el criterio descarta justo el tamaño, la medida más informativa. En el subgrupo sin calcio, donde el tamaño sí pasa el filtro (CCC 1,00), el puntaje robusto se mantiene entre 0,90 y 0,92 hasta ×1/20, mientras el del 06 cae a 0,77.
- *Conclusión:* el criterio por CCC funciona mientras no deje afuera al tamaño. A ×1/10, el CCC del tamaño no es confiable porque lo dominan pocas fugas de la segmentación; el efecto del ruido sobre el tamaño se describe mejor con el Dice y la proporción de aceptables.
- *Calcio robusto (principal):* nódulos sin calcio que parecen calcificados con el umbral simple: 6 % a ×1/4, 20 % a ×1/10 y 45 % a ×1/20; con la medida robusta: 0 %, 7 % y 21 %. Detecta el 90–95 % de los calcificados reales, contra el 100 % del umbral simple.

Los resultados de los anclajes `ruido` y `hibrido` con σe² = 30, a continuación, se mantienen como comparación.

El puntaje robusto elige medidas con CCC ≥ 0,9 a una "dosis de uso" (el CCC no usa las etiquetas, así que este filtro no espía la clasificación) y AUC ≥ 0,7 **dentro de cada pliegue** de armado (`cv_robusto`), como pedía la devolución. Se corrió con dos dosis de uso (`--dosis-uso 4` y `--dosis-uso 10`) para ver cuánto depende de esa elección.

- **Anclaje `ruido`, criterio ×1/4:** pasan 21 medidas, casi todas. El puntaje robusto coincide con el del 06: a esa dosis el modelo ya es robusto.
- **Anclaje `ruido`, criterio ×1/10:** pasan 12. AUC 0,909 [0,85-0,95] en la original, 0,873 [0,81-0,93] a ×1/10 y 0,876 [0,81-0,93] a ×1/20, contra 0,938 [0,89-0,97], 0,862 [0,78-0,93] y 0,819 [0,72-0,90] del puntaje del 06. Rinde un poco menos con la imagen original y casi no cae con el ruido. Los intervalos se superponen en todas las dosis: con esta muestra no se puede afirmar que uno sea mejor.
- **Anclaje `hibrido`, criterio ×1/10:** pasan solo HU media, desvío y calcio, y el puntaje cae a 0,73, peor que el diámetro. Con ruido electrónico alto, a ×1/10 ya falla la segmentación (CCC del área 0,25, del diámetro 0,67), así que tampoco el tamaño es reproducible. Ahí el límite lo pone la segmentación, no la elección de medidas.

**Calcio robusto (anclaje `ruido`).** Nódulos sin calcio que parecen calcificados: con el umbral simple, 4 % a ×1/4, 13 % a ×1/10 y 23 % a ×1/20; con la medida robusta, 0 % hasta ×1/10 y 4 % a ×1/20. El costo: en la imagen original detecta el 90 % de los calcificados en lugar del 100 %. Con `hibrido` el calcio falso es mayor (robusto: 14 % a ×1/10, 38 % a ×1/20).

**Subgrupo sin calcio.** No se interpreta la comparación diámetro contra textura en ese subgrupo: excluir los calcificados saca 34 benignos y 7 sospechosos, los benignos que quedan son más chicos, y el diámetro mejora en buena parte por ese cambio de muestra (observación de la devolución). Además los intervalos son muy anchos.

#### 13: la selección del puntaje robusto completamente dentro de cada pliegue

La devolución señaló que el puntaje robusto elegía sus medidas con todo el conjunto de desarrollo. El filtro de AUC ya estaba dentro de cada pliegue; el 13 hace también el filtro de CCC dentro de cada pliegue (resultado principal):

- *Criterio ×1/4:* el AUC da exactamente igual que en el 08 en todas las dosis (diferencia 0,000; sin calcio, entre −0,020 y +0,017).
- *Criterio ×1/10:* da algo más alto que en el 08 (+0,045 en la original, +0,016 a ×1/10), porque dentro de algunos pliegues el tamaño sí pasa el filtro.

Filtrar por CCC con todos los nódulos no le pasa información de las etiquetas al puntaje: el resultado del 08 no es optimista.

#### 11: parámetros de adquisición de la muestra, y diferencia entre grupos

178 estudios: 120 kVp (algunos de 140), corriente mediana 240 mA (40 – 570), espesor 1,5 mm (0,6 – 2,5). GE 123, Siemens 34, Philips 20, Toshiba 1; 23 modelos de tomógrafo. Kernels: suaves 111, duros 55, intermedios 12 (el más frecuente, STANDARD de GE, en 82). El CTDIvol no figura en ningún encabezado. LIDC mezcla tomografías diagnósticas y de tamizaje: no es una base de "dosis diagnóstica".

**Los dos grupos no se adquirieron igual** (por nódulo):

| | Benignos (109) | Sospechosos (108) |
|---|---|---|
| Corriente mediana | 195 mA | 265 mA |
| Kernel duro | 24 % (26) | 37 % (40) |
| Espesor mediano | 1,5 mm | 2,0 mm |

Un kernel duro deja más textura de ruido y una corriente mayor deja menos ruido: parte de las diferencias de textura entre grupos puede venir de la adquisición y no del nódulo. No se corrige en el análisis; se reporta como **posible factor de confusión** de las medidas de textura (limitación).

#### Control: criterio de aceptabilidad original (112) contra el justo (92)

Se decidió mantener el análisis principal con el criterio original (112 nódulos de desarrollo), porque los 21 nódulos que el criterio justo deja afuera quedan a una mediana de 0,008 de Dice (máximo 0,048) de la concordancia entre radiólogos: son segmentaciones al nivel de un radiólogo, no fallas. Como control, `comparaciones/sensibilidad_aceptables.py` repite los AUC principales (diámetro, puntaje del interior y puntaje robusto, en la original y en cada dosis del resultado principal) con los 92 nódulos del criterio justo. Resultados (AUC [IC 95 %], criterio justo con 92 / original con 112):

| | Diámetro solo | Puntaje del interior |
|---|---|---|
| original | 0,885 [0,81–0,94] / 0,846 [0,77–0,92] | 0,922 [0,87–0,97] / 0,938 [0,89–0,97] |
| ×1/2 | 0,881 / 0,842 | 0,925 / 0,936 |
| ×1/4 | 0,882 [0,81–0,94] / 0,841 [0,76–0,91] | 0,905 [0,84–0,96] / 0,915 [0,86–0,96] |
| ×1/10 | 0,851 [0,76–0,92] / 0,822 [0,73–0,90] | 0,818 [0,72–0,91] / 0,824 [0,74–0,90] |
| ×1/20 | 0,852 / 0,822 | 0,788 / 0,760 |

(El puntaje robusto con criterio ×1/4 da idéntico al del interior con las dos poblaciones.) La mayor diferencia entre criterios es 0,041, menor que el ancho de cualquiera de los intervalos (±0,05 a ±0,10).

- **No cambia:** el puntaje del interior es igual o mejor que el diámetro hasta ×1/4, y hacia ×1/10 la ventaja desaparece (con 92 el diámetro queda apenas por encima: 0,851 contra 0,818); a ×1/20 el diámetro es mejor.
- **Sí cambia, y se reporta:** con los 92 el diámetro solo sube (0,846 → 0,885), así que la ventaja de la textura sobre el diámetro en la imagen original se achica de 0,09 a 0,04, con intervalos que se superponen. La ventaja de la textura a dosis completa depende de qué nódulos se incluyen y no es robusta a esa elección.

#### Incertidumbre de la dosis real (`incertidumbre_dosis.py`)

Diseño de la corrida: anclaje `ruido` con 110 nódulos y 50 sorteos por nódulo; `mas` e `hibrido` con 30 nódulos cada uno, porque solo se usan para comparar. Los nódulos se eligen alternando fabricante × familia de kernel, para no tomar todos del mismo equipo. El criterio de convergencia (que la mediana, P25 y P75 entre nódulos se muevan menos de ±5 % al agregar nódulos) no se alcanza con los nódulos disponibles: con 65 nódulos se mueven ±14 %, y bajan como 1/√n, así que con 110 se espera ~±10 % y con los 217, ~±7 %. Se reporta esa precisión junto con los intervalos.

Dosis real de cada etiqueta (mediana; mitad central; IC 95 %):

| Etiqueta | `ruido` (110 nódulos) | `hibrido`, σe² = 30 (30 nódulos) | `mas` (30 nódulos) |
|---|---|---|---|
| ×1/2 | 1/2,3; 1/1,9 – 1/2,8; 1/1,2 – 1/4,4 | 1/2,8; 1/2,3 – 1/3,4; 1/1,5 – 1/6,7 | 1/3,5; 1/2,6 – 1/5,1; 1/1,5 – 1/11,1 |
| ×1/4 | 1/4,6; 1/3,4 – 1/6,0; 1/1,7 – 1/10,4 | 1/6,2; 1/4,7 – 1/8,1; 1/2,5 – 1/17,8 | 1/8,2; 1/5,6 – 1/12,7; 1/2,4 – 1/28,1 |
| ×1/10 | 1/9,9; 1/6,7 – 1/13,8; 1/2,6 – 1/25,9 | 1/16,2; 1/11,5 – 1/22,7; 1/5,5 – 1/50 | 1/21,8; 1/14,0 – 1/35,0; 1/5,1 – 1/79,5 |
| ×1/20 | 1/16,6; 1/10,4 – 1/24,3; 1/3,7 – 1/48 | 1/33,1; 1/22,3 – 1/47,6; 1/10,5 – 1/107 | 1/44,3; 1/27,8 – 1/72,9; 1/9,2 – 1/173 |

Ancho del IC 95 % de ×1/4 (máx/mín): `ruido` 6,3, `hibrido` 7,1, `mas` 11,5. Sesgo de la mediana en ×1/4: +15 %, +54 % y +105 %. Qué explica la incertidumbre de ×1/4: con `ruido`, ruido electrónico 36 %, error al medir el ruido 27 %, ruido no cuántico 22 %, bowtie 7 %; con `mas`, el error de Φ explica el 74 %.

Estos intervalos usan para σe² el rango 5 – 150 definido antes de estimarlo; el 07a lo acota a 0 – 34. Como la parte de 34 a 150 no es compatible con los datos, **los intervalos reportados son conservadores** (más anchos de lo necesario). No se repitió el análisis con el rango acotado. Tampoco se corrigió el anclaje por la fracción de ruido no cuántico: su estimación es una cota superior (inflada por la dispersión de Φ) y corregir con ella empeoraría el resultado; su efecto está incluido en estos intervalos y es probablemente la causa del +15 % de sesgo en ×1/4.


### Lo que no es un parámetro

Hay supuestos que no se pueden variar como un número: la geometría (sinograma 2D de haz paralelo en lugar de abanico helicoidal), la dispersión, la modulación de corriente dentro de un corte y el endurecimiento del haz en el paciente. Para estos no hay intervalo propio. Dos cosas los acotan:
1. Naziroglu et al. también simulan con FBP de haz paralelo, y sus imágenes reprodujeron el ruido de adquisiciones reales de baja dosis.
2. La validación del 07a contra los estudios de LIDC que ya son de baja dosis mide el error total del modelo, incluidos estos supuestos: lo que el 07a no logra predecir es la suma de todo lo que no modelamos.

### Referencias de esta sección

- Boone JM. Method for evaluating bow tie filter angle-dependent attenuation in CT: theory and simulation results. *Med Phys.* 2010;37(1):40-48. doi:10.1118/1.3264616
- McKenney SE, Nosratieh A, et al. Experimental validation of a method characterizing bow tie filters in CT scanners using a real-time dose probe. *Med Phys.* 2011;38(3):1406-1415.
- Whiting BR, Evans JD, Dohatcu AC, Williamson JF, Politte DG. Measurement of bow tie profiles in CT scanners using a real-time dosimeter. *Med Phys.* 2014;41(10):101915. doi:10.1118/1.4896196
- Whiting BR, Dohatcu A, Politte DG, Williamson JF, Evans JD. Technical Note: Measurement of bow tie profiles in CT scanners using radiochromic film. *Med Phys.* 2015;42(6):2908-2914.
- Yang K, Li X, George Xu X, Liu B. Direct and fast measurement of CT beam filter profiles with simultaneous geometrical calibration. *Med Phys.* 2017;44(1):57-70. doi:10.1002/mp.12024
- Yang K, Ruan C, Li X, Liu B. Data of CT bow tie filter profiles from three modern CT scanners. *Data Brief.* 2019;25:104261. doi:10.1016/j.dib.2019.104261
- Kramer R, Cassola VF, Andrade MEA, de Araújo MWC, Brenner DJ, Khoury HJ. Mathematical modelling of scanner-specific bowtie filters for Monte Carlo CT dosimetry. *Phys Med Biol.* 2017;62:781-809. doi:10.1088/1361-6560/aa5343
