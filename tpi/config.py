"""Parámetros del pipeline (valores iniciales del anteproyecto v2).

Todo lo que se ajusta con el 80 % de entrenamiento vive acá, así queda
documentado en un solo lugar qué se cambió y por qué.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "datos"
DIR_DICOM = DIR_DATOS / "LIDC-IDRI"      # acá se descargan los DICOM
DIR_CORTES = DIR_DATOS / "cortes"        # un .npz por nódulo (corte de trabajo)
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = RAIZ / "figuras"

SEMILLA = 42

# --- Selección de nódulos -------------------------------------------------
MAX_ESPESOR_MM = 2.5     # cortes finos: el corte de mayor área es más representativo
MIN_ANOTACIONES = 3      # ≥3 radiólogos -> consenso y concordancia con sentido
MIN_TEXTURA = 4          # mediana de textura ≥4 = sólido
MIN_DIAMETRO_MM = 6.0    # por debajo de 6 mm Lung-RADS no cambia conducta
MAX_MALIG_BENIGNO = 2    # mediana ≤2 -> benigno
MIN_MALIG_SOSPECHOSO = 4 # mediana ≥4 -> sospechoso (3 se excluye)
FRAC_ENTRENAMIENTO = 0.8

# --- Corte de trabajo -----------------------------------------------------
PIXEL_MM = 0.7           # tamaño de píxel común tras remuestrear
ROI_MM = 64.0            # recorte centrado en la semilla
CLEVEL_CONSENSO = 0.5    # píxel en consenso si lo marcó ≥50 % de los radiólogos

# --- Segmentación ---------------------------------------------------------
UMBRAL_HU = -400         # ajustar en entrenamiento entre -600 y -300
RADIO_APERTURA_PX = 1    # 1-2 px: corta vasos finos adheridos
RADIO_MAX_MM = 30.0      # freno a fugas hacia pared torácica / mediastino
RADIO_CIERRE_PULMON_MM = 30.0  # cierre del pulmón: incorpora nódulos pegados a la pared (se ajusta en 04)

# --- Características ------------------------------------------------------
HU_CALCIO = 200
# Un nódulo se considera "calcificado" si más del 2 % de su área supera
# HU_CALCIO. Se usa para el análisis "--sin-calcio" (05 y 06): el calcio es el
# signo de benignidad que los radiólogos y Lung-RADS ya usan, y la pregunta
# clínica abierta son los nódulos SIN calcio (indeterminados).
FRAC_CALCIO_MAX = 0.02
HU_AIRE = -600
# GLCM: la materia recomienda 8-16 niveles (con más, la matriz queda rala e
# inestable, y un nódulo de 6 mm tiene apenas ~60 píxeles). Usamos 16.
NIVELES_GLCM = 16
# El rango que se reparte en esos 16 niveles tiene que cubrir los valores que
# HAY en la región: con la ventana de pulmón entera [-1000, 400] cada nivel
# valdría 87 HU y todo el interior de un nódulo sólido caería en 2-3 niveles
# (textura "plana" para todos). Por eso un rango fijo por tipo de región,
# igual para todos los nódulos (condición de la materia: mismos parámetros).
VENTANA_TEXTURA_NODULO = (-500, 300)   # 50 HU por nivel; >300 (calcio) satura
VENTANA_TEXTURA_ENTORNO = (-1000, 0)   # 62,5 HU por nivel; parénquima y vasos
DISTANCIAS_GLCM = (1,)                 # textura fina -> distancia 1
ANGULOS_GLCM = (0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490614922)

# --- Entorno --------------------------------------------------------------
HU_PULMON = -400
ANILLO_INTERNO_MM = 3.0
ANILLO_EXTERNO_MM = 6.0
HU_DENSO_ENTORNO = -500

# --- Reconstrucción -------------------------------------------------------
MU_AGUA = 0.02           # mm^-1 (aprox. a ~70 keV)
N_ANGULOS_COMPLETO = 180
N_ANGULOS_REDUCIDO = 60
I0_REFERENCIA = 1e5      # solo para comparaciones/reconstruccion.py (comparar MÉTODOS)
I0_BAJA_DOSIS = (2.5e4, 1e4)

# --- Simulación de baja dosis (tpi/baja_dosis.py, scripts 07a y 07) -------
# Sigue a Naziroglu et al. (2017). Ver DECISIONES.md, "Simulación de baja
# dosis". Todo lo que no está en el DICOM de LIDC sale de acá, y lo que más
# pesa (FOTONES_POR_MAS_MM2) se calibra con la propia base en el 07a.
N_VISTAS_DOSIS = 360          # vistas del sinograma virtual (en 180°, haz paralelo)
# Fotones que llegan por mAs y por mm² en el isocentro (integrados en la
# rotación, centro del bowtie, 120 kVp). Valor inicial del orden que da el
# kerma en aire típico (~0,2 mGy/mAs a 120 kV) con ~3e7 fotones/(mm²·mGy) a
# ~60 keV, menos filtrado y eficiencia del detector. El 07a lo reemplaza por
# el valor calibrado con el ruido de los estudios de LIDC.
FOTONES_POR_MAS_MM2 = 2e6
EXPONENTE_KVP = 2.5           # rendimiento del tubo ∝ kVp^~2-2,5
# Ruido electrónico (de lectura) en "fotones equivalentes" por lectura del
# detector. Elhamiasl y Nuyts (2020) reportan ~29 para un equipo moderno; los
# equipos de LIDC (1999-2004) probablemente tenían más. Se reporta con y sin.
# El 07a lo estima con los propios datos de LIDC (07a_electronico.csv). Para
# correr el 07 con otro valor: --sigma-e2 (lo pasa por la variable de entorno
# TPI_SIGMA_E2, que heredan los procesos en paralelo).
import os as _os
SIGMA_E2_LECTURA = float(_os.environ.get("TPI_SIGMA_E2", 30.0))
VISTAS_POR_ROTACION = 1000    # típico de los equipos de la época (no está en el DICOM)
ANCHO_DETECTOR_ISO_MM = 0.6   # ancho de un canal proyectado al isocentro
ANCHO_FILA_MM = 0.625         # ancho de una fila (si el DICOM no trae colimación)
# Bowtie: transmisión relativa al rayo central. El perfil real de cada equipo
# es información del fabricante y en LIDC no se puede medir (DECISIONES §6).
# Modelo por defecto, "kramer": Kramer et al. (2017, Phys Med Biol 62:781) —
# filtro de PMMA con espesor t(θ) sigmoide (Boltzmann), diseñado para un
# fantoma de cuerpo de 32 cm, escalado por un "scanner filter factor" (SFF)
# que ellos ajustaron para 11 tomógrafos con los CTDI publicados (sin datos del
# fabricante). SFF por modelo: tpi/encabezados.py (SFF_BOWTIE_KRAMER).
BOWTIE_MODELO = "kramer"
BOWTIE_SFF = 1.01             # mediana de los 11 equipos (filtro de cuerpo), si el modelo no está en la tabla
BOWTIE_SFF_RANGO = (0.66, 1.76)   # mínimo y máximo de los 11 equipos
BOWTIE_DSI_MM = 570.0         # distancia fuente-isocentro (~540-600 mm según fabricante)
MU_PMMA_MM = 0.0229           # μ del PMMA a ~60 keV (NIST: μ/ρ ≈ 0,192 cm²/g, ρ = 1,19 g/cm³)
# Modelos alternativos (para sensibilidad): "cilindro" (compensa una fracción
# del agua que le falta a un rayo lateral), "gauss" (campana) y "plano".
BOWTIE_DIAMETRO_MM = 320.0
BOWTIE_COMPENSACION = 0.5
BOWTIE_MIN = 0.05
BOWTIE_ANCHO_MM = 160.0
CUENTAS_MIN = 1.0             # piso de cuentas antes del logaritmo (fotones)
# Ruido de la imagen original: "ruido global" de Christianson et al. (2015):
# moda del desvío local en tejido blando (sin estructura, a diferencia del
# parénquima). Ventana local ~6 mm.
HU_TEJIDO_BLANDO = (0, 100)
LADO_RUIDO_LOCAL_MM = 6.0
# Fracciones de la dosis DEL ESTUDIO que se simulan en el 07.
FRACCIONES_DOSIS = (0.5, 0.25, 0.1, 0.05)

# --- Compresión -----------------------------------------------------------
VENTANA_8BIT = (-1000, 400)
CALIDADES_JPEG = (95, 75, 50, 25)
