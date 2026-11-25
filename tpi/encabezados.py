"""Parámetros de adquisición del encabezado DICOM.

LIDC-IDRI no es una base "de dosis diagnóstica": mezcla tomografías
diagnósticas y de tamizaje de varios equipos (GE, Siemens, Philips, Toshiba),
con corrientes de ~40 a ~600 mA, 120-140 kVp y kernels de reconstrucción que
van de suaves (STANDARD, B30f) a muy duros (LUNG, BONE, B70f). Para simular
una FRACCIÓN de la dosis de cada estudio hace falta saber de qué dosis parte
cada uno (mAs) y con qué kernel se reconstruyó (textura del ruido): está en
el encabezado de cada corte.
"""
import re

import numpy as np

# Tag DICOM -> nombre de columna. Algunos equipos no completan todos.
CAMPOS = {
    "Manufacturer": "fabricante",
    "ManufacturerModelName": "modelo",
    "KVP": "kvp",
    "XRayTubeCurrent": "corriente_ma",       # del corte (con modulación, cambia por corte)
    "ExposureTime": "tiempo_exposicion_ms",
    "Exposure": "exposicion_mas",
    "ConvolutionKernel": "kernel",
    "SliceThickness": "espesor_mm",
    "ReconstructionDiameter": "diametro_reconstruccion_mm",
    "DataCollectionDiameter": "diametro_adquisicion_mm",
    "CTDIvol": "ctdivol_mgy",
    "SpiralPitchFactor": "pitch",
    "SingleCollimationWidth": "colimacion_fila_mm",
    "TotalCollimationWidth": "colimacion_total_mm",
    "FilterType": "filtro_bowtie",
    "StudyDescription": "descripcion_estudio",
}


def _valor(v):
    if v is None:
        return np.nan
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        v = v[0] if len(v) else None
        return _valor(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        s = str(v).strip()
        return s if s else np.nan


def leer_encabezado(ds) -> dict:
    """ds: pydicom Dataset de UN corte."""
    return {col: _valor(getattr(ds, tag, None)) for tag, col in CAMPOS.items()}


def mas_efectivo(enc) -> float:
    """Índice de dosis del corte para anclar los fotones: la CORRIENTE del tubo
    en ese corte (mA, tag XRayTubeCurrent).

    Por qué no los mAs: en LIDC los tags Exposure y ExposureTime los llena
    cada fabricante a su manera (tiempo de una rotación, tiempo total del
    barrido, o vacío). Con ellos salían "mAs" de 1 a 7200, imposibles en un
    tórax. La corriente la informan todos igual y por corte (incluye la
    modulación). Los fotones son ∝ mA × tiempo de rotación / pitch: los dos
    últimos no son confiables en estos DICOM, así que quedan absorbidos en la
    constante Φ, que se calibra por fabricante en el 07a; lo que varíen dentro
    de un fabricante aparece como dispersión de Φ (medida)."""
    ma = enc.get("corriente_ma", np.nan)
    return float(ma) if _ok(ma) else np.nan


def _ok(x):
    return isinstance(x, (int, float, np.floating)) and np.isfinite(x) and x > 0


# Familias de kernel -> ventana de apodización del FBP (skimage.iradon).
# De más dura a más suave: ramp > shepp-logan > cosine > hamming > hann.
# Los kernels "de pulmón" y "de hueso" de los equipos REFUERZAN frecuencias
# medias-altas (más que la rampa): la rampa es la aproximación más dura que
# tenemos, no una copia. Naziroglu estima la ventana con el NPS de un cilindro
# de agua escaneado en el mismo equipo; en LIDC no hay esos barridos.
_FAMILIAS = [
    ("duro", r"^(LUNG|BONE|EDGE|DETAIL)$|^B[6-9]\d|^[DEL]$|^YA$|^YB$|^FC5\d|^FC8\d", "ramp"),
    # Intermedio: "cosine" y no "shepp-logan". En el 07a con LIDC, los Φ de
    # los estudios con kernel intermedio salían ~2,3 veces los de duro y suave
    # (que coinciden entre sí): con Shepp-Logan la simulación daba más ruido
    # que el real, o sea que esos kernels son más suaves (12 estudios).
    ("intermedio", r"^B[4-5]\d|^C$|^FC3\d|^FC4\d", "cosine"),
    ("suave", r"^(STANDARD|SOFT|SMOOTH)$|^B[0-3]\d|^[AB]$|^FC[0-2]\d", "hann"),
]


def familia_kernel(kernel) -> tuple[str, str]:
    """('duro'|'intermedio'|'suave'|'desconocido', ventana)."""
    if not isinstance(kernel, str):
        return "desconocido", "shepp-logan"
    k = kernel.strip().upper()
    for fam, patron, ventana in _FAMILIAS:
        if re.search(patron, k):
            return fam, ventana
    return "desconocido", "shepp-logan"


# Factor del filtro (SFF) del bowtie de CUERPO que Kramer et al. (2017, Phys
# Med Biol 62:781-809, Tablas 4-5) ajustaron para cada tomógrafo, sin datos del
# fabricante: variaron el SFF en Monte Carlo hasta que el CTDI calculado
# coincidió con el publicado (CTDosimetry) dentro del 5 %. Son justo equipos de
# la época de LIDC. (patrón en minúsculas del fabricante + modelo, SFF)
SFF_BOWTIE_KRAMER = [
    ("ge", "ultra", 0.95), ("ge", "pro 16", 1.17), ("ge", "pro16", 1.17), ("ge", "rt", 0.87),
    ("ge", "vct", 1.76), ("philips", "mx8000", 0.90), ("philips", "brilliance 16", 0.90),
    ("philips", "big bore", 0.94), ("philips", "brilliance 64", 1.01), ("philips", "brilliance 40", 1.01),
    ("siemens", "sensation 16", 0.66), ("siemens", "sensation 64", 1.27), ("siemens", "definition as", 1.57),
    ("toshiba", "aquilion 16", 1.71),
]


def sff_bowtie(fabricante, modelo):
    """(SFF, fuente). Si el modelo no está en la tabla de Kramer, la mediana
    de los 11 equipos (y en el análisis de incertidumbre, todo su rango)."""
    from . import config as C
    fab = str(fabricante).lower() if isinstance(fabricante, str) else ""
    mod = str(modelo).lower() if isinstance(modelo, str) else ""
    for f, m, sff in SFF_BOWTIE_KRAMER:
        if f in fab and m in mod:
            return sff, "Kramer 2017 (modelo del equipo)"
    return C.BOWTIE_SFF, "mediana de Kramer 2017 (modelo no tabulado)"
