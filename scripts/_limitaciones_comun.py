"""Funciones comunes de los scripts 09 y 10 (limitaciones del calcio).
Re-usa EXACTAMENTE la misma segmentación que el script 04, así los números
son comparables con resultados/caracteristicas.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.datos import cargar_corte  # noqa: E402
from tpi.segmentacion import segmentar_nodulo  # noqa: E402


def parametros_guardados():
    ajuste = C.DIR_RESULTADOS / "04_ajuste_umbral.csv"
    if ajuste.exists():
        t = pd.read_csv(ajuste)
        if "radio_cierre_mm" in t:
            m = t.loc[t.dice_medio.idxmax()]
            return int(m["umbral_hu"]), float(m["radio_cierre_mm"])
    return C.UMBRAL_HU, C.RADIO_CIERRE_PULMON_MM


def nodulos_aceptables(conjunto="entrenamiento"):
    """Misma selección que 05_estadistica.py: desarrollo + segmentación aceptable."""
    df = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    df = df[df["conjunto"] == conjunto] if df["conjunto"].notna().any() else df
    return df[df["aceptable"]].reset_index(drop=True)


def cortes_y_mascaras(df, carpeta=C.DIR_CORTES):
    """Para cada fila de df devuelve (fila, img_hu, mascara) con la máscara del script 04."""
    T, R = parametros_guardados()
    for _, f in df.iterrows():
        ruta = Path(carpeta) / f"{f.paciente}_s{int(f.scan_id)}_n{int(f.nodulo)}.npz"
        if not ruta.exists():
            print("  falta", ruta.name)
            continue
        d = cargar_corte(ruta)
        seg = segmentar_nodulo(d, umbral=T, radio_cierre_mm=R)
        yield f, d["img"], seg.astype(bool)


def auc(valores, es_sospechoso):
    """AUC (Mann-Whitney) y en qué grupo la medida es mayor. 'auc' es siempre >= 0,5."""
    v = np.asarray(valores, float)
    s = np.asarray(es_sospechoso, bool)
    ok = ~np.isnan(v)
    v, s = v[ok], s[ok]
    if s.sum() < 2 or (~s).sum() < 2:
        return np.nan, "", int(s.sum()), int((~s).sum())
    u = stats.mannwhitneyu(v[s], v[~s]).statistic
    a = u / (s.sum() * (~s).sum())
    return max(a, 1 - a), ("sospechoso" if a >= .5 else "benigno"), int(s.sum()), int((~s).sum())
