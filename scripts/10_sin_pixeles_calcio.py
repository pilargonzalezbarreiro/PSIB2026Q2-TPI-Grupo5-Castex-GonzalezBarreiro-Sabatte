"""Limitación 2 - Textura (GLCM) e histograma SIN los píxeles de calcio.

En vez de sacar los nódulos con calcio, se sacan solo los PÍXELES de calcio
(> HU_CALCIO) de la máscara de cada nódulo y se recalculan asimetría, desvío,
curtosis y las medidas GLCM. Así se usan los 112 nódulos (todos).
Compara el AUC original (con calcio) contra el AUC sin los píxeles de calcio
y repite el Spearman parcial de la asimetría controlando por tamaño.

Salidas: resultados/10_medidas_sin_pixeles_calcio.csv  (una fila por nódulo)
         resultados/10_comparacion_sin_pixeles_calcio.csv
Uso:     python scripts/10_sin_pixeles_calcio.py [--carpeta datos/cortes] [--umbral 200]
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from _limitaciones_comun import C, auc, cortes_y_mascaras, nodulos_aceptables

from tpi.caracteristicas import caracteristicas_interior
from tpi.estadistica import spearman_parcial

MEDIDAS = ["hu_asimetria", "hu_desvio", "hu_curtosis", "glcm_energia", "glcm_entropia",
           "glcm_contraste", "glcm_homogeneidad", "glcm_correlacion",
           "glcm_cluster_shade", "glcm_cluster_prominence"]


def medidas_sin_calcio(img, mascara, umbral):
    """Mismas medidas del script 04 pero con la máscara sin los píxeles > umbral."""
    m2 = mascara & ~(img > umbral)
    d = caracteristicas_interior(img, m2)
    return {k: d.get(k, np.nan) for k in MEDIDAS}, int(mascara.sum() - m2.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    ap.add_argument("--umbral", type=float, default=C.HU_CALCIO)
    a = ap.parse_args()
    df = nodulos_aceptables()
    print(f"Nódulos de desarrollo con segmentación aceptable: {len(df)}")
    filas = []
    for f, img, m in cortes_y_mascaras(df, a.carpeta):
        med, n_quit = medidas_sin_calcio(img, m, a.umbral)
        filas.append(dict(paciente=f.paciente, scan_id=f.scan_id, nodulo=f.nodulo, grupo=f.grupo,
                          diametro_eq_mm=f.diametro_eq_mm, frac_calcio=f.frac_calcio,
                          px_calcio_quitados=n_quit, **{k + "_sc": v for k, v in med.items()},
                          **{k + "_orig": f[k] for k in MEDIDAS}))
    t = pd.DataFrame(filas)
    t.to_csv(C.DIR_RESULTADOS / "10_medidas_sin_pixeles_calcio.csv", index=False)
    sosp = (t.grupo == "sospechoso").to_numpy()
    sin = (t.frac_calcio <= C.FRAC_CALCIO_MAX).to_numpy()
    res = []
    for k in MEDIDAS:
        r = dict(medida=k)
        for etiqueta, sufijo in (("con_calcio_pixeles", "_orig"), ("sin_pixeles_calcio", "_sc")):
            a_, may, ns, nb = auc(t[k + sufijo], sosp)
            r[f"auc_{etiqueta}"], r[f"mayor_en_{etiqueta}"] = round(a_, 3), may
        a_sub, _, ns2, nb2 = auc(t.loc[sin, k + "_sc"], sosp[sin])  # y en el subgrupo sin calcio
        r["auc_sin_px_calcio_subgrupo_sin_calcio"] = round(a_sub, 3)
        ok = t[k + "_sc"].notna() & t.diametro_eq_mm.notna()
        r["spearman_parcial_vs_tamano_sc"] = round(spearman_parcial(
            t.loc[ok, k + "_sc"], sosp[ok].astype(int), t.loc[ok, "diametro_eq_mm"]), 3)
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(C.DIR_RESULTADOS / "10_comparacion_sin_pixeles_calcio.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"n = {len(t)} ({sosp.sum()} sospechosos / {(~sosp).sum()} benignos); "
          f"subgrupo sin calcio: {sin.sum()}")
    print(f"Píxeles de calcio quitados: mediana {t.px_calcio_quitados.median():.0f} por nódulo "
          f"(máx {t.px_calcio_quitados.max()})")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
