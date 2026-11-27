"""Limitación 2 - Textura (GLCM) e histograma SIN los píxeles de calcio.

En vez de sacar los nódulos con calcio, se sacan solo los PÍXELES de calcio
(> HU_CALCIO) de la máscara de cada nódulo y se recalculan asimetría, desvío,
curtosis y las medidas GLCM. Así se usan los 112 nódulos (todos).
Compara el AUC original (con calcio) contra el AUC sin los píxeles de calcio
y repite el Spearman parcial de cada medida controlando por tamaño.

Anillo de volumen parcial (--dilatar): el desenfoque del tomógrafo "derrama"
el calcio sobre los píxeles vecinos, que quedan en ~100-200 HU, por debajo
del umbral. Si solo se sacan los píxeles > 200 HU, ese anillo queda adentro y
sigue aportando diferencias grandes con el tejido (el contraste GLCM las eleva
al cuadrado). Por eso se repite sacando también 1 y 2 píxeles alrededor de
cada mancha de calcio (dilatación con un disco). Si el contraste de los
benignos calcificados baja al sacar el anillo, el calcio sí lo inflaba.

Salidas: resultados/10_medidas_sin_pixeles_calcio{_dilN}.csv  (una fila por nódulo)
         resultados/10_comparacion_sin_pixeles_calcio{_dilN}.csv
         resultados/10_resumen_dilatacion.csv  (las dilataciones lado a lado)
Uso:     python scripts/10_sin_pixeles_calcio.py [--carpeta datos/cortes] [--umbral 200] [--dilatar 0 1 2]
         (sin argumentos corre las tres dilataciones: 0, 1 y 2 píxeles)
"""
import argparse

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology
from _limitaciones_comun import C, auc, cortes_y_mascaras, nodulos_aceptables

from tpi.caracteristicas import caracteristicas_interior
from tpi.estadistica import spearman_parcial

MEDIDAS = ["hu_asimetria", "hu_desvio", "hu_curtosis", "glcm_energia", "glcm_entropia",
           "glcm_contraste", "glcm_homogeneidad", "glcm_correlacion",
           "glcm_cluster_shade", "glcm_cluster_prominence"]


def zona_calcio(img, mascara, umbral, dilatar_px=0):
    """Píxeles de calcio del nódulo (> umbral) y, si dilatar_px > 0, también
    el anillo de dilatar_px píxeles alrededor de cada mancha (dentro del nódulo)."""
    cal = (img > umbral) & mascara
    if dilatar_px > 0 and cal.any():
        cal = ndi.binary_dilation(cal, structure=morphology.disk(dilatar_px)) & mascara
    return cal


def medidas_sin_calcio(img, mascara, umbral, dilatar_px=0):
    """Mismas medidas del script 04 pero con la máscara sin la zona de calcio."""
    m2 = mascara & ~zona_calcio(img, mascara, umbral, dilatar_px)
    d = caracteristicas_interior(img, m2)
    return {k: d.get(k, np.nan) for k in MEDIDAS}, int(mascara.sum() - m2.sum())


def correr(df, carpeta, umbral, dilatar_px):
    sufijo = f"_dil{dilatar_px}" if dilatar_px else ""
    filas = []
    for f, img, m in cortes_y_mascaras(df, carpeta):
        med, n_quit = medidas_sin_calcio(img, m, umbral, dilatar_px)
        filas.append(dict(paciente=f.paciente, scan_id=f.scan_id, nodulo=f.nodulo, grupo=f.grupo,
                          diametro_eq_mm=f.diametro_eq_mm, frac_calcio=f.frac_calcio,
                          px_calcio_quitados=n_quit, **{k + "_sc": v for k, v in med.items()},
                          **{k + "_orig": f[k] for k in MEDIDAS}))
    t = pd.DataFrame(filas)
    t.to_csv(C.DIR_RESULTADOS / f"10_medidas_sin_pixeles_calcio{sufijo}.csv", index=False)
    sosp = (t.grupo == "sospechoso").to_numpy()
    sin = (t.frac_calcio <= C.FRAC_CALCIO_MAX).to_numpy()
    res = []
    for k in MEDIDAS:
        r = dict(medida=k)
        for etiqueta, suf in (("con_calcio_pixeles", "_orig"), ("sin_pixeles_calcio", "_sc")):
            a_, may, ns, nb = auc(t[k + suf], sosp)
            r[f"auc_{etiqueta}"], r[f"mayor_en_{etiqueta}"] = round(a_, 3), may
        a_sub, _, ns2, nb2 = auc(t.loc[sin, k + "_sc"], sosp[sin])  # y en el subgrupo sin calcio
        r["auc_sin_px_calcio_subgrupo_sin_calcio"] = round(a_sub, 3)
        ok = t[k + "_sc"].notna() & t.diametro_eq_mm.notna()
        r["spearman_parcial_vs_tamano_sc"] = round(spearman_parcial(
            t.loc[ok, k + "_sc"], sosp[ok].astype(int), t.loc[ok, "diametro_eq_mm"]), 3)
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(C.DIR_RESULTADOS / f"10_comparacion_sin_pixeles_calcio{sufijo}.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\n######## Sacando el calcio (> {umbral:g} HU)"
          + (f" y un anillo de {dilatar_px} px alrededor" if dilatar_px else " (solo esos píxeles)") + " ########")
    print(f"n = {len(t)} ({sosp.sum()} sospechosos / {(~sosp).sum()} benignos); subgrupo sin calcio: {sin.sum()}")
    con = t[t.frac_calcio > C.FRAC_CALCIO_MAX]
    print(f"Píxeles quitados en los nódulos con calcio: mediana {con.px_calcio_quitados.median():.0f} "
          f"(máx {con.px_calcio_quitados.max()})")
    print(res.to_string(index=False))
    return t, res


def resumen(resultados):
    """Lo que responde la pregunta: ¿el contraste (y las demás) de los benignos
    calcificados baja al sacar el calcio y su anillo?"""
    filas = []
    for dil, (t, res) in resultados.items():
        benig_cal = t[(t.grupo == "benigno") & (t.frac_calcio > C.FRAC_CALCIO_MAX)]
        benig_sin = t[(t.grupo == "benigno") & (t.frac_calcio <= C.FRAC_CALCIO_MAX)]
        sosp = t[t.grupo == "sospechoso"]
        for k in MEDIDAS:
            fila = res.set_index("medida").loc[k]
            filas.append(dict(medida=k, anillo_px=dil, auc_original=fila.auc_con_calcio_pixeles,
                              auc_sin_calcio=fila.auc_sin_pixeles_calcio,
                              mayor_en_sin_calcio=fila.mayor_en_sin_pixeles_calcio,
                              mediana_benignos_calcificados_orig=benig_cal[k + "_orig"].median(),
                              mediana_benignos_calcificados_sin_calcio=benig_cal[k + "_sc"].median(),
                              mediana_benignos_sin_calcio=benig_sin[k + "_sc"].median(),
                              mediana_sospechosos=sosp[k + "_sc"].median()))
    r = pd.DataFrame(filas)
    r.to_csv(C.DIR_RESULTADOS / "10_resumen_dilatacion.csv", index=False)
    print("\n######## Resumen: contraste GLCM y asimetría según cuánto se saca alrededor del calcio ########")
    cols = ["medida", "anillo_px", "auc_original", "auc_sin_calcio", "mayor_en_sin_calcio",
            "mediana_benignos_calcificados_orig", "mediana_benignos_calcificados_sin_calcio",
            "mediana_benignos_sin_calcio", "mediana_sospechosos"]
    print(r[r.medida.isin(["glcm_contraste", "hu_asimetria"])][cols].round(3).to_string(index=False))
    print("Si la mediana de los benignos calcificados baja hacia la de los benignos sin calcio al sacar el "
          "anillo, el calcio (con su volumen parcial) inflaba la medida.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    ap.add_argument("--umbral", type=float, default=C.HU_CALCIO)
    ap.add_argument("--dilatar", type=int, nargs="*", default=[0, 1, 2],
                    help="píxeles de anillo a sacar alrededor del calcio (por defecto 0, 1 y 2)")
    a = ap.parse_args()
    df = nodulos_aceptables()
    print(f"Nódulos de desarrollo con segmentación aceptable: {len(df)}")
    resultados = {dil: correr(df, a.carpeta, a.umbral, dil) for dil in a.dilatar}
    if len(resultados) > 1:
        resumen(resultados)


if __name__ == "__main__":
    main()
