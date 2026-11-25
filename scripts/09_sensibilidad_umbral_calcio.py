"""Limitación 1 - ¿Cambia algo si el umbral de calcio no es 200 HU?

Repite la clasificación 'con calcio / sin calcio' con umbrales de 130, 200 y
300 HU (se mantiene la regla: con calcio = más del 2 % del área sobre el umbral)
y, en cada subgrupo 'sin calcio', calcula el AUC de las medidas principales.

Salida: resultados/09_sensibilidad_umbral_calcio.csv
Uso:    python scripts/09_sensibilidad_umbral_calcio.py [--carpeta datos/cortes]
Necesita: datos/cortes/*.npz (script 03) y resultados/caracteristicas.csv (script 04).
"""
import argparse

import numpy as np
import pandas as pd
from _limitaciones_comun import C, auc, cortes_y_mascaras, nodulos_aceptables

UMBRALES = (130, 200, 300)
FRACS = (0.02,)               # se puede agregar 0.0, 0.05 ...
MEDIDAS = ["diametro_eq_mm", "glcm_contraste", "hu_asimetria", "nitidez_borde_hu_mm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    a = ap.parse_args()
    df = nodulos_aceptables()
    print(f"Nódulos de desarrollo con segmentación aceptable: {len(df)}")

    frac = {u: [] for u in UMBRALES}
    idx = []
    for f, img, m in cortes_y_mascaras(df, a.carpeta):
        idx.append(f.name)
        for u in UMBRALES:
            frac[u].append(float((img[m] > u).mean()))
    sub = df.loc[idx].copy()
    for u in UMBRALES:
        sub[f"frac_{u}"] = frac[u]

    filas = []
    for u in UMBRALES:
        for fmax in FRACS:
            con = sub[f"frac_{u}"] > fmax
            sin = sub[~con]
            es_sosp = sin["grupo"].str.lower().str.startswith("sosp")
            fila = dict(umbral_hu=u, frac_max=fmax, n_total=len(sub),
                        con_calcio_benig=int((con & (sub.grupo.str.lower().str.startswith("benig"))).sum()),
                        con_calcio_sosp=int((con & (sub.grupo.str.lower().str.startswith("sosp"))).sum()),
                        n_sin_calcio=len(sin), sin_calcio_sosp=int(es_sosp.sum()),
                        sin_calcio_benig=int((~es_sosp).sum()))
            for med in MEDIDAS:
                r = auc(sin[med], es_sosp)
                fila[f"auc_{med}"] = round(r[0], 3)
                fila[f"mayor_en_{med}"] = r[1]
            filas.append(fila)
    out = pd.DataFrame(filas)
    ruta = C.DIR_RESULTADOS / "09_sensibilidad_umbral_calcio.csv"
    out.to_csv(ruta, index=False)
    pd.set_option("display.width", 250)
    print(out.T.to_string())
    print("->", ruta)


if __name__ == "__main__":
    main()
