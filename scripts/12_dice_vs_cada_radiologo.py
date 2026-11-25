"""Comparación justa del Dice: algoritmo vs CADA radiólogo.

Antes:  aceptable = Dice(algoritmo, consenso) >= concordancia media entre pares de radiólogos
        (el consenso se arma con los mismos radiólogos, así que queda más cerca
        del algoritmo que un radiólogo cualquiera: comparación con ventaja).
Ahora:  aceptable_justo = Dice medio (algoritmo vs cada radiólogo) >= concordancia media entre pares
        (los dos números miden "parecido a UN radiólogo").

Usa TODOS los nódulos de resultados/caracteristicas.csv y la misma segmentación del script 04.
Salidas: resultados/12_dice_vs_radiologos.csv  (una fila por nódulo)
         resultados/12_resumen_dice_vs_radiologos.csv
Uso:     python scripts/12_dice_vs_cada_radiologo.py [--carpeta datos/cortes]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from _limitaciones_comun import C, cortes_y_mascaras

from tpi.datos import cargar_corte
from tpi.validacion import concordancia_radiologos, dice


def dice_vs_cada_radiologo(seg, radiologos):
    """Dice medio entre la segmentación y cada radiólogo que dibujó en este corte."""
    v = [dice(seg, r) for r in radiologos if r.any()]
    return float(np.mean(v)) if v else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    a = ap.parse_args()
    df = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv").reset_index(drop=True)
    filas = []
    for f, img, seg in cortes_y_mascaras(df, a.carpeta):
        d = cargar_corte(Path(a.carpeta) / f"{f.paciente}_s{int(f.scan_id)}_n{int(f.nodulo)}.npz")
        conc = concordancia_radiologos(d["radiologos"])
        dr = dice_vs_cada_radiologo(seg, d["radiologos"])
        filas.append(dict(paciente=f.paciente, scan_id=f.scan_id, nodulo=f.nodulo, grupo=f.grupo,
                          conjunto=f.conjunto, dice_vs_consenso=f.dice, dice_vs_cada_radiologo=dr,
                          concordancia_radiologos=conc, aceptable_original=bool(f.aceptable),
                          aceptable_justo=bool(np.isfinite(conc) and np.isfinite(dr) and dr >= conc)))
    t = pd.DataFrame(filas)
    t.to_csv(C.DIR_RESULTADOS / "12_dice_vs_radiologos.csv", index=False)
    res = []
    for nombre, sub in (("todos", t), *t.groupby("conjunto")):
        res.append(dict(conjunto=nombre, n=len(sub),
                        dice_medio_vs_consenso=round(sub.dice_vs_consenso.mean(), 3),
                        dice_medio_vs_cada_radiologo=round(sub.dice_vs_cada_radiologo.mean(), 3),
                        concordancia_media_radiologos=round(sub.concordancia_radiologos.mean(), 3),
                        aceptables_original=int(sub.aceptable_original.sum()),
                        aceptables_justo=int(sub.aceptable_justo.sum()),
                        dejan_de_ser_aceptables=int((sub.aceptable_original & ~sub.aceptable_justo).sum()),
                        pasan_a_aceptables=int((~sub.aceptable_original & sub.aceptable_justo).sum())))
    for (conj, g), sub in t.groupby(["conjunto", "grupo"]):
        res.append(dict(conjunto=f"{conj}/{g}", n=len(sub),
                        aceptables_original=int(sub.aceptable_original.sum()),
                        aceptables_justo=int(sub.aceptable_justo.sum())))
    res = pd.DataFrame(res)
    res.to_csv(C.DIR_RESULTADOS / "12_resumen_dice_vs_radiologos.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
