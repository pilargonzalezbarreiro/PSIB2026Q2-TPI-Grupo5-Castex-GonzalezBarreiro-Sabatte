"""Puntaje robusto con TODA la selección dentro de cada pliegue.

En el 08 el filtro "AUC >= 0,7" ya se hace dentro de cada pliegue (usa solo los
pacientes de armado), pero el filtro "CCC >= 0,9 a dosis x1/4" se calcula con
todo el conjunto de desarrollo. El CCC no usa las etiquetas benigno/sospechoso,
pero la crítica es válida en principio: la lista de candidatas "vio" a los
pacientes de prueba. Acá el CCC también se calcula SOLO con los pacientes de
armado de cada pliegue, y recién después se elige por AUC.

Compara contra el resultado del 08 (selección con todo el desarrollo).
Salida: resultados/13_puntaje_robusto_estricto{_anclaje_...}{_usoN}.csv
Uso:    python scripts/13_puntaje_robusto_estricto.py [--anclaje ruido|mas|hibrido] [--dosis-uso 4|10] [--sigma-e2 lidc]
        (por defecto el anclaje principal, ruido, y la dosis de uso del 08, x1/4)
Necesita el 07 y el 08 corridos con el mismo anclaje y la misma dosis de uso.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.estadistica import META  # noqa: E402
from tpi.puntaje import auc, ic_bootstrap  # noqa: E402

_spec = importlib.util.spec_from_file_location("e08", Path(__file__).with_name("08_propuesta.py"))
E8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E8)
E7 = E8.E7


def candidatas_en_pliegue(base_tr, uso_tr, medidas):
    """Medidas con CCC >= 0,9 (original vs dosis de uso) usando SOLO los nódulos de armado."""
    ok = []
    for c in medidas:
        x, y = base_tr[c].to_numpy(float), uso_tr[c].to_numpy(float)
        if E8.degenerada(x):
            continue
        v = E7.ccc(x, y)
        if np.isfinite(v) and v >= E8.UMBRAL_REPRO:
            ok.append(c)
    return ok


def cv_estricto(base, uso, otra, y, pacientes, medidas, k=5):
    pl = E7.pliegues(pacientes)
    s = np.full(len(y), np.nan)
    for f in np.unique(pl):
        tr, te = pl != f, pl == f
        cols = candidatas_en_pliegue(base[tr], uso[tr], medidas)
        if not cols or len(np.unique(y[tr])) < 2:
            continue
        s[te] = E8.PuntajeRobusto(k).ajustar(base.loc[tr, cols], y[tr]).puntuar(otra.loc[te, cols])
    return s


def correr(df, etiqueta, k=5):
    medidas = [c for c in df.select_dtypes("number").columns
               if c not in META and c not in E8.NO_MEDIDAS and c != "calcio_original"]
    entorno = [c for c in medidas if c.startswith("anillo_") or c in E7.ENTORNO_EXTRA]
    interior = [c for c in medidas if c not in entorno]
    base, uso = E8.tabla(df, E8.DOSIS_USO)[0].reset_index(), E8.tabla(df, E8.DOSIS_USO)[1].reset_index()
    y = (base.grupo == "sospechoso").to_numpy(int)
    pac = base.paciente.to_numpy()
    filas = []
    for cond in ["original"] + E8.DOSIS:
        otra = E8.tabla(df, cond)[1].reset_index()
        s = cv_estricto(base, uso, otra, y, pac, interior, k)
        ok = np.isfinite(s)
        lo, hi = ic_bootstrap(y[ok], s[ok], base.paciente[ok]) if ok.sum() > 10 else (np.nan, np.nan)
        filas.append(dict(grupo_analisis=etiqueta, condicion=cond, n=int(ok.sum()),
                          auc_estricto=auc(y[ok], s[ok]) if ok.sum() > 10 else np.nan,
                          ic95_inf=lo, ic95_sup=hi))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anclaje", choices=["mas", "ruido", "hibrido"], default="ruido")
    ap.add_argument("--dosis-uso", type=int, default=4)
    ap.add_argument("--sigma-e2", default=None,
                    help="igual que en el 08: el σe² con el que se corrió el 07 (número, 'lidc' o 'lidc_sup')")
    args = ap.parse_args()
    E8.DOSIS_USO = f"dosis ×1/{args.dosis_uso}"
    if E8.DOSIS_USO not in E8.DOSIS:
        sys.exit(f"{E8.DOSIS_USO} no está entre las condiciones simuladas: {E8.DOSIS}")
    a = {"mas": "", "ruido": "_anclaje_ruido", "hibrido": "_anclaje_hibrido"}[args.anclaje]
    a += E8.sufijo_sigma_e2(args.sigma_e2, args.anclaje)
    uso = "" if args.dosis_uso == 4 else f"_uso{args.dosis_uso}"
    print(f"Anclaje: {args.anclaje} (resultados/07_baja_dosis{a}.csv) | dosis de uso: {E8.DOSIS_USO}")
    df = E8.poblacion(pd.read_csv(C.DIR_RESULTADOS / f"07_baja_dosis{a}.csv"))
    sin = df[df.calcio_original <= C.FRAC_CALCIO_MAX]
    res = pd.concat([correr(df, "todos"), correr(sin, "sin calcio")], ignore_index=True)
    # lo que había en el 08 (selección con todo el desarrollo)
    for suf, nombre in (("", "todos"), ("_sincalcio", "sin calcio")):
        r = pd.read_csv(C.DIR_RESULTADOS / f"08_puntaje_robusto{a}{uso}{suf}.csv")
        r = r[r.puntaje == "interior robusto (propuesta)"].set_index("condicion").auc_transferido
        m = res.grupo_analisis == nombre
        res.loc[m, "auc_seleccion_en_08"] = res.loc[m, "condicion"].map(r).to_numpy()
    res["diferencia"] = res.auc_estricto - res.auc_seleccion_en_08
    res.to_csv(C.DIR_RESULTADOS / f"13_puntaje_robusto_estricto{a}{uso}.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
