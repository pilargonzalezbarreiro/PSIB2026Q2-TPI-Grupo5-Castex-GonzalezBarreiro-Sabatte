"""Control: ¿cambian los resultados si se usa el criterio JUSTO de aceptabilidad?

El análisis principal usa los nódulos con segmentación "aceptable" según el
criterio original: Dice(algoritmo, consenso) >= concordancia entre radiólogos
(112 nódulos de desarrollo). La devolución señaló que esa comparación favorece
al algoritmo, y el script 12 calcula el criterio justo: Dice medio contra CADA
radiólogo >= concordancia entre radiólogos (92 nódulos de desarrollo).

Este script repite, con las dos poblaciones, los AUC principales (validación
cruzada por paciente, IC 95 % por bootstrap de pacientes), con el mismo código
del 08:
  - diámetro solo, puntaje del interior (como el 06) y puntaje robusto;
  - en la imagen original (= el resultado del 06) y transferidos a cada dosis
    de la simulación principal.
Si los AUC y sus intervalos casi no cambian, la elección del criterio no
afecta las conclusiones.

Uso:    python comparaciones/sensibilidad_aceptables.py [--anclaje hibrido] [--sigma-e2 lidc]
Necesita: resultados/caracteristicas.csv (04), resultados/12_dice_vs_radiologos.csv (12)
          y el 07 con ese anclaje.
Salida:  resultados/comparaciones/sensibilidad_aceptables{sufijo}.csv
"""
import argparse
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.estadistica import META  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "e08", Path(__file__).resolve().parents[1] / "scripts" / "08_propuesta.py")
E8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E8)

CLAVES = ["paciente", "scan_id", "nodulo"]


def poblacion(df07, car, claves_ok):
    """Nódulos de desarrollo benignos/sospechosos cuyas claves están en claves_ok."""
    pob = car[car.grupo.isin(["benigno", "sospechoso"]) & (car.conjunto == "entrenamiento")]
    pob = pob[CLAVES + ["grupo", "frac_calcio"]].rename(columns={"frac_calcio": "calcio_original"})
    pob = pob.merge(claves_ok, on=CLAVES)
    return df07.drop(columns=["concordancia_radiologos"], errors="ignore").merge(pob, on=CLAVES)


def auc_de(df):
    medidas = [c for c in df.select_dtypes("number").columns
               if c not in META and c not in E8.NO_MEDIDAS and c != "calcio_original"]
    with redirect_stdout(io.StringIO()):
        rt = E8.ruido_tolerable(df, medidas)
    robustas = set(rt.loc[rt[f"CCC {E8.DOSIS_USO}"] >= E8.UMBRAL_REPRO, "medida"])
    pr, _ = E8.puntaje_robusto(df, medidas, robustas)
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anclaje", choices=["mas", "ruido", "hibrido"], default="hibrido")
    ap.add_argument("--sigma-e2", default="lidc")
    args = ap.parse_args()
    a = {"mas": "", "ruido": "_anclaje_ruido", "hibrido": "_anclaje_hibrido"}[args.anclaje]
    a += E8.sufijo_sigma_e2(args.sigma_e2, args.anclaje)
    ruta07 = C.DIR_RESULTADOS / f"07_baja_dosis{a}.csv"
    print(f"Simulación: {ruta07.name} | dosis de uso del puntaje robusto: {E8.DOSIS_USO}")

    df07 = pd.read_csv(ruta07)
    car = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    d12 = pd.read_csv(C.DIR_RESULTADOS / "12_dice_vs_radiologos.csv")
    criterios = {
        "original (vs consenso)": car.loc[car.aceptable.astype(bool), CLAVES],
        "justo (vs cada radiólogo)": d12.loc[d12.aceptable_justo.astype(bool), CLAVES],
    }
    filas = []
    for nombre, claves in criterios.items():
        df = poblacion(df07, car, claves)
        n = df.drop_duplicates(CLAVES).grupo.value_counts().to_dict()
        print(f"\nCriterio {nombre}: {sum(n.values())} nódulos {n}")
        pr = auc_de(df).assign(criterio=nombre, n=sum(n.values()))
        filas.append(pr)
    res = pd.concat(filas, ignore_index=True)
    dir_res = C.DIR_RESULTADOS / "comparaciones"
    dir_res.mkdir(parents=True, exist_ok=True)
    res.to_csv(dir_res / f"sensibilidad_aceptables{a}.csv", index=False)

    res["txt"] = [f"{x:.3f} [{lo:.2f}-{hi:.2f}]" for x, lo, hi in zip(res.auc_transferido, res.ic95_inf, res.ic95_sup)]
    orden = ["original"] + E8.DOSIS
    pd.set_option("display.width", 250)
    for puntaje in res.puntaje.unique():
        t = res[res.puntaje == puntaje].pivot(index="condicion", columns="criterio", values="txt").reindex(orden)
        print(f"\n== {puntaje}: AUC [IC 95 %] con cada criterio de aceptabilidad ==")
        print(t.to_string())
    d = res.pivot_table(index=["puntaje", "condicion"], columns="criterio", values="auc_transferido")
    dif = (d.iloc[:, 1] - d.iloc[:, 0]).abs()
    print(f"\nMayor diferencia de AUC entre criterios: {dif.max():.3f} "
          f"({' / '.join(dif.idxmax())}); compararla con el ancho de los IC de arriba")


if __name__ == "__main__":
    main()
