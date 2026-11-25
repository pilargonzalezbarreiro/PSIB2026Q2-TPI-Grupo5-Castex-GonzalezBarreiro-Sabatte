"""¿Cambian las conclusiones del 07 según cómo se anclan los fotones?

  mas     (07_baja_dosis.csv):                 mAs del DICOM × Φ del fabricante (07a)
  ruido   (07_baja_dosis_anclaje_ruido.csv):   ruido medido en cada imagen
  hibrido (07_baja_dosis_anclaje_hibrido.csv): ruido de la imagen + ruido electrónico

Ninguno de los dos es "la verdad": el primero arrastra el error de usar un Φ
por fabricante; el segundo, el de medir el ruido original y suponerlo todo
cuántico. Si las conclusiones (qué medidas son reproducibles, cuánto cae el
AUC) no cambian entre los dos, no dependen de esa elección.

Compara cada anclaje alternativo contra el de mAs (los que estén corridos).
Uso (después de correr el 07 con --anclaje mas, ruido y/o hibrido):
  python comparaciones/comparar_anclajes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402

R = C.DIR_RESULTADOS
CLAVES = ["paciente", "scan_id", "nodulo", "condicion"]


def main():
    for alt in ("ruido", "hibrido"):
        if (R / f"07_baja_dosis_anclaje_{alt}.csv").exists():
            print(f"\n################ mAs vs {alt} ################")
            comparar(alt)


def comparar(alt):
    suf = f"_anclaje_{alt}"
    a = pd.read_csv(R / "07_baja_dosis.csv")
    b = pd.read_csv(R / f"07_baja_dosis{suf}.csv")
    m = a.merge(b, on=CLAVES, suffixes=("_mas", "_alt"))
    m = m[m.condicion.str.startswith("dosis")].copy()
    m["cociente"] = m.ruido_agregado_hu_mas / m.ruido_agregado_hu_alt
    print(f"== Ruido agregado (pulmón): anclaje por mAs / {alt} ==")
    print(m.groupby("condicion").cociente.describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]]
          .round(2).to_string())

    ra = pd.read_csv(R / "07_reproducibilidad.csv", index_col=0)
    rb = pd.read_csv(R / f"07_reproducibilidad{suf}.csv", index_col=0)
    cols = [c for c in ra.columns if c.startswith("dosis") and c in rb]
    dif = (ra[cols] - rb.reindex(ra.index)[cols])
    print(f"\n== CCC: diferencia (mAs - {alt}) por condición ==")
    print(dif.abs().describe().loc[["mean", "max"]].round(3).to_string())
    for c in cols:
        ca, cb = ra[c] >= 0.9, rb.reindex(ra.index)[c] >= 0.9
        cambian = sorted(ra.index[ca != cb])
        print(f"   {c}: medidas que cambian de 'robusta' (CCC ≥ 0,9) a no, o al revés: "
              f"{', '.join(cambian) if cambian else 'ninguna'}")

    ca = pd.read_csv(R / "07_clasificacion.csv")
    cb = pd.read_csv(R / f"07_clasificacion{suf}.csv")
    t = ca.merge(cb, on=["condicion", "puntaje"], suffixes=("_mas", "_alt"))
    t["dif_auc_transferido"] = t.auc_transferido_mas - t.auc_transferido_alt
    print(f"\n== AUC transferido: mAs vs {alt} ==")
    print(t[["condicion", "puntaje", "auc_transferido_mas", "auc_transferido_alt", "dif_auc_transferido"]]
          .round(3).to_string(index=False))
    print(f"\nMáxima diferencia de AUC: {np.nanmax(np.abs(t.dif_auc_transferido)):.3f} "
          f"(compararla con el ancho del IC 95 % por bootstrap del 07)")


if __name__ == "__main__":
    main()
