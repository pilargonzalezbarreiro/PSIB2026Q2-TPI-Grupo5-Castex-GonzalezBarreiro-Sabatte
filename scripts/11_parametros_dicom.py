"""Parámetros de adquisición de las tomografías usadas (encabezado DICOM).

No vuelve a abrir los DICOM: usa resultados/encabezados_dicom.csv, que el
script 03b ya armó leyendo el encabezado del corte de cada nódulo (kVp,
corriente del tubo, tiempo, Exposure, espesor, kernel, fabricante, modelo,
CTDIvol). Resume: mediana y rango de lo numérico, y cuántos estudios hay de
cada kernel / fabricante, en total (por estudio) y por grupo (por nódulo).

Sobre los mAs: el campo Exposure de LIDC no es comparable entre fabricantes
(en GE va de 1 a 7200, Philips no informa el tiempo, Siemens guarda mAs
efectivos ya divididos por el pitch; ver 07a). Por eso el índice de dosis es
la corriente (mA). Exposure se muestra solo como descripción, por fabricante,
y no se estima corriente x tiempo.

Salidas: resultados/11_parametros_dicom.csv          (una fila por estudio)
         resultados/11_resumen_parametros_dicom.csv  (la tabla para el informe)
Uso:     python scripts/11_parametros_dicom.py
Necesita resultados/encabezados_dicom.csv (03b) y resultados/nodulos_seleccionados.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.encabezados import familia_kernel  # noqa: E402

NUMERICOS = ["kvp", "corriente_ma", "espesor_mm", "ctdivol_mgy"]
CATEGORICOS = ["fabricante", "modelo", "kernel", "familia_kernel"]


def main():
    ruta = C.DIR_RESULTADOS / "encabezados_dicom.csv"
    if not ruta.exists():
        sys.exit(f"Falta {ruta}: se genera con scripts/03b_encabezados_dicom.py --descargar --borrar-dicom")
    enc = pd.read_csv(ruta)
    sel = pd.read_csv(C.DIR_RESULTADOS / "nodulos_seleccionados.csv")
    enc = enc.merge(sel[["paciente", "scan_id", "nodulo", "grupo", "conjunto"]],
                    on=["paciente", "scan_id", "nodulo"], how="left")
    enc["fabricante"] = enc.fabricante.fillna("?").astype(str).str.split().str[0].str.upper()
    enc["familia_kernel"] = [familia_kernel(k)[0] for k in enc.kernel]
    for c in NUMERICOS:
        if c not in enc:
            enc[c] = np.nan

    estudios = enc.drop_duplicates("scan_id")
    estudios.to_csv(C.DIR_RESULTADOS / "11_parametros_dicom.csv", index=False)
    print(f"Nódulos: {len(enc)} | estudios: {len(estudios)}")

    res = []
    # Todos: un valor por estudio. Por grupo: un valor por nódulo (cada nódulo
    # lleva los parámetros de su estudio; un estudio puede tener nódulos de los dos grupos).
    for nombre, sub, unidad in (("todos", estudios, "estudios"),
                                *((g, s, "nódulos") for g, s in enc.groupby("grupo"))):
        for col in NUMERICOS:
            v = pd.to_numeric(sub[col], errors="coerce").dropna()
            res.append(dict(grupo=nombre, unidad=unidad, parametro=col, n=len(v),
                            mediana=v.median() if len(v) else np.nan,
                            minimo=v.min() if len(v) else np.nan,
                            maximo=v.max() if len(v) else np.nan, valor=""))
        for col in CATEGORICOS:
            for val, n in sub[col].value_counts().items():
                res.append(dict(grupo=nombre, unidad=unidad, parametro=col, n=n, valor=val))
    res = pd.DataFrame(res)
    res.to_csv(C.DIR_RESULTADOS / "11_resumen_parametros_dicom.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))

    cols = [c for c in ("exposicion_mas", "tiempo_exposicion_ms", "corriente_ma") if c in estudios]
    print("\nPor qué el índice de dosis es la corriente y no los mAs (por estudio):")
    print(estudios.groupby("fabricante")[cols].describe(percentiles=[.5])
          .loc[:, (slice(None), ["count", "min", "50%", "max"])].to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
