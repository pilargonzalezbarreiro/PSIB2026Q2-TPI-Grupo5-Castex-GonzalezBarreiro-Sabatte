"""Parámetros de adquisición de las tomografías usadas (encabezado DICOM).

Lee UN corte (el del medio) de cada serie y guarda kVp, corriente del tubo,
tiempo de exposición, mAs, espesor de corte, kernel de reconstrucción,
fabricante y modelo. Después resume: mediana y rango de lo numérico, y
cuántos estudios hay de cada kernel / fabricante, en total y por grupo.

Salidas: resultados/11_parametros_dicom.csv          (una fila por tomografía)
         resultados/11_resumen_parametros_dicom.csv  (la tabla para el informe)
Uso:     python scripts/11_parametros_dicom.py [--carpeta datos/LIDC-IDRI]
Necesita los DICOM en disco (no sirve si se borraron con --borrar-dicom:
en ese caso esas series figuran como 'sin DICOM').
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402

from tpi import config as C  # noqa: E402

CAMPOS = {"KVP": "kvp", "XRayTubeCurrent": "corriente_ma", "ExposureTime": "tiempo_ms",
          "Exposure": "exposicion_mas", "SliceThickness": "espesor_mm",
          "ConvolutionKernel": "kernel", "Manufacturer": "fabricante",
          "ManufacturerModelName": "modelo", "CTDIvol": "ctdi_vol_mgy"}
NUMERICOS = ["kvp", "corriente_ma", "tiempo_ms", "mas", "espesor_mm", "ctdi_vol_mgy"]
CATEGORICOS = ["kernel", "fabricante", "modelo"]


def carpeta_serie(raiz, paciente, series_uid):
    for p in Path(raiz).glob(f"{paciente}/*/{series_uid}"):
        return p
    for p in Path(raiz).rglob(series_uid):  # por si la estructura es otra
        return p
    return None


def leer_encabezado(carpeta):
    archivos = sorted(Path(carpeta).glob("*.dcm"))
    if not archivos:
        return None
    ds = pydicom.dcmread(archivos[len(archivos) // 2], stop_before_pixels=True)
    fila = {}
    for etiqueta, nombre in CAMPOS.items():
        v = ds.get(etiqueta)
        fila[nombre] = str(v).strip() if nombre in CATEGORICOS else (float(v) if v not in (None, "") else np.nan)
    # Muchos estudios de LIDC no traen 'Exposure' (mAs): se estima corriente x tiempo
    mas = fila.pop("exposicion_mas")
    if np.isnan(mas) and not np.isnan(fila["corriente_ma"]) and not np.isnan(fila["tiempo_ms"]):
        mas = fila["corriente_ma"] * fila["tiempo_ms"] / 1000
    fila["mas"] = mas
    return fila


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_DICOM))
    a = ap.parse_args()
    sel = pd.read_csv(C.DIR_RESULTADOS / "nodulos_seleccionados.csv")
    series = sel.drop_duplicates("series_uid")[["paciente", "scan_id", "series_uid", "grupo"]]
    filas = []
    for _, s in series.iterrows():
        c = carpeta_serie(a.carpeta, s.paciente, s.series_uid)
        h = leer_encabezado(c) if c else None
        filas.append(dict(paciente=s.paciente, scan_id=s.scan_id, series_uid=s.series_uid,
                          grupo=s.grupo, con_dicom=h is not None, **(h or {})))
    t = pd.DataFrame(filas)
    t.to_csv(C.DIR_RESULTADOS / "11_parametros_dicom.csv", index=False)
    ok = t[t.con_dicom]
    print(f"Tomografías: {len(t)} | con DICOM en disco: {len(ok)} | sin DICOM: {len(t) - len(ok)}")
    if ok.empty:
        sys.exit("No se encontró ningún DICOM: revisá --carpeta.")

    res = []
    for nombre, sub in (("todos", ok), *ok.groupby("grupo")):
        for col in NUMERICOS:
            v = sub[col].dropna()
            res.append(dict(grupo=nombre, parametro=col, n=len(v),
                            mediana=v.median() if len(v) else np.nan,
                            minimo=v.min() if len(v) else np.nan,
                            maximo=v.max() if len(v) else np.nan, valor=""))
        for col in CATEGORICOS:
            for val, n in sub[col].value_counts().items():
                res.append(dict(grupo=nombre, parametro=col, n=n, valor=val))
    res = pd.DataFrame(res)
    res.to_csv(C.DIR_RESULTADOS / "11_resumen_parametros_dicom.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
