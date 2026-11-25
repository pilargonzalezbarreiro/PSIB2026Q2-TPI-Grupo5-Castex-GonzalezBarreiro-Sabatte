"""Etapa 3b - Parámetros de adquisición de cada corte de trabajo (encabezado DICOM).

Para cada nódulo ya extraído (datos/cortes/*.npz) lee el encabezado del corte
axial que se usó: fabricante, kVp, mA, tiempo, mAs, kernel, espesor, CTDIvol
(si está). Sirve para dos cosas:
  1. describir la muestra (LIDC mezcla estudios de dosis, equipos y kernels
     muy distintos: no es una base "de dosis diagnóstica");
  2. anclar la simulación de baja dosis a la dosis de CADA estudio (07a, 07).

Solo lee encabezados (sin píxeles), así que es rápido. Si los DICOM ya se
borraron (03 --borrar-dicom), --descargar baja de nuevo esas series con
idc-index, lee los encabezados y (con --borrar-dicom) las vuelve a borrar.

Uso:  python scripts/03b_encabezados_dicom.py [--descargar] [--borrar-dicom]
Salidas:
  resultados/encabezados_dicom.csv       una fila por nódulo
  resultados/03b_resumen_adquisicion.csv resumen para describir la muestra
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.compat import configurar_pylidc, importar_pylidc  # noqa: E402
from tpi.encabezados import familia_kernel, leer_encabezado, mas_efectivo  # noqa: E402

PATRON = re.compile(r"(?P<paciente>.+)_s(?P<scan_id>\d+)_n(?P<nodulo>\d+)$")


def encabezados_de_la_serie(carpeta):
    """{z: dataset} de todos los cortes de la serie, sin leer los píxeles."""
    import pydicom
    salida = {}
    archivos = list(Path(carpeta).glob("*.dcm")) or [f for f in Path(carpeta).iterdir() if f.is_file()]
    for f in archivos:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:  # noqa: BLE001  (archivos que no son DICOM: .xml de LIDC, etc.)
            continue
        if hasattr(ds, "ImagePositionPatient"):
            salida[float(ds.ImagePositionPatient[2])] = ds
    return salida


def z_del_corte(scan, k):
    """z del corte k en el MISMO orden que usa scan.to_volume() (el del 03)."""
    zs = getattr(scan, "slice_zvals", None)
    if zs is not None and len(zs) > k:
        return float(zs[k])
    imgs = scan.load_all_dicom_images(verbose=False)  # más lento, mismo orden seguro
    return float(imgs[k].ImagePositionPatient[2])


def descargar(series_uid):
    from idc_index import IDCClient
    IDCClient().download_from_selection(
        downloadDir=str(C.DIR_DICOM), seriesInstanceUID=[series_uid],
        dirTemplate="%PatientID/%StudyInstanceUID/%SeriesInstanceUID", quiet=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descargar", action="store_true", help="bajar las series que ya no están")
    ap.add_argument("--borrar-dicom", action="store_true", help="borrar cada serie después de leerla")
    args = ap.parse_args()
    configurar_pylidc(C.DIR_DICOM)
    pl = importar_pylidc()

    cortes = []
    for ruta in sorted(C.DIR_CORTES.glob("*.npz")):
        m = PATRON.match(ruta.stem)
        with np.load(ruta) as z:
            k = int(z["indice_corte"]) if "indice_corte" in z.files else -1
        cortes.append(dict(paciente=m["paciente"], scan_id=int(m["scan_id"]), nodulo=int(m["nodulo"]),
                           indice_corte=k))
    cortes = pd.DataFrame(cortes)
    sel = pd.read_csv(C.DIR_RESULTADOS / "nodulos_seleccionados.csv")
    uid = sel.drop_duplicates("scan_id").set_index("scan_id")["series_uid"]

    filas, sin_dicom = [], 0
    for scan_id, grupo in cortes.groupby("scan_id"):
        scan = pl.query(pl.Scan).filter(pl.Scan.id == int(scan_id)).first()
        bajada = False
        try:
            carpeta = scan.get_path_to_dicom_files()
        except (RuntimeError, IOError):
            carpeta = None
        if (carpeta is None or not Path(carpeta).exists()) and args.descargar:
            descargar(uid[scan_id])
            bajada = True
            carpeta = scan.get_path_to_dicom_files()
        if carpeta is None or not Path(carpeta).exists():
            sin_dicom += len(grupo)
            continue
        dss = encabezados_de_la_serie(carpeta)
        zs = np.array(sorted(dss))
        for _, f in grupo.iterrows():
            z = z_del_corte(scan, f.indice_corte)
            ds = dss[zs[np.argmin(np.abs(zs - z))]]
            enc = leer_encabezado(ds)
            fam, ventana = familia_kernel(enc["kernel"])
            filas.append(dict(**f.to_dict(), z_mm=z, **enc, mas_efectivo=mas_efectivo(enc),
                              familia_kernel=fam, ventana_fbp=ventana))
        if args.borrar_dicom and bajada:
            shutil.rmtree(carpeta, ignore_errors=True)

    df = pd.DataFrame(filas)
    df.to_csv(C.DIR_RESULTADOS / "encabezados_dicom.csv", index=False)
    print(f"Encabezados leídos: {len(df)} nódulos | sin DICOM (usar --descargar): {sin_dicom}")
    if df.empty:
        return

    # Resumen para describir la muestra en el informe (por estudio, no por nódulo)
    est = df.drop_duplicates("scan_id")
    print(f"\n{len(est)} estudios")
    print("\nFabricante:\n" + est.fabricante.value_counts(dropna=False).to_string())
    print("\nFamilia de kernel:\n" + est.familia_kernel.value_counts(dropna=False).to_string())
    print("\nKernels:\n" + est.kernel.value_counts(dropna=False).head(15).to_string())
    num = est[["kvp", "corriente_ma", "mas_efectivo", "espesor_mm", "ctdivol_mgy"]]
    res = num.describe(percentiles=[0.1, 0.5, 0.9]).T
    res["faltantes"] = num.isna().sum()
    print("\n" + res.round(1).to_string())
    res.to_csv(C.DIR_RESULTADOS / "03b_resumen_adquisicion.csv")


if __name__ == "__main__":
    main()
