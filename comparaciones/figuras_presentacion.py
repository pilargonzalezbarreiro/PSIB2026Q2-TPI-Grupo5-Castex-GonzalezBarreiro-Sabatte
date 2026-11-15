"""Dos figuras para la presentación, con los cortes reales de datos/cortes.

1. figuras/presentacion_segmentacion.png
   El nódulo pegado a la pared en el que más mejora la segmentación al
   restringir el crecimiento al pulmón cerrado: antes (crecimiento libre)
   y después (pulmón cerrado), contra el consenso de los radiólogos.

2. figuras/presentacion_calcio.png
   Un nódulo SIN calcio a dosis x1/10: píxeles que pasan 200 HU con el umbral
   simple (calcio falso) y lo que queda con el calcio robusto
   (mediana 3x3 + umbral + etiquetado, manchas de al menos 3 píxeles).

Usa el mismo umbral y radio de cierre que el resto del pipeline
(resultados/04_ajuste_umbral.csv) y el mismo ruido que el 07 (misma semilla
por nódulo), así que las imágenes son las del análisis, no otras.

Uso:  python comparaciones/figuras_presentacion.py
"""
import importlib.util
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import measure  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.datos import cargar_corte  # noqa: E402
from tpi.segmentacion import crecimiento_regiones, segmentar_nodulo  # noqa: E402
from tpi.validacion import dice  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "e07", Path(__file__).resolve().parents[1] / "scripts" / "07_baja_dosis.py")
E7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E7)

VENTANA = (-1000, 400)


def ver(ax, img, titulo):
    ax.imshow(np.clip(img, *VENTANA), cmap="gray", vmin=VENTANA[0], vmax=VENTANA[1])
    ax.set_title(titulo, fontsize=11)
    ax.axis("off")


def contorno(ax, m, color, lw=1.6, ls="-"):
    if m is not None and m.any():
        ax.contour(m.astype(float), levels=[0.5], colors=color, linewidths=lw, linestyles=ls)


def casos():
    car = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    car = car[car.grupo.isin(["benigno", "sospechoso"]) & (car.conjunto == "entrenamiento")]
    for _, f in car.iterrows():
        ruta = C.DIR_CORTES / f"{f.paciente}_s{int(f.scan_id)}_n{int(f.nodulo)}.npz"
        if ruta.exists():
            yield f, ruta


def figura_segmentacion(T, R):
    mejor = None
    for f, ruta in casos():
        d = cargar_corte(ruta)
        antes = crecimiento_regiones(d["img"], d["semilla"], umbral=T)
        despues = segmentar_nodulo(d, umbral=T, radio_cierre_mm=R)
        da, dd = dice(antes, d["consenso"]), dice(despues, d["consenso"])
        if mejor is None or dd - da > mejor[0]:
            mejor = (dd - da, ruta, d, antes, despues, da, dd)
    if mejor is None:
        print("No se encontraron cortes en", C.DIR_CORTES)
        return
    _, ruta, d, antes, despues, da, dd = mejor
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    ver(ax[0], d["img"], f"Antes: crecimiento libre (Dice {da:.2f})".replace(".", ","))
    ver(ax[1], d["img"], f"Después: dentro del pulmón cerrado (Dice {dd:.2f})".replace(".", ","))
    for a, m in zip(ax, (antes, despues)):
        contorno(a, d["consenso"], "#3CCB5A", 1.4, "--")
        contorno(a, m, "#E5572F", 1.8)
    fig.text(0.5, 0.02, "Rojo: algoritmo.  Verde punteado: consenso de los radiólogos.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salida = C.DIR_FIGURAS / "presentacion_segmentacion.png"
    fig.savefig(salida, dpi=200)
    plt.close(fig)
    print("->", salida, "|", ruta.stem, f"| Dice {da:.2f} -> {dd:.2f}")


def mascara_calcio_robusta(img, seg, umbral=C.HU_CALCIO, area_min_px=3):
    """Misma operación que tpi.caracteristicas.fraccion_calcio_robusta, devolviendo la máscara."""
    cal = (ndi.median_filter(img, size=3) > umbral) & seg
    lab = measure.label(cal, connectivity=2)
    tam = np.bincount(lab.ravel())
    tam[0] = 0
    return tam[lab] >= area_min_px


def figura_calcio(T, R):
    I0 = 1e4  # dosis x1/10, como en el 07
    elegido = None
    for f, ruta in casos():
        if not (f.frac_calcio <= C.FRAC_CALCIO_MAX and f.get("aceptable", True)):
            continue
        d = cargar_corte(ruta)
        semilla = zlib.crc32(ruta.stem.encode())  # el mismo ruido que en el 07
        d2 = E7.como_si_fuera_original(d, E7.degradar(d, "fbp", I0, semilla))
        seg = segmentar_nodulo(d2, umbral=T, radio_cierre_mm=R)
        if seg.sum() < 20:
            continue
        simple = (d2["img"] > C.HU_CALCIO) & seg
        robusta = mascara_calcio_robusta(d2["img"], seg)
        fs, fr = simple.sum() / seg.sum(), robusta.sum() / seg.sum()
        # caso típico de calcio falso: el umbral simple lo marca y el robusto no
        if fs > C.FRAC_CALCIO_MAX and fr <= C.FRAC_CALCIO_MAX:
            if elegido is None or seg.sum() > elegido[-1].sum():
                elegido = (ruta, d2, simple, robusta, fs, fr, seg)
    if elegido is None:
        print("No apareció un nódulo sin calcio con calcio falso a x1/10.")
        return
    ruta, d2, simple, robusta, fs, fr, seg = elegido
    r, c = np.nonzero(seg)
    m = 12
    sl = (slice(max(r.min() - m, 0), r.max() + m), slice(max(c.min() - m, 0), c.max() + m))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    ver(ax[0], d2["img"][sl], f"Umbral simple: {100 * fs:.0f} % del área".replace(".", ","))
    ver(ax[1], d2["img"][sl], f"Calcio robusto: {100 * fr:.0f} % del área".replace(".", ","))
    for a, cal in zip(ax, (simple, robusta)):
        contorno(a, seg[sl], "#0E8A8A", 1.4)
        yy, xx = np.nonzero(cal[sl])
        a.scatter(xx, yy, s=14, c="#E5572F", marker="s", linewidths=0)
    fig.text(0.5, 0.02, "Nódulo sin calcio a un décimo de la dosis. Naranja: píxeles contados como calcio.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salida = C.DIR_FIGURAS / "presentacion_calcio.png"
    fig.savefig(salida, dpi=200)
    plt.close(fig)
    print("->", salida, "|", ruta.stem, f"| simple {fs:.1%} -> robusto {fr:.1%}")


def main():
    T, R = E7.parametros()
    print(f"Umbral {T} HU, cierre {R:g} mm")
    figura_segmentacion(T, R)
    figura_calcio(T, R)


if __name__ == "__main__":
    main()
