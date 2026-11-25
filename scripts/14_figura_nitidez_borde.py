"""Figura para EXPLICAR la nitidez del borde (defensa oral e informe).

Para un nódulo muestra, de izquierda a derecha:
  1. el corte con el contorno de la segmentación;
  2. la banda de ~3 píxeles sobre el contorno donde se mide;
  3. el módulo del gradiente de Sobel (HU/mm) y el valor medio dentro de la banda.
Hace lo mismo que tpi.caracteristicas.nitidez_borde (mismo cálculo, paso a paso).

Sin argumentos toma el nódulo con borde MÁS nítido y el MÁS difuso de resultados/caracteristicas.csv.
Salida: figuras/14_nitidez_borde.png
Uso:    python scripts/14_figura_nitidez_borde.py [--carpeta datos/cortes] [--nombres A B]
        (--nombres: dos nombres de .npz sin extensión, p. ej. LIDC-IDRI-0004_s15_n0)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _limitaciones_comun import C, parametros_guardados
from scipy import ndimage as ndi

from tpi.caracteristicas import nitidez_borde
from tpi.datos import cargar_corte
from tpi.segmentacion import segmentar_nodulo


def pasos(img_hu, mascara, pixel_mm=C.PIXEL_MM):
    """Los mismos pasos que nitidez_borde, devolviendo cada paso para dibujarlo."""
    borde = ndi.binary_dilation(mascara & ~ndi.binary_erosion(mascara))
    gx, gy = ndi.sobel(img_hu, 0), ndi.sobel(img_hu, 1)
    g = np.hypot(gx, gy) / (8 * pixel_mm)  # HU/mm
    return borde, g, float(g[borde].mean())


def dibujar(ejes, nombre, img, seg):
    borde, g, valor = pasos(img, seg)
    ys, xs = np.where(seg)
    r = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.2 + 12)
    cy, cx = int(ys.mean()), int(xs.mean())
    sl = (slice(max(cy - r, 0), cy + r), slice(max(cx - r, 0), cx + r))
    ejes[0].imshow(img[sl], cmap="gray", vmin=-1000, vmax=400)
    ejes[0].contour(seg[sl], [0.5], colors="r", linewidths=1)
    ejes[0].set_title(f"{nombre}\n1. contorno de la segmentación", fontsize=8)
    ejes[1].imshow(img[sl], cmap="gray", vmin=-1000, vmax=400)
    ov = np.ma.masked_where(~borde[sl], borde[sl])
    ejes[1].imshow(ov, cmap="autumn", alpha=0.8)
    ejes[1].set_title("2. banda de ~3 px sobre el contorno", fontsize=8)
    im = ejes[2].imshow(g[sl], cmap="viridis", vmin=0, vmax=np.percentile(g[sl], 99))
    ejes[2].contour(borde[sl], [0.5], colors="w", linewidths=0.6)
    ejes[2].set_title(f"3. gradiente Sobel (HU/mm)\nmedia en la banda = {valor:.0f} HU/mm", fontsize=8)
    for e in ejes:
        e.axis("off")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    ap.add_argument("--nombres", nargs=2)
    a = ap.parse_args()
    df = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    df = df[df.aceptable].dropna(subset=["nitidez_borde_hu_mm"])
    if a.nombres:
        nombres = a.nombres
    else:
        f = [df.loc[df.nitidez_borde_hu_mm.idxmax()], df.loc[df.nitidez_borde_hu_mm.idxmin()]]
        nombres = [f"{x.paciente}_s{int(x.scan_id)}_n{int(x.nodulo)}" for x in f]
    T, R = parametros_guardados()
    C.DIR_FIGURAS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(10, 6.6))
    for fila, nombre in zip(ax, nombres):
        d = cargar_corte(Path(a.carpeta) / f"{nombre}.npz")
        seg = segmentar_nodulo(d, umbral=T, radio_cierre_mm=R).astype(bool)
        dibujar(fila, nombre, d["img"], seg)
        assert abs(pasos(d["img"], seg)[2] - nitidez_borde(d["img"], seg)) < 1e-6
    fig.tight_layout()
    ruta = C.DIR_FIGURAS / "14_nitidez_borde.png"
    fig.savefig(ruta, dpi=200)
    print("->", ruta, "|", nombres)


if __name__ == "__main__":
    main()
