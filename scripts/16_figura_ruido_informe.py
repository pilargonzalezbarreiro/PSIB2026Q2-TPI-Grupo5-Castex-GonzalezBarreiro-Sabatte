"""Etapa 16 - Figura 5 del informe: el ruido de la baja dosis.

  a. AUC del puntaje del interior y del diámetro solo según el ruido agregado
     en el pulmón (HU), en los tres escenarios de ruido electrónico
     (σe² = 0, 6 y 34). Se grafica contra el ruido en HU y no contra la
     fracción de dosis porque así las curvas de los tres escenarios coinciden:
     la conclusión no depende del valor de σe².
  b. Calcio falso por el ruido (σe² = 6): nódulos sin calcio que parecen
     calcificados con el umbral simple y con el calcio robusto del 08, y
     calcificados que el calcio robusto sigue detectando.

Uso:    python scripts/16_figura_ruido_informe.py
Necesita (no necesita imágenes):
  resultados/07_segmentacion_* y 07_clasificacion_* con los sufijos
    anclaje_ruido, anclaje_hibrido_se6 y anclaje_hibrido_se33.7
    (script 07 con --anclaje ruido y --anclaje hibrido con cada σe²)
  resultados/08_calcio_robusto_anclaje_hibrido_se6.csv (script 08)
Salida: figuras/16_ruido_informe.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from tpi import config as C  # noqa: E402

R = C.DIR_RESULTADOS
ESCENARIOS = [("anclaje_ruido", "σe² = 0", ":", "s"),
              ("anclaje_hibrido_se6", "σe² = 6 (principal)", "-", "o"),
              ("anclaje_hibrido_se33.7", "σe² = 34", "--", "^")]
CONDICIONES = ["original", "dosis ×1/2", "dosis ×1/4", "dosis ×1/10", "dosis ×1/20"]


def _tabla(nombre):
    return pd.read_csv(R / nombre).set_index("condicion").loc[CONDICIONES]


def panel_auc(ax):
    for suf, _, ls, mk in ESCENARIOS:
        g = _tabla(f"07_segmentacion_{suf}.csv")
        c = pd.read_csv(R / f"07_clasificacion_{suf}.csv")
        for puntaje, color in [("interior", "C3"), ("diámetro solo", "C0")]:
            y = c[c.puntaje == puntaje].set_index("condicion").loc[CONDICIONES, "auc_transferido"]
            ax.plot(g.ruido_agregado_hu, y, ls=ls, marker=mk, ms=4, color=color, lw=1.3)
    ax.axhline(0.5, color="k", lw=0.6)
    ax.set_xlabel("ruido agregado en el pulmón (HU)")
    ax.set_ylabel("AUC (puntaje transferido)")
    ax.set_ylim(0.65, 0.97)
    ax.set_title("a. Clasificación según el ruido", fontsize=9.5)
    h1 = [Line2D([], [], color="C3", lw=1.5, label="puntaje del interior"),
          Line2D([], [], color="C0", lw=1.5, label="diámetro solo")]
    h2 = [Line2D([], [], color="0.3", ls=ls, marker=mk, ms=4, label=nom) for _, nom, ls, mk in ESCENARIOS]
    ax.add_artist(ax.legend(handles=h1, fontsize=6.5, loc="upper right", frameon=False))
    ax.legend(handles=h2, fontsize=6.5, loc="center right", bbox_to_anchor=(1, 0.62), frameon=False)


def panel_calcio(ax):
    k = _tabla("08_calcio_robusto_anclaje_hibrido_se6.csv")
    x = _tabla("07_segmentacion_anclaje_hibrido_se6.csv").ruido_agregado_hu
    simple = k["sin calcio: % que parece calcificado (umbral simple)"]
    ax.plot(x, simple, "-o", ms=4, color="C3", label="sin calcio que parecen calcificados (umbral)")
    ax.plot(x, k["sin calcio: % que parece calcificado (robusto)"], "-o", ms=4, color="C0",
            label="sin calcio que parecen calcificados (robusto)")
    ax.plot(x, k["calcificados: % detectado (robusto)"], "--s", ms=4, color="C0", alpha=0.6,
            label="calcificados detectados (robusto)")
    for xi, yi, et in zip(x, simple, ["", "1/2", "1/4", "1/10", "1/20"]):
        ax.annotate(et, (xi, yi + 4), ha="center", fontsize=7, color="0.35")
    ax.set_ylim(-3, 103)
    ax.set_xlabel("ruido agregado en el pulmón (HU)")
    ax.set_ylabel("% de nódulos")
    ax.set_title("b. Calcio falso por el ruido (σe² = 6)", fontsize=9.5)
    ax.legend(fontsize=6.3, loc="center left", bbox_to_anchor=(0, 0.66), frameon=False)


def main():
    plt.rcParams.update({"font.size": 9})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    panel_auc(a)
    panel_calcio(b)
    fig.tight_layout()
    C.DIR_FIGURAS.mkdir(exist_ok=True)
    ruta = C.DIR_FIGURAS / "16_ruido_informe.png"
    fig.savefig(ruta, dpi=300)
    print(f"Figura guardada en {ruta}")


if __name__ == "__main__":
    main()
