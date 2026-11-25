"""Etapa 15 - Las tablas y la figura del calcio que van al informe.

Responde a la devolución sobre el calcio (conclusión central de H1), sin
descartar nódulos y con significancias:
  1. El calcio solo: cuántos benignos y sospechosos lo tienen, qué tan
     específico y sensible es, y su AUC (presencia y fracción).
  2. Boxplot de tres grupos (benignos con calcio, benignos sin calcio y
     sospechosos) para el contraste GLCM y la asimetría de las HU, con
     Mann-Whitney y p ajustado por Holm (4 comparaciones).
  3. Tabla A5: AUC del diámetro por estrato de calcio y de tamaño (8-16 mm).
  4. Tabla A6: Spearman parcial de cada medida con el grupo controlando por
     tamaño, por calcio y por los dos a la vez (el calcio como covariable,
     una de las alternativas que propone la devolución).
  5. Tabla A7: sensibilidad al umbral de fracción de calcio (0 a 10 %).
     El umbral de 200 HU se varía en el script 09 (necesita las imágenes).

Población: nódulos de desarrollo con segmentación aceptable (112), igual que
el 05 y el 06. "Con calcio" = más del 2 % del área por encima de 200 HU.

Uso:    python scripts/15_calcio_informe.py
Necesita: resultados/caracteristicas.csv (script 04). No necesita imágenes.
Salidas: resultados/15_calcio_resumen.csv, 15_boxplot_calcio_tests.csv,
         15_tabla_A5_auc_diametro_estratos.csv, 15_tabla_A6_spearman_parcial.csv,
         15_tabla_A7_umbral_fraccion.csv y figuras/15_boxplot_calcio.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.estadistica import holm  # noqa: E402
from tpi.puntaje import auc  # noqa: E402

MEDIDAS_A6 = ["hu_asimetria", "glcm_contraste", "hu_desvio", "nitidez_borde_hu_mm", "solidez",
              "glcm_correlacion", "glcm_cluster_prominence", "circularidad"]
NOMBRES = {"hu_asimetria": "Asimetría de las HU", "glcm_contraste": "Contraste GLCM",
           "hu_desvio": "Desvío de las HU", "nitidez_borde_hu_mm": "Nitidez del borde",
           "solidez": "Solidez", "glcm_correlacion": "Correlación GLCM",
           "glcm_cluster_prominence": "Cluster prominence", "circularidad": "Circularidad"}


def poblacion():
    df = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    df = df[(df.conjunto == "entrenamiento") & df.aceptable & df.grupo.isin(["benigno", "sospechoso"])]
    df = df.reset_index(drop=True)
    df["sosp"] = (df.grupo == "sospechoso").astype(int)
    df["con_calcio"] = df.frac_calcio > C.FRAC_CALCIO_MAX
    return df


def auc_orientado(y, x):
    a = auc(y, np.asarray(x, float))
    return max(a, 1 - a)


def spearman_parcial_multi(x, y, covariables):
    """Correlación de Spearman entre x e y descontando VARIAS covariables:
    rangos de todo, se le saca a x y a y lo que explican (regresión lineal) los
    rangos de las covariables, y se correlacionan los residuos. p con una t de
    Student con n - 2 - k grados de libertad (k = número de covariables)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z = np.column_stack([np.ones(len(rx))] + [stats.rankdata(z) for z in covariables])

    def residuo(v):
        return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]

    r = float(np.corrcoef(residuo(rx), residuo(ry))[0, 1])
    gl = len(rx) - 2 - len(covariables)
    t = r * np.sqrt(gl / max(1 - r ** 2, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), gl))


def resumen_calcio(df):
    b, s = df[df.sosp == 0], df[df.sosp == 1]
    con = df[df.con_calcio]
    filas = [
        ("benignos con calcio", f"{int(b.con_calcio.sum())} de {len(b)} ({100 * b.con_calcio.mean():.0f} %)"),
        ("sospechosos con calcio", f"{int(s.con_calcio.sum())} de {len(s)} ({100 * s.con_calcio.mean():.0f} %)"),
        ("si tiene calcio, es benigno (valor predictivo; depende de la muestra 50/50)",
         f"{int((con.sosp == 0).sum())} de {len(con)} ({100 * (con.sosp == 0).mean():.0f} %)"),
        ("especificidad (sospechosos SIN calcio)", f"{100 * (1 - s.con_calcio.mean()):.0f} %"),
        ("sensibilidad (benignos CON calcio)", f"{100 * b.con_calcio.mean():.0f} %"),
        ("AUC del calcio, presencia (sí/no)", f"{auc_orientado(df.sosp, df.con_calcio.astype(float)):.3f}"),
        ("AUC del calcio, fracción del área", f"{auc_orientado(df.sosp, df.frac_calcio):.3f}"),
        ("diámetro mediano, benignos con / sin calcio",
         f"{b[b.con_calcio].diametro_eq_mm.median():.1f} / {b[~b.con_calcio].diametro_eq_mm.median():.1f} mm"),
    ]
    t = pd.DataFrame(filas, columns=["qué", "valor"])
    t.to_csv(C.DIR_RESULTADOS / "15_calcio_resumen.csv", index=False)
    print("== 1. El calcio solo (112 nódulos de desarrollo, segmentación aceptable) ==")
    print(t.to_string(index=False))


def boxplot_tres_grupos(df):
    grupos = {"benignos\ncon calcio": df[(df.sosp == 0) & df.con_calcio],
              "benignos\nsin calcio": df[(df.sosp == 0) & ~df.con_calcio],
              "sospechosos": df[df.sosp == 1]}
    medidas = ["glcm_contraste", "hu_asimetria"]
    comparaciones = [("benignos\ncon calcio", "benignos\nsin calcio"), ("benignos\nsin calcio", "sospechosos")]
    filas = []
    for m in medidas:
        for g1, g2 in comparaciones:
            p = stats.mannwhitneyu(grupos[g1][m].dropna(), grupos[g2][m].dropna()).pvalue
            filas.append(dict(medida=m, grupo_1=g1.replace("\n", " "), grupo_2=g2.replace("\n", " "),
                              mediana_1=grupos[g1][m].median(), mediana_2=grupos[g2][m].median(), p=p))
    t = pd.DataFrame(filas)
    t["p_holm"] = holm(t.p)
    t.to_csv(C.DIR_RESULTADOS / "15_boxplot_calcio_tests.csv", index=False)
    print("\n== 2. Boxplot de tres grupos: Mann-Whitney con Holm (4 comparaciones) ==")
    print(t.round(4).to_string(index=False))

    def texto_p(p):
        return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",") + ("" if p < 0.05 else " (ns)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    colores = ["#4a7ab5", "#9cc3e6", "#c44e52"]
    for a, m in zip(ax, medidas):
        datos = [g[m].dropna().to_numpy() for g in grupos.values()]
        bp = a.boxplot(datos, tick_labels=[f"{k}\n(n = {len(v)})" for k, v in grupos.items()],
                       widths=0.55, patch_artist=True, showfliers=False)
        for caja, col in zip(bp["boxes"], colores):
            caja.set_facecolor(col)
            caja.set_alpha(0.55)
        rng = np.random.default_rng(0)
        for i, v in enumerate(datos):
            a.plot(i + 1 + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=3, color="k", alpha=0.45)
        tope = max(np.nanmax(v) for v in datos)
        rango = tope - min(np.nanmin(v) for v in datos)
        sub = t[t.medida == m].reset_index(drop=True)
        for k, (x1, x2) in enumerate([(1, 2), (2, 3)]):
            y = tope + rango * (0.08 + 0.12 * k)
            a.plot([x1, x1, x2, x2], [y - rango * 0.02, y, y, y - rango * 0.02], "k", lw=0.8)
            a.text((x1 + x2) / 2, y + rango * 0.01, texto_p(sub.p_holm[k]), ha="center", fontsize=9)
        a.set_ylim(top=tope + rango * 0.38)
        a.set_title(NOMBRES[m], fontsize=11)
        a.tick_params(labelsize=9)
    fig.suptitle("Benignos con calcio, benignos sin calcio y sospechosos (desarrollo, segmentación aceptable)\n"
                 "Mann-Whitney, p ajustado por Holm (4 comparaciones)", fontsize=10)
    fig.tight_layout()
    C.DIR_FIGURAS.mkdir(exist_ok=True)
    ruta = C.DIR_FIGURAS / "15_boxplot_calcio.png"
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print("->", ruta)


def tabla_a5(df):
    filas = []
    rango = df[(df.diametro_medio_mm >= 8) & (df.diametro_medio_mm < 16)]
    for etiqueta, sub in (("todos", df), ("sin calcio (≤ 2 %)", df[~df.con_calcio]),
                          ("con calcio (> 2 %)", df[df.con_calcio]),
                          ("8 a 16 mm, todos", rango), ("8 a 16 mm, sin calcio", rango[~rango.con_calcio]),
                          ("8 a 16 mm, con calcio", rango[rango.con_calcio])):
        filas.append(dict(estrato=etiqueta, nodulos=len(sub), benignos=int((sub.sosp == 0).sum()),
                          sospechosos=int(sub.sosp.sum()),
                          auc_diametro=auc_orientado(sub.sosp, sub.diametro_eq_mm)
                          if sub.sosp.nunique() == 2 else np.nan))
    t = pd.DataFrame(filas)
    t.to_csv(C.DIR_RESULTADOS / "15_tabla_A5_auc_diametro_estratos.csv", index=False)
    print("\n== 3. Tabla A5: AUC del diámetro equivalente por estrato (8-16 mm: diámetro de los radiólogos) ==")
    print(t.round(3).to_string(index=False))
    print("   Los estratos con calcio tienen muy pocos sospechosos: su AUC es muy incierto.")


def tabla_a6(df):
    filas = []
    for m in MEDIDAS_A6:
        ok = df[[m, "sosp", "diametro_eq_mm", "frac_calcio"]].dropna()
        r_t, p_t = spearman_parcial_multi(ok[m], ok.sosp, [ok.diametro_eq_mm])
        r_c, p_c = spearman_parcial_multi(ok[m], ok.sosp, [ok.frac_calcio])
        r_a, p_a = spearman_parcial_multi(ok[m], ok.sosp, [ok.diametro_eq_mm, ok.frac_calcio])
        filas.append(dict(medida=NOMBRES[m], n=len(ok), controlando_tamano=r_t, p_tamano=p_t,
                          controlando_calcio=r_c, p_calcio=p_c, controlando_ambos=r_a, p_ambos=p_a))
    t = pd.DataFrame(filas)
    t["p_ambos_bonferroni"] = np.minimum(t.p_ambos * len(t), 1.0)
    t["p_ambos_holm"] = holm(t.p_ambos)
    t.to_csv(C.DIR_RESULTADOS / "15_tabla_A6_spearman_parcial.csv", index=False)
    print("\n== 4. Tabla A6: Spearman parcial de cada medida con el grupo (signo +: mayor en sospechosos) ==")
    print(t.round(3).to_string(index=False))
    print("   Exploratorio (8 medidas): se reporta el p corregido por Bonferroni y por Holm.")


def tabla_a7(df):
    filas = []
    for umbral in (0.0, 0.01, 0.02, 0.05, 0.10):
        sin = df[df.frac_calcio <= umbral]
        filas.append(dict(umbral_fraccion=f"{100 * umbral:g} %", nodulos_sin_calcio=len(sin),
                          benignos=int((sin.sosp == 0).sum()), sospechosos=int(sin.sosp.sum()),
                          auc_diametro=auc_orientado(sin.sosp, sin.diametro_eq_mm)))
    t = pd.DataFrame(filas)
    t.to_csv(C.DIR_RESULTADOS / "15_tabla_A7_umbral_fraccion.csv", index=False)
    print("\n== 5. Tabla A7: sensibilidad al umbral de fracción de calcio (200 HU fijo; el 09 varía los HU) ==")
    print(t.round(3).to_string(index=False))


def main():
    df = poblacion()
    resumen_calcio(df)
    boxplot_tres_grupos(df)
    tabla_a5(df)
    tabla_a6(df)
    tabla_a7(df)


if __name__ == "__main__":
    main()
