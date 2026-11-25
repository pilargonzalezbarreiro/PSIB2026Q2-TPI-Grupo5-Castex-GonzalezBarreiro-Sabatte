"""¿Cuánto error mete elegir mal la constante del tubo (Φ) y el bowtie?

En LIDC no se puede medir ni Φ ni el perfil del bowtie de cada equipo (ver
DECISIONES §6), así que la simulación usa valores genéricos. Este script mide
cuánto cambia el resultado si esos valores están mal.

1. Φ. Si se usa k veces el Φ verdadero, el código cree que la imagen tenía k
   veces más fotones y agrega 1/k de la varianza que correspondía. La fracción
   de dosis que se simula de verdad es
       f_real = 1 / (1 + (1/f - 1) / k)
   Se calcula esa fórmula y se la VERIFICA simulando con Φ·k.

2. Bowtie. Su efecto no tiene fórmula: cambia cómo se reparte el ruido en la
   imagen. Se prueban muchos bowties (plano, campana "gauss" y el modelo físico
   "cilindro" con distintos diámetros, compensaciones y pisos) y, para cada
   uno, se calibran los fotones con el ruido del tejido blando, IGUAL que hace
   el pipeline. Lo que se mide es el error que QUEDA después de calibrar: el
   ruido agregado en el pulmón y cerca del nódulo, comparado con el bowtie por
   defecto (en el fantoma, comparado con el verdadero, que sí se conoce).

3. Estimar el bowtie desde la imagen. Con más o menos fotones en los rayos
   laterales, el ruido cambia distinto en el centro y en la periferia del
   cuerpo. Se mide el ruido del tejido blando en bandas según la distancia al
   centro del cuerpo y se busca el bowtie que mejor reproduce ese perfil. Es
   una estimación aproximada (la modulación de corriente, la anatomía y el
   kernel también cambian ese perfil); en el fantoma se puede comprobar si
   recupera el bowtie verdadero.

Uso:
  python comparaciones/sensibilidad_dosis.py                 # fantoma (verdad conocida)
  python comparaciones/sensibilidad_dosis.py --carpeta datos/cortes --casos 3
Salidas: resultados/comparaciones/sensibilidad_{phi,bowtie,bowtie_estimado}.csv
         figuras/comparaciones/sensibilidad_dosis.png
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tests"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from tpi import baja_dosis as B  # noqa: E402
from tpi import config as C  # noqa: E402

FRACCIONES = (0.5, 0.25, 0.1)
KS_PHI = (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0)
BANDAS_MM = ((0, 60), (60, 100), (100, 140), (140, 200))
N_FANTOMA = 5e5


def candidatos():
    """Bowties a probar: nombre -> parámetros de perfil_bowtie."""
    c = {"plano": dict(modelo="plano")}
    for b_min in (0.05, 0.15, 0.3, 0.5):
        for ancho in (100, 160, 250):
            c[f"gauss piso={b_min} ancho={ancho}"] = dict(modelo="gauss", b_min=b_min, ancho_mm=ancho)
    for a in (0.25, 0.5, 0.75, 1.0):
        for D in (240, 320, 400):
            for b_min in (0.02, 0.05, 0.1):
                c[f"cilindro α={a} D={D} piso={b_min}"] = dict(modelo="cilindro", compensacion=a,
                                                                diametro_mm=D, b_min=b_min)
    return c


DEFECTO = "cilindro α=0.5 D=320 piso=0.05"


# ---------------------------------------------------------------------------
# Casos: fantoma (verdad conocida) o cortes reales
# ---------------------------------------------------------------------------
def casos_fantoma():
    from test_baja_dosis import N, PX, fantoma_limpio
    verdad = fantoma_limpio()
    rr, cc = np.mgrid[:N, :N]
    pulmon_izq = ((rr - 250) / 140) ** 2 + ((cc - 165) / 65) ** 2 <= 1
    pulmon = pulmon_izq | (((rr - 250) / 140) ** 2 + ((cc - 347) / 65) ** 2 <= 1)
    d2 = (rr - 250) ** 2 + (cc - 165) ** 2
    nodulo = (d2 <= (15 / PX) ** 2) & (d2 > 14 ** 2) & pulmon_izq   # pulmón alrededor del nódulo
    adq = B.Adquisicion(verdad, PX, bowtie=candidatos()[DEFECTO])
    adq.N_alta, adq.ventana = N_FANTOMA, "shepp-logan"
    r0 = adq.ruido_original(np.random.default_rng(7))
    yield dict(nombre="fantoma", img=verdad + r0, px=PX, pulmon=pulmon, nodulo=nodulo,
               ventana="shepp-logan", verdad_bowtie=DEFECTO, sigma0_pulmon=float(r0[pulmon].std()))


def casos_reales(carpeta, n):
    from tpi.datos import cargar_corte
    from tpi.encabezados import familia_kernel
    from tpi.entorno import mascara_pulmon
    enc = {}
    ruta_enc = C.DIR_RESULTADOS / "encabezados_dicom.csv"
    if ruta_enc.exists():
        t = pd.read_csv(ruta_enc)
        enc = {f"{r.paciente}_s{r.scan_id}_n{r.nodulo}": r.kernel for _, r in t.iterrows()}
    for ruta in sorted(Path(carpeta).glob("*.npz"))[:n]:
        d = cargar_corte(ruta)
        px = float(d["pixel_mm_original"])
        img = np.maximum(d["corte_completo"], -1000).astype(np.float32)
        pulmon = mascara_pulmon(img, pixel_mm=px)
        sr, sc = np.asarray(d["semilla_completo"], float) * C.PIXEL_MM / px
        rr, cc = np.mgrid[:img.shape[0], :img.shape[1]]
        nodulo = (rr - sr) ** 2 + (cc - sc) ** 2 <= (15 / px) ** 2
        yield dict(nombre=ruta.stem, img=img, px=px, pulmon=pulmon, nodulo=nodulo,
                   ventana=familia_kernel(enc.get(ruta.stem))[1], verdad_bowtie=None, sigma0_pulmon=None)


# ---------------------------------------------------------------------------
def perfil_radial(ruido, tejido, dist):
    """Ruido local (mediana del desvío en ventana de ~6 mm) del tejido blando
    en bandas de distancia al centro del cuerpo, normalizado a su media."""
    k = B._lado(C.PIXEL_MM)
    x = np.asarray(ruido, float)
    m1 = ndi.uniform_filter(x, k)
    sd = np.sqrt(np.maximum(ndi.uniform_filter(x * x, k) - m1 * m1, 0))
    v = np.array([np.median(sd[m]) if m.sum() > 300 else np.nan
                  for m in (tejido & (dist >= a) & (dist < b) for a, b in BANDAS_MM)])
    return v / np.nanmean(v)


def analizar_caso(caso, rng_base=0):
    img, px = caso["img"], caso["px"]
    adq = B.Adquisicion(img, px)      # el sinograma virtual se calcula una sola vez
    adq.ventana = caso["ventana"]
    tejido = B.mascara_tejido_blando(adq.corte, px)
    s0 = B.ruido_global(adq.corte, px, tejido)
    rr, cc = np.mgrid[:img.shape[0], :img.shape[1]]
    cy, cx = np.argwhere(adq.corte > -500).mean(0)
    dist = np.hypot(rr - cy, cc - cx) * px
    obs = perfil_radial(adq.corte - ndi.median_filter(adq.corte, 3), tejido, dist)  # sin anatomía gruesa

    filas, filas_est = [], []
    for nombre, par in candidatos().items():
        adq.bowtie = B.perfil_bowtie(adq.p.shape[0], px, **par)
        adq.desde_ruido(s0, rng=np.random.default_rng(rng_base))    # calibrar como el pipeline
        r0 = adq.ruido_original(np.random.default_rng(rng_base + 1))
        fila = dict(caso=caso["nombre"], bowtie=nombre, modelo=par["modelo"],
                    sigma0_pulmon_modelo=float(r0[caso["pulmon"]].std()))
        for f in FRACCIONES:
            _, r = adq.a_fraccion(f, np.random.default_rng(rng_base + 2))
            fila[f"agregado_pulmon_f{f}"] = float(r[caso["pulmon"]].std())
            fila[f"agregado_nodulo_f{f}"] = float(r[caso["nodulo"] & caso["pulmon"]].std())
        filas.append(fila)
        sim = perfil_radial(r0 - ndi.median_filter(r0, 3), tejido, dist)  # mismo paso que a la imagen
        filas_est.append(dict(caso=caso["nombre"], bowtie=nombre, error_perfil=float(np.nansum((sim - obs) ** 2))))
    return pd.DataFrame(filas), pd.DataFrame(filas_est), s0


def fraccion_real(caso, df, ref_bowtie):
    """Con cada bowtie, ¿qué fracción de dosis se simula de verdad en el
    pulmón? Referencia: el bowtie verdadero (fantoma) o el por defecto (real)."""
    ref = df.set_index("bowtie").loc[ref_bowtie]
    s0 = caso["sigma0_pulmon"] or ref["sigma0_pulmon_modelo"]
    for f in FRACCIONES:
        df[f"f_real_{f}"] = s0 ** 2 / (s0 ** 2 + df[f"agregado_pulmon_f{f}"] ** 2)
        df[f"rel_nodulo_f{f}"] = df[f"agregado_nodulo_f{f}"] / ref[f"agregado_nodulo_f{f}"] - 1
    return df


def sensibilidad_phi(caso):
    """Fórmula y verificación simulando con Φ·k (bowtie por defecto)."""
    adq = B.Adquisicion(caso["img"], caso["px"], bowtie=candidatos()[DEFECTO])
    adq.ventana = caso["ventana"]
    tejido = B.mascara_tejido_blando(adq.corte, caso["px"])
    adq.desde_ruido(B.ruido_global(adq.corte, caso["px"], tejido), rng=np.random.default_rng(0))
    N_ok = adq.N_alta
    s0 = caso["sigma0_pulmon"] or float(adq.ruido_original(np.random.default_rng(1))[caso["pulmon"]].std())
    filas = []
    for k in KS_PHI:
        adq.N_alta = N_ok * k
        for f in FRACCIONES:
            _, r = adq.a_fraccion(f, np.random.default_rng(3))
            sa = float(r[caso["pulmon"]].std())
            filas.append(dict(caso=caso["nombre"], k=k, f=f, f_real_formula=1 / (1 + (1 / f - 1) / k),
                              f_real_simulado=s0 ** 2 / (s0 ** 2 + sa ** 2)))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=None, help="cortes reales (sin esto: fantoma)")
    ap.add_argument("--casos", type=int, default=3)
    args = ap.parse_args()
    casos = list(casos_reales(args.carpeta, args.casos) if args.carpeta else casos_fantoma())
    dir_res = C.DIR_RESULTADOS / "comparaciones"
    dir_res.mkdir(parents=True, exist_ok=True)
    todos, todos_est, todos_phi = [], [], []
    for caso in casos:
        print(f"{caso['nombre']}: {len(candidatos())} bowties...", flush=True)
        df, est, _ = analizar_caso(caso)
        todos.append(fraccion_real(caso, df, caso["verdad_bowtie"] or DEFECTO))
        todos_est.append(est)
        todos_phi.append(sensibilidad_phi(caso))
    bt, est, phi = pd.concat(todos), pd.concat(todos_est), pd.concat(todos_phi)
    bt.to_csv(dir_res / "sensibilidad_bowtie.csv", index=False)
    est.to_csv(dir_res / "sensibilidad_bowtie_estimado.csv", index=False)
    phi.to_csv(dir_res / "sensibilidad_phi.csv", index=False)

    print("\n== 1. Φ mal elegido (×k): fracción que se simula de verdad cuando la etiqueta dice f ==")
    t = phi.groupby(["f", "k"])[["f_real_formula", "f_real_simulado"]].mean().unstack("f")
    print((1 / t).round(1).rename(columns=lambda c: c).to_string(header=True)
          .replace("f_real_formula", "1/f real (fórmula)").replace("f_real_simulado", "1/f real (simulado)"))

    print(f"\n== 2. Bowtie: ruido agregado cerca del nódulo a ×1/4, relativo a {'el verdadero' if not args.carpeta else 'el por defecto'} ==")
    r = bt.groupby("modelo")["rel_nodulo_f0.25"].describe(percentiles=[.1, .5, .9])
    print((100 * r[["min", "10%", "50%", "90%", "max"]]).round(0).to_string())
    g = bt["f_real_0.25"]
    print(f"   la etiqueta ×1/4 corresponde, según el bowtie, a entre 1/{1 / g.max():.1f} y 1/{1 / g.min():.1f} "
          f"(P10-P90: 1/{1 / g.quantile(.9):.1f} a 1/{1 / g.quantile(.1):.1f})")

    print("\n== 3. Bowtie estimado desde el perfil radial del ruido (los 5 que mejor ajustan) ==")
    for c, sub in est.groupby("caso"):
        mejor = sub.nsmallest(5, "error_perfil")
        print(f"   {c}: " + " | ".join(mejor.bowtie))
        if not args.carpeta:
            pos = int((sub.sort_values("error_perfil").bowtie.reset_index(drop=True) == DEFECTO).idxmax()) + 1
            print(f"      el verdadero ({DEFECTO}) quedó en el puesto {pos} de {len(sub)}")
    figura(bt, phi, C.DIR_FIGURAS / "comparaciones" / "sensibilidad_dosis.png")


def figura(bt, phi, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for f, m in zip(FRACCIONES, "os^"):
        sub = phi[phi.f == f].groupby("k").mean(numeric_only=True)
        ax[0].plot(sub.index, 1 / sub.f_real_formula, "-", label=f"fórmula, etiqueta 1/{1 / f:g}")
        ax[0].plot(sub.index, 1 / sub.f_real_simulado, m, color=ax[0].lines[-1].get_color())
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Φ usado / Φ verdadero")
    ax[0].set_ylabel("1 / (fracción de dosis simulada de verdad)")
    ax[0].set_title("a. Error de Φ (líneas: fórmula; puntos: simulado)")
    ax[0].legend(fontsize=7)
    orden = ["plano", "gauss", "cilindro"]
    datos = [100 * bt.loc[bt.modelo == m, "rel_nodulo_f0.25"].to_numpy() for m in orden]
    ax[1].boxplot(datos, tick_labels=orden)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("ruido agregado cerca del nódulo (% vs referencia)")
    ax[1].set_title("b. Error del bowtie después de calibrar (×1/4)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("->", ruta)


if __name__ == "__main__":
    main()
