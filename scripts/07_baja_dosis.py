"""Etapa 7 - ¿Los biomarcadores sobreviven a la BAJA DOSIS y a la compresión?

Pregunta (la parte innovadora del trabajo):
El tamizaje de cáncer de pulmón se hace con tomografía de BAJA DOSIS (para
irradiar poco a gente sana). Muchos trabajos de radiómica arman sus modelos
con tomografías de dosis normal, como LIDC. Si una medida cambia cuando baja
la dosis, un modelo armado con dosis normal se equivoca en el tamizaje
aunque en el paper haya dado muy bien. Lo mismo pasa si la imagen se
comprime para mandarla por telemedicina.

Cómo se responde, con la materia (reconstrucción, ruido y compresión):
  1. Cada corte real se lleva a una FRACCIÓN de la dosis de su propio
     estudio con el método de Naziroglu et al. (2017) (tpi/baja_dosis.py):
     sinograma virtual (Radon) -> fotones de la adquisición original anclados
     al mAs, kVp y espesor del DICOM (constante del tubo calibrada en el 07a)
     y bowtie -> ruido de Poisson + electrónico COMPENSADOS (solo el que le
     falta a la imagen, que ya trae el de su dosis) -> FBP SOLO del ruido con
     la ventana del kernel del equipo -> se suma a la original. "×1" no
     agrega nada y la imagen no se re-reconstruye.
     Si un estudio no tiene mAs, se ancla con el ruido medido en la imagen.
     Con --objetivo-mas se simula además un mAs absoluto (p. ej. 25 mAs,
     como un tamizaje), que para cada estudio es otra fracción.
     También se comprime con JPEG (calidad 50 y 25).
  2. Sobre CADA versión se corre el pipeline completo, igual que en la
     clínica: pulmón, crecimiento de regiones desde la semilla, validación
     contra los radiólogos y todas las medidas.
  3. Se mide:
     a. segmentación: ¿sigue siendo aceptable con menos dosis?
     b. reproducibilidad de cada medida: CCC de Lin entre el valor en la
        imagen original y en la degradada (1 = idéntico; a diferencia de la
        correlación, penaliza también que todo se corra para arriba o abajo);
     c. clasificación, de dos maneras:
        - re-armado: el puntaje se arma y evalúa con imágenes de la misma
          dosis (lo mejor que se puede hacer con esa dosis);
        - TRANSFERIDO: el puntaje se arma con las imágenes originales y se
          aplica a las degradadas (lo que pasa en la vida real cuando un
          modelo publicado se usa en un tamizaje). Validación cruzada por
          paciente en los dos casos.

Uso:
  python scripts/07_baja_dosis.py                  # simula (lento) y analiza
  python scripts/07_baja_dosis.py --solo-analisis  # reusa resultados/07_baja_dosis.csv
  opciones: --procesos N, --limite N (para probar), --todos, --sin-calcio,
            --objetivo-mas 40 25, --sin-electronico, --anclaje mas|ruido|hibrido
  antes: scripts/03b_encabezados_dicom.py y scripts/07a_calibrar_dosis.py

Salidas:
  resultados/07_baja_dosis.csv          medidas de cada nódulo en cada condición
  resultados/07_segmentacion.csv        ruido, Dice y % aceptables por condición
  resultados/07_reproducibilidad.csv    CCC de cada medida en cada condición
  resultados/07_clasificacion.csv       AUC re-armado y transferido por condición
  figuras/07_baja_dosis.png
"""
import argparse
import os
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import baja_dosis as B  # noqa: E402
from tpi import compresion as Z  # noqa: E402
from tpi import config as C  # noqa: E402
from tpi.caracteristicas import caracteristicas_interior, fraccion_calcio_robusta  # noqa: E402
from tpi.datos import _recortar, _remuestrear, cargar_corte  # noqa: E402
from tpi.entorno import caracteristicas_entorno, mascara_pulmon  # noqa: E402
from tpi.encabezados import familia_kernel, mas_efectivo  # noqa: E402
from tpi.estadistica import META  # noqa: E402
from tpi.puntaje import Puntaje, auc, ic_bootstrap  # noqa: E402
from tpi.segmentacion import segmentar_nodulo  # noqa: E402
from tpi.validacion import evaluar  # noqa: E402
from tpi.visualizacion import estimar_ruido_hu  # noqa: E402

PATRON = re.compile(r"(?P<paciente>.+)_s(?P<scan_id>\d+)_n(?P<nodulo>\d+)$")

# (nombre, tipo, parámetro). "fraccion": fracción de la dosis DEL ESTUDIO
# (×1/4 = un cuarto de los mAs con que se adquirió ese estudio); "mas": mAs
# absoluto (se agrega con --objetivo-mas). Como LIDC mezcla estudios de 40 a
# 600 mA, la misma fracción es otro nivel de ruido en cada estudio: por eso
# se guarda también el ruido agregado en HU (exacto: es la imagen de ruido).
def _etiqueta(f):
    return f"dosis ×1/{round(1 / f):g}" if f < 1 else "dosis ×1"


CONDICIONES = ([("original", "original", None)]
               + [(_etiqueta(f), "fraccion", f) for f in C.FRACCIONES_DOSIS]
               + [("JPEG q50", "jpeg", 50), ("JPEG q25", "jpeg", 25)])
# Columnas numéricas que describen la simulación (no son medidas del nódulo)
COLUMNAS_SIMULACION = {"ruido_hu", "ruido_agregado_hu", "ruido_agregado_tejido_hu", "ruido_original_tejido_hu",
                       "ruido_original_pulmon_hu", "ruido_total_pulmon_hu", "fraccion_dosis", "mas_estudio",
                       "mas_equivalente", "fotones_por_rayo", "frac_calcio_robusto", "scan_id", "nodulo"}
ORDEN = [c[0] for c in CONDICIONES]
TAMANO = {"area_mm2", "diametro_eq_mm"}
ENTORNO_EXTRA = {"frac_contacto_no_pulmon", "esqueleto_mm_por_mm_borde", "n_estructuras_convergentes",
                 "convergentes_por_10mm_borde"}


def adquisicion(d, enc, phi, electronico=True, anclaje="mas"):
    """Ancla la adquisición original del corte (ver DECISIONES §5-6):
      mas     fotones = Φ (del fabricante, 07a) × mAs del DICOM [lo que hace el paper];
      ruido   fotones calibrados con el ruido medido en la imagen, sin electrónico;
      hibrido fotones del ruido de la imagen + ruido electrónico.
    Ninguno es "el correcto": el error de cada uno se cuantifica en
    comparaciones/incertidumbre_dosis.py y se comparan con comparar_anclajes.py.
    Si el estudio no tiene mAs, el modo mas cae a ruido (y queda registrado)."""
    px = float(d["pixel_mm_original"])
    adq = B.Adquisicion(d["corte_completo"], px)
    tiene_mas = enc is not None and np.isfinite(enc.get("mas_efectivo", np.nan))
    if anclaje == "mas" and tiene_mas:
        fab = str(enc.get("fabricante", "?")).split()[0].upper() if isinstance(enc.get("fabricante"), str) else "?"
        return adq.desde_dicom(enc, phi=phi.get(fab, phi.get("TODOS")), electronico=electronico)
    s0 = B.ruido_global(adq.corte, px, B.mascara_tejido_blando(adq.corte, px))
    if anclaje == "hibrido":
        return adq.desde_ruido_hibrido(s0, enc, rng=np.random.default_rng(C.SEMILLA), electronico=electronico)
    ventana = enc.get("ventana_fbp") if enc is not None else None
    return adq.desde_ruido(s0, ventana=ventana, rng=np.random.default_rng(C.SEMILLA))


def degradar(d, adq, tipo, param, semilla):
    """Corte completo (resolución original, HU) en esa condición, la imagen
    de ruido agregada (None para JPEG) y la fracción de dosis."""
    orig = np.maximum(d["corte_completo"], -1000).astype(np.float32)
    if tipo == "original":
        return orig, None, 1.0
    if tipo in ("fraccion", "mas"):
        f = param if tipo == "fraccion" else param / mas_de(adq)
        if not np.isfinite(f) or f >= 1:
            return None, None, f  # el estudio ya tiene esa dosis o menos
        img, ruido = adq.a_fraccion(f, np.random.default_rng(semilla))
        return img, ruido, f
    if tipo == "jpeg":
        img8 = Z.hu_a_8bits(orig)
        return Z.de_8bits_a_hu(Z.comprimir(img8, "JPEG", param)[0]), None, np.nan
    raise ValueError(tipo)


def mas_de(adq):
    return getattr(adq, "mas", np.nan)


def como_si_fuera_original(d, img_completa):
    """Mismo remuestreo y recorte que datos.preparar_corte, con otra imagen."""
    rem = _remuestrear(np.maximum(img_completa, -1024), float(d["pixel_mm_original"]), 1)
    d2 = {k: v for k, v in d.items() if not k.startswith("_")}  # sin cachés del pulmón
    d2["corte_completo_rem"] = rem
    d2["img"] = _recortar(rem, tuple(d["semilla_completo"]), d["img"].shape[0], -1000.0)
    return d2


def procesar(args):
    """Un nódulo en todas las condiciones (corre en un proceso aparte)."""
    ruta, T, R, enc, phi, condiciones, electronico, anclaje = args
    d = cargar_corte(Path(ruta))
    m = PATRON.match(Path(ruta).stem)
    clave = dict(paciente=m["paciente"], scan_id=int(m["scan_id"]), nodulo=int(m["nodulo"]))
    semilla = zlib.crc32(Path(ruta).stem.encode())  # mismo ruido cada vez que se corre
    px = float(d["pixel_mm_original"])
    adq = adquisicion(d, enc, phi, electronico, anclaje)
    adq.mas = float(enc["mas_efectivo"]) if enc is not None and np.isfinite(enc.get("mas_efectivo", np.nan)) else np.nan
    pulmon_orig = mascara_pulmon(d["corte_completo_rem"])
    rem = lambda x: _remuestrear(x, px, 1)  # noqa: E731  (misma grilla de 0,7 mm del análisis)
    tejido_rem = B.mascara_tejido_blando(d["corte_completo_rem"], C.PIXEL_MM)
    # Ruido que según el modelo ya trae la original, en el pulmón (lo que ve
    # el nódulo) y en tejido blando (donde se mide el real, para comparar)
    r0 = rem(adq.ruido_original(np.random.default_rng(semilla + 1)))
    sim = dict(anclaje=adq.anclaje, ventana_fbp=adq.ventana, fotones_por_rayo=adq.N_alta, mas_estudio=adq.mas,
               ruido_original_tejido_hu=B.ruido_global(d["corte_completo_rem"], C.PIXEL_MM, tejido_rem),
               ruido_original_pulmon_hu=float(r0[pulmon_orig].std()) if pulmon_orig.any() else np.nan)
    filas = []
    for i, (nombre, tipo, param) in enumerate(condiciones):
        img, ruido, f = degradar(d, adq, tipo, param, semilla + 10 * i)
        if img is None:
            continue
        d2 = como_si_fuera_original(d, img)
        if ruido is not None:  # ruido agregado EXACTO: es la imagen de ruido
            rr = rem(ruido)
            agregado = float(rr[pulmon_orig].std()) if pulmon_orig.any() else np.nan
            agregado_tejido = float(rr[tejido_rem].std()) if tejido_rem.any() else np.nan
        elif tipo == "original":
            agregado = agregado_tejido = 0.0
        else:  # JPEG: diferencia con la original dentro del pulmón (MAD, para que los bordes no pesen)
            dif = (d2["corte_completo_rem"] - d["corte_completo_rem"])[pulmon_orig]
            agregado = float(1.4826 * np.median(np.abs(dif - np.median(dif)))) if dif.size else np.nan
            agregado_tejido = np.nan
        seg, cruda = segmentar_nodulo(d2, umbral=T, devolver_cruda=True, radio_cierre_mm=R)
        fila = dict(**clave, condicion=nombre, **sim, fraccion_dosis=f, mas_equivalente=f * adq.mas,
                    ruido_hu=estimar_ruido_hu(d2["img"]), ruido_agregado_hu=agregado,
                    ruido_agregado_tejido_hu=agregado_tejido,
                    ruido_total_pulmon_hu=(float(np.hypot(sim["ruido_original_pulmon_hu"], agregado))
                                           if tipo != "jpeg" else np.nan),
                    **evaluar(seg, d["consenso"], d["radiologos"]))
        if seg.sum() >= 5:
            fila.update(caracteristicas_interior(d2["img"], seg, mascara_cruda=cruda))
            fila["frac_calcio_robusto"] = fraccion_calcio_robusta(d2["img"], seg)  # para el 08
            pulmon = _recortar(mascara_pulmon(d2["corte_completo_rem"]), tuple(d["semilla_completo"]),
                               d["img"].shape[0], False)
            fila.update(caracteristicas_entorno(d2["img"], seg, pulmon)[0])
        filas.append(fila)
    return filas


def parametros():
    ajuste = C.DIR_RESULTADOS / "04_ajuste_umbral.csv"
    if ajuste.exists():
        t = pd.read_csv(ajuste)
        mejor = t.loc[t.dice_medio.idxmax()]
        return int(mejor["umbral_hu"]), float(mejor.get("radio_cierre_mm", C.RADIO_CIERRE_PULMON_MM))
    return C.UMBRAL_HU, C.RADIO_CIERRE_PULMON_MM


def constante_tubo():
    """Φ calibrado en el 07a (por fabricante); si no se corrió, el de config."""
    ruta = C.DIR_RESULTADOS / "07a_constante.csv"
    if ruta.exists():
        t = pd.read_csv(ruta)
        return dict(zip(t.fabricante, t.phi)), "07a (calibrado con LIDC)"
    return {"TODOS": C.FOTONES_POR_MAS_MM2}, "config (SIN calibrar: correr el 07a)"


def encabezados():
    ruta = C.DIR_RESULTADOS / "encabezados_dicom.csv"
    if not ruta.exists():
        print("AVISO: no está resultados/encabezados_dicom.csv (correr 03b): todo se ancla con el ruido de la imagen")
        return {}
    t = pd.read_csv(ruta)
    out = {}
    for r in t.to_dict("records"):
        r["mas_efectivo"] = mas_efectivo(r)   # índice de dosis = corriente (ver tpi/encabezados.py)
        # ventana del kernel con la regla ACTUAL (el CSV guarda la del 03b)
        r["familia_kernel"], r["ventana_fbp"] = familia_kernel(r.get("kernel"))
        out[(r["paciente"], int(r["scan_id"]), int(r["nodulo"]))] = r
    return out


def simular_todo(args):
    T, R = parametros()
    rutas = sorted(Path(args.carpeta).glob("*.npz"))
    if args.limite:
        rutas = rutas[:args.limite]
    if not rutas:
        sys.exit("No hay cortes en " + args.carpeta)
    condiciones = list(CONDICIONES) + [(f"{m:g} mAs", "mas", float(m)) for m in args.objetivo_mas]
    phi, origen = constante_tubo()
    encs = encabezados()
    trabajos = []
    for r in rutas:
        m = PATRON.match(r.stem)
        enc = encs.get((m["paciente"], int(m["scan_id"]), int(m["nodulo"])))
        trabajos.append((str(r), T, R, enc, phi, condiciones, not args.sin_electronico, args.anclaje))
    print(f"Simulando {len(rutas)} nódulos × {len(condiciones)} condiciones "
          f"(T = {T} HU, cierre {R:g} mm) con {args.procesos} procesos...")
    print(f"  constante del tubo: {origen}; ruido electrónico: {'no' if args.sin_electronico else 'sí'}")
    filas = []
    with ProcessPoolExecutor(args.procesos) as ex:
        for i, f in enumerate(ex.map(procesar, trabajos), 1):
            filas += f
            if i % 20 == 0 or i == len(rutas):
                print(f"  {i}/{len(rutas)}", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(C.DIR_RESULTADOS / f"07_baja_dosis{sufijo_anclaje(args)}.csv", index=False)
    return df


def sufijo_anclaje(args):
    suf = {"mas": "", "ruido": "_anclaje_ruido", "hibrido": "_anclaje_hibrido"}[args.anclaje]
    se = getattr(args, "sigma_e2", None)
    if se is not None and args.anclaje != "ruido":
        suf += f"_se{se:g}"
    return suf


def ccc(x, y):
    """Coeficiente de concordancia de Lin."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6:
        return np.nan
    vx, vy = x.var(), y.var()
    den = vx + vy + (x.mean() - y.mean()) ** 2
    return float(2 * np.cov(x, y, ddof=0)[0, 1] / den) if den > 0 else np.nan


def pliegues(pacientes, n=5, semilla=0):
    unicos = np.unique(pacientes)
    np.random.default_rng(semilla).shuffle(unicos)
    asig = {p: i % n for i, p in enumerate(unicos)}
    return np.array([asig[p] for p in pacientes])


def cv_puntaje(X_arma, X_aplica, y, pacientes, k):
    """Validación cruzada por paciente: arma con X_arma (pliegues de
    entrenamiento) y puntúa X_aplica (pliegue de prueba)."""
    pl = pliegues(pacientes)
    s = np.full(len(y), np.nan)
    for f in np.unique(pl):
        tr, te = pl != f, pl == f
        modelo = Puntaje(k).ajustar(X_arma[tr], y[tr])
        s[te] = modelo.puntuar(X_aplica[te])
    return s


def analizar(df, args):
    car = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    pob = car[car.grupo.isin(["benigno", "sospechoso"]) & (car.conjunto == "entrenamiento")]
    if not args.todos:
        pob = pob[pob.aceptable]  # aceptables en la imagen ORIGINAL: misma población en todas las condiciones
    if args.sin_calcio:
        pob = pob[pob.frac_calcio <= C.FRAC_CALCIO_MAX]
    claves = ["paciente", "scan_id", "nodulo"]
    pob = pob[claves + ["grupo", "concordancia_radiologos"]]
    df = df.drop(columns=["concordancia_radiologos"], errors="ignore").merge(pob, on=claves)
    sufijo = ("_sincalcio" if args.sin_calcio else "") + sufijo_anclaje(args)
    n = df.drop_duplicates(claves).grupo.value_counts().to_dict()
    print(f"\nPoblación: {sum(n.values())} nódulos de entrenamiento {n}"
          f"{' sin calcio' if args.sin_calcio else ''}")

    # a. segmentación
    if "ruido_agregado_hu" not in df:
        df["ruido_agregado_hu"] = np.nan
    orden = ORDEN + [c for c in df.condicion.unique() if c not in ORDEN]
    for col in ("ruido_total_pulmon_hu", "fraccion_dosis", "mas_equivalente"):
        if col not in df:
            df[col] = np.nan
    seg = (df.groupby("condicion").agg(ruido_agregado_hu=("ruido_agregado_hu", "median"),
                                        ruido_total_pulmon_hu=("ruido_total_pulmon_hu", "median"),
                                        mas_equivalente=("mas_equivalente", "median"),
                                        n=("dice", "size"),
                                        frac_calcio_media=("frac_calcio", "mean"),
                                        dice_medio=("dice", "mean"),
                                        aceptables=("aceptable", "mean")).reindex(orden).dropna(how="all"))
    seg.to_csv(C.DIR_RESULTADOS / f"07_segmentacion{sufijo}.csv")
    print("\n== a. Segmentación en cada condición ==")
    print(seg.round(3).to_string())

    # b. reproducibilidad
    medidas = [c for c in df.select_dtypes("number").columns
               if c not in META and c not in COLUMNAS_SIMULACION]
    base = df[df.condicion == "original"].set_index(claves)
    rep = {}
    for cond in seg.index:
        if cond == "original":
            continue
        otra = df[df.condicion == cond].set_index(claves).reindex(base.index)
        rep[cond] = {c: ccc(base[c].to_numpy(float), otra[c].to_numpy(float)) for c in medidas}
    rep = pd.DataFrame(rep)
    rep.index.name = "medida"
    rep.to_csv(C.DIR_RESULTADOS / f"07_reproducibilidad{sufijo}.csv")
    col = "dosis ×1/10" if "dosis ×1/10" in rep else rep.columns[-1]
    print(f"\n== b. Reproducibilidad (CCC de Lin vs original; ≥ 0,9 robusta, < 0,7 frágil) ==")
    print(rep.sort_values(col, ascending=False).round(2).to_string())

    # c. clasificación
    entorno = [c for c in medidas if c.startswith("anillo_") or c in ENTORNO_EXTRA]
    interior = [c for c in medidas if c not in entorno]
    conjuntos = {"diámetro solo": (["diametro_eq_mm"], 1),
                 "interior sin tamaño": ([c for c in interior if c not in TAMANO], args.k),
                 "interior": (interior, args.k),
                 "interior + entorno": (interior + entorno, args.k)}
    base = base.reset_index()
    y = (base.grupo == "sospechoso").to_numpy(int)
    filas = []
    for cond in seg.index:
        otra = df[df.condicion == cond].set_index(claves).reindex(base.set_index(claves).index).reset_index()
        for nombre, (cols, k) in conjuntos.items():
            s_re = cv_puntaje(otra[cols], otra[cols], y, base.paciente.to_numpy(), k)
            s_tr = cv_puntaje(base[cols], otra[cols], y, base.paciente.to_numpy(), k)
            lo, hi = ic_bootstrap(y, s_tr, base.paciente)
            filas.append(dict(condicion=cond, puntaje=nombre, auc_rearmado=auc(y, s_re),
                              auc_transferido=auc(y, s_tr), transf_ic95_inf=lo, transf_ic95_sup=hi))
    cla = pd.DataFrame(filas)
    cla.to_csv(C.DIR_RESULTADOS / f"07_clasificacion{sufijo}.csv", index=False)
    print("\n== c. AUC (validación cruzada por paciente) ==")
    print("   re-armado = armado y evaluado con esa dosis; transferido = armado con la original")
    print(cla.pivot(index="condicion", columns="puntaje", values="auc_transferido").reindex(seg.index)
          .round(3).to_string(header=True).replace("puntaje", "TRANSFERIDO"))
    print(cla.pivot(index="condicion", columns="puntaje", values="auc_rearmado").reindex(seg.index)
          .round(3).to_string().replace("puntaje", "RE-ARMADO"))

    figura(seg, rep, cla, C.DIR_FIGURAS / f"07_baja_dosis{sufijo}.png")


def figura(seg, rep, cla, ruta):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3]))
    x = np.arange(len(seg))
    ax[0].plot(x, seg.dice_medio, "o-", label="Dice medio")
    ax[0].plot(x, seg.aceptables, "s--", label="fracción aceptable")
    a2 = ax[0].twinx()
    a2.bar(x, seg.ruido_agregado_hu, alpha=0.2, color="gray")
    a2.set_ylabel("ruido agregado (HU)", color="gray")
    ax[0].set_xticks(x, seg.index, rotation=40, ha="right", fontsize=7)
    ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_title("a. Segmentación")

    conds = [c for c in seg.index if c in rep.columns]
    top = rep[conds].sort_values(conds[-1] if "dosis ×1/10" not in conds else "dosis ×1/10", ascending=False)
    top = pd.concat([top.head(12), top.tail(12)]).drop_duplicates()
    im = ax[1].imshow(top.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax[1].set_yticks(range(len(top)), top.index, fontsize=6)
    ax[1].set_xticks(range(len(conds)), conds, rotation=40, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax[1], label="CCC vs original")
    ax[1].set_title("b. Reproducibilidad (12 más robustas / 12 más frágiles)")

    for (nombre, sub), color in zip(cla.groupby("puntaje", sort=False), plt.cm.tab10.colors):
        sub = sub.set_index("condicion").reindex(seg.index)
        ax[2].plot(x, sub.auc_transferido, "o-", color=color, label=f"{nombre} (transferido)")
        ax[2].plot(x, sub.auc_rearmado, ":", color=color, alpha=0.7)
    ax[2].axhline(0.5, color="k", lw=0.8)
    ax[2].set_xticks(x, seg.index, rotation=40, ha="right", fontsize=7)
    ax[2].set_ylabel("AUC")
    ax[2].set_title("c. Clasificación (línea punteada = re-armado)")
    ax[2].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("->", ruta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limite", type=int, default=0, help="solo los primeros N nódulos (para probar)")
    ap.add_argument("--solo-analisis", action="store_true")
    ap.add_argument("--todos", action="store_true", help="incluir segmentaciones no aceptables")
    ap.add_argument("--sin-calcio", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--objetivo-mas", type=float, nargs="*", default=[],
                    help="además, mAs absolutos a simular (p. ej. 40 25: dosis de tamizaje)")
    ap.add_argument("--sin-electronico", action="store_true", help="sin ruido electrónico (sensibilidad)")
    ap.add_argument("--anclaje", choices=["mas", "ruido", "hibrido"], default="mas",
                    help="mas: Φ × mAs del DICOM; ruido: ruido de la imagen; hibrido: ruido de la imagen + "
                         "electrónico. Correr los tres y comparar (comparaciones/comparar_anclajes.py)")
    ap.add_argument("--sigma-e2", default=None,
                    help="ruido electrónico por lectura (por defecto el de config, 30): un número, o 'lidc' "
                         "(el estimado por el 07a en 07a_electronico.csv) o 'lidc_sup' (el límite superior "
                         "de su IC 95 %%). La salida lleva _se<valor>")
    args = ap.parse_args()
    if args.sigma_e2 is not None:
        if args.sigma_e2 in ("lidc", "lidc_sup"):
            t = pd.read_csv(C.DIR_RESULTADOS / "07a_electronico.csv").iloc[0]
            args.sigma_e2 = round(float(t.sigma_e2 if args.sigma_e2 == "lidc" else t.ic95_sup), 1)
        args.sigma_e2 = float(args.sigma_e2)
        os.environ["TPI_SIGMA_E2"] = str(args.sigma_e2)   # lo heredan los procesos en paralelo
        C.SIGMA_E2_LECTURA = args.sigma_e2
        print(f"Ruido electrónico por lectura: {args.sigma_e2:g} (en lugar de 30)")
    C.DIR_FIGURAS.mkdir(exist_ok=True)
    ruta = C.DIR_RESULTADOS / f"07_baja_dosis{sufijo_anclaje(args)}.csv"
    df = pd.read_csv(ruta) if args.solo_analisis else simular_todo(args)
    analizar(df, args)


if __name__ == "__main__":
    main()
