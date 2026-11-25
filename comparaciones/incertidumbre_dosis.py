"""Incertidumbre de la simulación de baja dosis: ¿a qué dosis corresponde DE
VERDAD cada etiqueta ("×1/4", "×1/10"...), si los parámetros que no podemos
medir en LIDC están mal?

La pregunta, con un ejemplo: el 07 dice que una imagen es "×1/4" (un cuarto
de la dosis del estudio). Eso es cierto solo si el bowtie, el ruido electrónico,
el kernel, la constante del tubo, etc. son los que supusimos. Si el bowtie real
del equipo era otro, el mismo ruido agregado corresponde a otra fracción. Acá
se calcula ESA fracción real, f_real, para cada combinación posible de
parámetros, y con eso un intervalo.

Cómo se calcula f_real (para cada combinación θ de parámetros "verdaderos"):
  1. MUNDO VERDADERO (θ): se simula el equipo con esos parámetros, se ajustan
     los fotones para que el ruido del tejido blando dé el ruido MEDIDO en la
     imagen (es la imagen real: su ruido es un dato), y se obtiene, cerca del
     nódulo, la varianza cuántica A_v y electrónica E_v que la imagen ya trae.
  2. LO QUE HACE EL PIPELINE: con los parámetros por defecto (y el error de
     anclaje: Φ en modo mAs, o el error de medir el ruido en modo ruido)
     agrega, cerca del nódulo, una varianza σ²_agr(f).
  3. Esa varianza agregada, en el mundo verdadero, equivale a bajar la dosis
     a f_real, que sale de  A_v (1/f_real - 1) + E_v (1/f_real² - 1) = σ²_agr(f)
     (la varianza cuántica crece como 1/f, la electrónica como 1/f²).
  Se usan siempre las mismas semillas de ruido (números aleatorios comunes),
  así las diferencias entre combinaciones son por los parámetros y no por el azar.

Dos análisis:
  a. Uno por vez (gráfico de tornado): cada parámetro en su mínimo y su máximo
     plausibles, el resto por defecto. Muestra qué parámetro pesa más.
  b. Todos a la vez (Monte Carlo): se sortean todos los parámetros dentro de
     sus rangos y se reporta la mediana y el intervalo del 95 % de la dosis
     real de cada etiqueta, y qué parte de la variabilidad explica cada uno.

Los rangos y de dónde sale cada uno están en PARAMETROS (abajo) y en
DECISIONES.md §6. Para Φ: si existe resultados/07a_estudios.csv (corrida real
del 07a) se usa la dispersión real entre estudios; si no, un rango provisorio.

Uso:
  python comparaciones/incertidumbre_dosis.py                  # fantoma
  python comparaciones/incertidumbre_dosis.py --carpeta datos/cortes --casos 3
  opciones: --muestras 200 (Monte Carlo), --anclaje mas|ruido|hibrido, --procesos N
Salidas: resultados/comparaciones/incertidumbre_{tornado,montecarlo,resumen}.csv
         figuras/comparaciones/incertidumbre_dosis.png
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tests"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from tpi import baja_dosis as B  # noqa: E402
from tpi import config as C  # noqa: E402

ETIQUETAS = (0.5, 0.25, 0.1, 0.05)
VENTANAS = ["ramp", "shepp-logan", "cosine", "hamming", "hann"]   # de más dura a más suave
N_REF = 1e5
FNQ_MAX = 0.8   # tope para la fracción no cuántica (ver sortear)
_AVISOS = {}

# nombre: (por defecto, mínimo, máximo, escala, qué es, de dónde sale el rango)
PARAMETROS = {
    "sff_bowtie": (C.BOWTIE_SFF, *C.BOWTIE_SFF_RANGO, "lin", "factor del bowtie (Kramer)",
                   "mín. y máx. de los 11 tomógrafos ajustados por Kramer et al. 2017"),
    "dsi_mm": (C.BOWTIE_DSI_MM, 540.0, 600.0, "lin", "distancia fuente-isocentro",
               "rango de los equipos clínicos (~54-60 cm)"),
    "mu_pmma": (C.MU_PMMA_MM, 0.021, 0.025, "lin", "μ efectivo del PMMA del bowtie",
                "NIST, energía efectiva 80 → 50 keV (endurecimiento del haz)"),
    "sigma_e2": (C.SIGMA_E2_LECTURA, 5.0, 150.0, "log", "ruido electrónico por lectura",
                 "~29 en un equipo moderno (Elhamiasl y Nuyts 2020); rango amplio para equipos viejos"),
    "vistas_rotacion": (float(C.VISTAS_POR_ROTACION), 900.0, 2400.0, "lin", "vistas por rotación",
                        "equipos clínicos (~1000-2400)"),
    "ancho_canal_mm": (C.ANCHO_DETECTOR_ISO_MM, 0.5, 0.75, "lin", "ancho de canal en el isocentro",
                       "detectores clínicos"),
    "ancho_fila_mm": (C.ANCHO_FILA_MM, 0.5, 1.25, "lin", "ancho de fila",
                      "equipos de 4-64 filas de la época de LIDC"),
    "ventana": (1.0, 0.0, 2.0, "discreto", "ventana del kernel (familia vecina)",
                "el kernel real no es público: una familia más dura o más suave"),
    "n_vistas_sim": (float(C.N_VISTAS_DOSIS), 180.0, 720.0, "discreto", "vistas de la simulación",
                     "discretización numérica"),
    "k_phi": (1.0, 0.5, 2.0, "log", "error de la constante del tubo Φ",
              "provisorio; con el 07a corrido, la dispersión real Φ_fabricante/Φ_estudio"),
    "error_ruido": (0.0, -0.05, 0.05, "lin", "error al medir el ruido original",
                    "con el 07a corrido: error medido en cada imagen (mitades del tejido); si no, "
                    "Christianson et al. 2015 (3,4-4,7 %)"),
    "frac_no_cuantica": (0.0, 0.0, 0.2, "lin", "parte del ruido medido que no es cuántica",
                         "con el 07a corrido: IC 95 % del ajuste σ0² = cuántico/mAs + b; si no, provisorio 0-20 %"),
    "exponente_kvp": (C.EXPONENTE_KVP, 2.0, 3.0, "lin", "exponente kVp (estudio a 140 kVp)",
                      "solo pesa si el estudio no es de 120 kVp"),
}
# Parámetros que no intervienen en cada modo de anclaje
NO_APLICA = {"mas": {"error_ruido"}, "ruido": {"k_phi", "exponente_kvp"},
             "hibrido": {"k_phi", "exponente_kvp"}}
FISICOS = ["sff_bowtie", "dsi_mm", "mu_pmma", "sigma_e2", "vistas_rotacion", "ancho_canal_mm",
           "ancho_fila_mm", "ventana"]


# ---------------------------------------------------------------------------
# Caso de estudio
# ---------------------------------------------------------------------------
def caso_fantoma():
    from test_baja_dosis import ESP, N, PX, fantoma_limpio
    verdad = fantoma_limpio()
    rr, cc = np.mgrid[:N, :N]
    d2 = (rr - 250) ** 2 + (cc - 165) ** 2
    pulmon_izq = ((rr - 250) / 140) ** 2 + ((cc - 165) / 65) ** 2 <= 1
    nodulo = (d2 <= (15 / PX) ** 2) & (d2 > 14 ** 2) & pulmon_izq
    adq = B.Adquisicion(verdad, PX)
    adq.N_alta, adq.ventana = 5e5, "shepp-logan"
    img = verdad + adq.ruido_original(np.random.default_rng(7))
    return dict(nombre="fantoma", img=img, px=PX, espesor=ESP, kvp=120.0, nodulo=nodulo, ventana=1)


def elegir_casos(carpeta, n, enc):
    """n cortes VARIADOS, no los primeros n en orden alfabético (que pueden ser
    todos del mismo equipo): se recorren los grupos fabricante × familia de
    kernel, de a uno por grupo, empezando por los grupos más grandes."""
    rutas = sorted(Path(carpeta).glob("*.npz"))
    if not enc:
        return rutas[:n], {r.stem: ("?", "?") for r in rutas[:n]}
    from tpi.encabezados import familia_kernel
    grupos, grupo_de = {}, {}
    for r in rutas:
        e = enc.get(r.stem)
        clave = (str(e.fabricante).split()[0].upper(), familia_kernel(e.kernel)[0]) if e is not None else ("?", "?")
        grupos.setdefault(clave, []).append(r)
        grupo_de[r.stem] = clave
    orden = sorted(grupos.values(), key=len, reverse=True)
    elegidos, i = [], 0
    while len(elegidos) < n and any(i < len(g) for g in orden):
        for g in orden:
            if i < len(g) and len(elegidos) < n:
                elegidos.append(g[i])
        i += 1
    return elegidos, {r.stem: grupo_de[r.stem] for r in elegidos}


def casos_reales(carpeta, n):
    """(generador de casos, cantidad de grupos fabricante × kernel). Los casos
    se cargan de a uno (con --casos auto no se sabe de antemano cuántos)."""
    enc = _encabezados_por_corte()
    rutas, grupo_de = elegir_casos(carpeta, n, enc)
    return _cargar_casos(rutas, enc, grupo_de), len(set(grupo_de.values()))


def _encabezados_por_corte():
    ruta = C.DIR_RESULTADOS / "encabezados_dicom.csv"
    if not ruta.exists():
        return {}
    t = pd.read_csv(ruta)
    return {f"{r.paciente}_s{r.scan_id}_n{r.nodulo}": r for _, r in t.iterrows()}


def _cargar_casos(rutas, enc, grupo_de):
    from tpi.datos import cargar_corte
    from tpi.encabezados import familia_kernel
    from tpi.entorno import mascara_pulmon
    for r in rutas:
        d = cargar_corte(r)
        px = float(d["pixel_mm_original"])
        img = np.maximum(d["corte_completo"], -1000).astype(np.float32)
        sr, sc = np.asarray(d["semilla_completo"], float) * C.PIXEL_MM / px
        rr, cc = np.mgrid[:img.shape[0], :img.shape[1]]
        d2 = (rr - sr) ** 2 + (cc - sc) ** 2
        nodulo = (d2 <= (15 / px) ** 2) & mascara_pulmon(img, pixel_mm=px)
        e = enc.get(r.stem)
        ventana = familia_kernel(e.kernel if e is not None else None)[1]
        yield dict(nombre=r.stem, grupo="/".join(grupo_de.get(r.stem, ("?", "?"))), img=img, px=px,
                   nodulo=nodulo, ventana=VENTANAS.index(ventana),
                   espesor=float(e.espesor_mm) if e is not None and np.isfinite(e.espesor_mm) else C.MAX_ESPESOR_MM,
                   kvp=float(e.kvp) if e is not None and np.isfinite(e.kvp) else 120.0)


# ---------------------------------------------------------------------------
# Núcleo: varianzas cuántica y electrónica cerca del nódulo
# ---------------------------------------------------------------------------
_CTX = {}


def preparar(caso):
    """Sinogramas (uno por cantidad de vistas), máscara de tejido y ruido medido."""
    _CTX.clear()
    _CTX.update(caso)
    _CTX["adq"] = {n: B.Adquisicion(caso["img"], caso["px"], n_vistas=int(n)) for n in (180, 360, 720)}
    a = _CTX["adq"][360]
    _CTX["tejido"] = B.mascara_tejido_blando(a.corte, caso["px"])
    _CTX["sigma0"] = B.ruido_global(a.corte, caso["px"], _CTX["tejido"])


def varianzas(par, n_vistas=360, sigma0=None):
    """Con los parámetros físicos `par`: fotones que reproducen el ruido medido
    en tejido blando, y varianzas cuántica (A) y electrónica (E) que eso deja
    cerca del nódulo. Siempre las mismas semillas."""
    adq = _CTX["adq"][int(n_vistas)]
    px, esp = _CTX["px"], _CTX["espesor"]
    adq.usar_bowtie(modelo="kramer", sff=par["sff_bowtie"], dsi_mm=par["dsi_mm"], mu_pmma=par["mu_pmma"])
    adq.ventana = VENTANAS[int(round(par["ventana"]))]
    s_ref = adq.sigma_ruido(N_REF, _CTX["tejido"], rng=np.random.default_rng(11))
    N = N_REF * (s_ref / (_CTX["sigma0"] if sigma0 is None else sigma0)) ** 2
    m = ((par["vistas_rotacion"] / adq.n_vistas) * (px / par["ancho_canal_mm"]) * (esp / par["ancho_fila_mm"]))
    se2 = par["sigma_e2"] * m
    # Ruido cuántico y electrónico de la MISMA realización, separados: el FBP es
    # lineal, así que la imagen del ruido electrónico se puede reconstruir sola
    rng = np.random.default_rng(12)
    Nr = N * adq.bowtie[:, None]
    media = Nr * np.exp(-adq.p)
    P = np.maximum(rng.poisson(media).astype(float), C.CUENTAS_MIN)
    G = rng.normal(0.0, np.sqrt(se2), adq.p.shape)
    q = -np.log(P / Nr) - adq.p
    e = -np.log(np.maximum(P + G, C.CUENTAS_MIN) / Nr) - adq.p - q
    nod = _CTX["nodulo"]
    rq = B.reconstruir_ruido(q, px, adq.corte.shape[0], adq.ventana)[nod]
    re = B.reconstruir_ruido(e, px, adq.corte.shape[0], adq.ventana)[nod]
    return N, float(rq.var()), float(re.var())


def evaluar(args):
    """Fracción real de cada etiqueta para UNA combinación de parámetros."""
    par, modo = args
    # 1. mundo verdadero: si una fracción del ruido medido no es cuántica, los
    #    fotones verdaderos se calibran solo con la parte cuántica (+ electrónica)
    s0_v = _CTX["sigma0"] * np.sqrt(1 - par["frac_no_cuantica"])
    _, A_v, E_v = varianzas(par, sigma0=s0_v)
    A0, E0 = _CTX["pipeline"][int(par["n_vistas_sim"])]        # 2. pipeline (parámetros por defecto)
    # El pipeline calibra con TODO el ruido medido (no sabe cuánto no es cuántico)
    if modo == "mas":
        k = par["k_phi"] * (_CTX["kvp"] / 120.0) ** (C.EXPONENTE_KVP - par["exponente_kvp"])
        A_p, E_p = A0 / k, E0 / k ** 2
    elif modo == "hibrido":
        k = (1 + par["error_ruido"]) ** -2
        A_p, E_p = A0 / k, E0 / k ** 2                         # fotones del ruido + electrónico
    else:
        k = (1 + par["error_ruido"]) ** -2
        A_p, E_p = A0 / k, 0.0                                 # en modo ruido no se agrega electrónico
    fila = dict(par)
    for f in ETIQUETAS:
        v = A_p * (1 / f - 1) + E_p * (1 / f ** 2 - 1)         # varianza que agrega el pipeline
        if E_v > 1e-12:                                        # 3. A_v(u-1) + E_v(u²-1) = v
            u = (-A_v + np.sqrt(A_v ** 2 + 4 * E_v * (A_v + E_v + v))) / (2 * E_v)
        else:
            u = 1 + v / A_v
        fila[f"inv_f_real_{f:g}"] = float(u)                   # 1 / f_real
    return fila


def defecto():
    return {k: v[0] for k, v in PARAMETROS.items()}


def preparar_pipeline():
    """Lo que agrega el pipeline con los parámetros por defecto, para cada
    cantidad de vistas de la simulación (en modo mAs también lleva electrónico)."""
    par = defecto()
    par["ventana"] = _CTX["ventana"]
    _CTX["pipeline"] = {n: varianzas(par, n)[1:] for n in (180, 360, 720)}


def _init(caso):
    preparar(caso)
    preparar_pipeline()


# ---------------------------------------------------------------------------
def tornado(modo):
    filas = []
    base = defecto()
    base["ventana"] = _CTX["ventana"]
    ref = evaluar((base, modo))
    for nombre, (d, lo, hi, escala, desc, _) in PARAMETROS.items():
        if nombre in NO_APLICA[modo]:
            continue
        valores = [lo, hi]
        if nombre == "ventana":
            valores = [max(_CTX["ventana"] - 1, 0), min(_CTX["ventana"] + 1, len(VENTANAS) - 1)]
        for v in valores:
            p = dict(base, **{nombre: v})
            kvp_orig = _CTX["kvp"]
            if nombre == "exponente_kvp" and kvp_orig == 120.0:
                _CTX["kvp"] = 140.0   # a 120 kVp no influye: se muestra para un estudio a 140 kVp
            r = evaluar((p, modo))
            _CTX["kvp"] = kvp_orig
            filas.append(dict(parametro=nombre, descripcion=desc, valor=v,
                              **{k: r[k] for k in r if k.startswith("inv_f_real")}))
    t = pd.DataFrame(filas)
    for k in ref:
        if k.startswith("inv_f_real"):
            t["ref_" + k] = ref[k]   # resultado con todos los parámetros por defecto
    return t


def sortear(n, modo, semilla=0):
    rng = np.random.default_rng(semilla)
    ks = errs = nq = None
    ruta = C.DIR_RESULTADOS / "07a_estudios.csv"
    if ruta.exists():
        t = pd.read_csv(ruta)
        ks = t["k_phi"].dropna().to_numpy()
        if "error_ruido_rel" in t:
            errs = t["error_ruido_rel"].dropna().to_numpy()
    ruta = C.DIR_RESULTADOS / "07a_no_cuantico.csv"
    if ruta.exists():
        t = pd.read_csv(ruta).set_index("fabricante")
        if "TODOS" in t.index:
            nq = (float(t.loc["TODOS", "frac_no_cuantica_ic95_inf"]), float(t.loc["TODOS", "frac_no_cuantica_ic95_sup"]))
            if nq[1] > FNQ_MAX:
                # Si casi todo el ruido medido "no fuera cuántico", el ruido
                # cuántico verdadero sería ~0 y cualquier ruido agregado
                # equivaldría a una reducción de dosis enorme. Un valor así
                # indica que el ajuste del 07a falló (p. ej., la corriente no
                # refleja la dosis), no que la física sea esa.
                if not _AVISOS.get("fnq"):
                    print(f"AVISO: el 07a da hasta {100 * nq[1]:.0f} % de ruido no cuántico; se limita a "
                          f"{100 * FNQ_MAX:.0f} %. Revisar el ajuste del 07a.")
                    _AVISOS["fnq"] = True
                nq = (min(nq[0], FNQ_MAX), FNQ_MAX)
    out = []
    for _ in range(n):
        p = {}
        for nombre, (d, lo, hi, escala, _, _) in PARAMETROS.items():
            if nombre == "ventana":
                p[nombre] = float(np.clip(_CTX["ventana"] + rng.integers(-1, 2), 0, len(VENTANAS) - 1))
            elif nombre == "n_vistas_sim":
                p[nombre] = float(rng.choice([180, 360, 720]))
            elif nombre == "k_phi" and ks is not None and len(ks) > 5:
                p[nombre] = float(rng.choice(ks))
            elif nombre == "error_ruido" and errs is not None and len(errs) > 5:
                p[nombre] = float(rng.choice(errs) * rng.choice([-1, 1]))   # error medido, signo al azar
            elif nombre == "frac_no_cuantica" and nq is not None:
                p[nombre] = float(rng.uniform(*nq))
            elif escala == "log":
                p[nombre] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                p[nombre] = float(rng.uniform(lo, hi))
        out.append((p, modo))
    return out, dict(phi=ks is not None, error_ruido=errs is not None, no_cuantico=nq is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=None)
    ap.add_argument("--casos", default="auto",
                    help="'auto': agrega nódulos hasta que converge; 'todos': todos los nódulos; o un número")
    ap.add_argument("--max-casos", type=int, default=80, help="tope para --casos auto")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="convergencia: semiancho relativo del IC 95 %% (bootstrap por nódulo) de la mediana, "
                         "P25 y P75. Los límites 2,5 %% y 97,5 %% se reportan con su incertidumbre, pero no se "
                         "exige que converjan (en distribuciones de cola larga necesitarían cientos de nódulos)")
    ap.add_argument("--muestras", type=int, default=300, help="sorteos de Monte Carlo por nódulo y anclaje")
    ap.add_argument("--anclaje", choices=["mas", "ruido", "hibrido", "todos"], default="todos")
    ap.add_argument("--solo-comparar", action="store_true",
                    help="no calcula: arma el gráfico que compara los anclajes con lo ya guardado")
    ap.add_argument("--sff-rango", type=float, nargs=2, default=None,
                    help="rango del factor del bowtie si el modelo del tomógrafo está en la tabla de Kramer")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2)))
    args = ap.parse_args()
    if args.sff_rango:
        d, _, _, *resto = PARAMETROS["sff_bowtie"]
        PARAMETROS["sff_bowtie"] = (float(np.mean(args.sff_rango)), *args.sff_rango, *resto)
    suf_esc = "_sff_tabulado" if args.sff_rango else ""
    dir_res = C.DIR_RESULTADOS / "comparaciones"
    dir_res.mkdir(parents=True, exist_ok=True)
    if args.solo_comparar:
        res = {m: pd.read_csv(dir_res / f"incertidumbre_resumen_{m}{suf_esc}.csv") for m in ("mas", "ruido", "hibrido")
               if (dir_res / f"incertidumbre_resumen_{m}{suf_esc}.csv").exists()}
        comparar_modos(res, C.DIR_FIGURAS / "comparaciones" / f"incertidumbre_anclajes{suf_esc}.png")
        return

    modos = ["mas", "ruido", "hibrido"] if args.anclaje == "todos" else [args.anclaje]
    modo_txt = "_".join(modos) + suf_esc
    auto = str(args.casos) == "auto"
    todos = str(args.casos) == "todos"
    n_max = args.max_casos if auto else (10 ** 6 if todos else int(args.casos))
    if args.carpeta:
        casos, n_grupos = casos_reales(args.carpeta, n_max)
    else:
        casos, n_grupos = iter([caso_fantoma()]), 1
    minimo = max(5, n_grupos) if auto else n_max

    # Lo ya calculado en una corrida anterior (se guarda nódulo por nódulo)
    parcial = {m: dir_res / f"incertidumbre_parcial_{m}{suf_esc}.csv" for m in modos}
    parcial_tor = {m: dir_res / f"incertidumbre_parcial_tornado_{m}{suf_esc}.csv" for m in modos}
    previo = {m: pd.read_csv(parcial[m]) if parcial[m].exists() else pd.DataFrame() for m in modos}
    hechos = set.intersection(*[set(previo[m].caso) if len(previo[m]) else set() for m in modos])
    if hechos:
        print(f"Retomando: {len(hechos)} nódulos ya calculados en una corrida anterior.")

    datos = {m: dict(tor=[], mc=[]) for m in modos}
    for m in modos:
        if parcial_tor[m].exists():
            datos[m]["tor"].append(pd.read_csv(parcial_tor[m]))
    historia = []
    k = 0
    for caso in casos:
        k += 1
        if caso["nombre"] in hechos:
            for m in modos:
                datos[m]["mc"].append(previo[m][previo[m].caso == caso["nombre"]])
            print(f"[{k}] {caso['nombre']}: ya estaba", flush=True)
        else:
            print(f"\n[{k}] {caso['nombre']} ({caso.get('grupo', '')})", flush=True)
            _init(caso)
            trabajos = []
            for m in modos:
                if k <= 3 and not parcial_tor[m].exists() or (k <= 3 and caso["nombre"] not in
                                                              set(pd.read_csv(parcial_tor[m]).caso)):
                    t = tornado(m)   # ilustrativo: con 3 nódulos alcanza para el gráfico
                    t.insert(0, "caso", caso["nombre"])
                    datos[m]["tor"].append(t)
                    t.to_csv(parcial_tor[m], mode="a", header=not parcial_tor[m].exists(), index=False)
                trabajos += sortear(args.muestras, m, semilla=k)[0]
            # Un solo grupo de procesos por nódulo (preparar el nódulo es lo caro)
            with ProcessPoolExecutor(args.procesos, initializer=_init, initargs=(caso,)) as ex:
                res = list(ex.map(evaluar, trabajos, chunksize=4))
            for m in modos:
                mc = pd.DataFrame([r for r, (_, mm) in zip(res, trabajos) if mm == m])
                mc.insert(0, "caso", caso["nombre"])
                datos[m]["mc"].append(mc)
                mc.to_csv(parcial[m], mode="a", header=not parcial[m].exists(), index=False)   # guardado YA
        fila, ok = convergencia(datos, k, args)
        historia.append(fila)
        if auto and k >= minimo and ok:
            print(f"\n==> Convergió con {k} nódulos: agregar más ya no mueve la mediana ni la mitad central "
                  f"(P25-P75) más de ±{100 * args.tol:g} %.")
            break
        if k >= n_max:
            break
    else:
        if auto:
            print(f"\n==> NO convergió con {k} nódulos (se acabaron): los intervalos se reportan igual, "
                  "con su incertidumbre (ver incertidumbre_convergencia).")
    if auto and k >= n_max:
        print(f"\n==> NO convergió con {k} nódulos (tope): los intervalos se reportan igual, con su incertidumbre.")
    hist = pd.DataFrame(historia)
    hist.to_csv(dir_res / f"incertidumbre_convergencia_{modo_txt}.csv", index=False)
    figura_convergencia(hist, modos, C.DIR_FIGURAS / "comparaciones" / f"incertidumbre_convergencia_{modo_txt}.png")

    resumenes = {m: finalizar(m, datos[m], fuentes_de(m), suf_esc, dir_res) for m in modos}
    if len(resumenes) > 1:
        comparar_modos(resumenes, C.DIR_FIGURAS / "comparaciones" / f"incertidumbre_anclajes{suf_esc}.png")


def fuentes_de(modo):
    """De dónde salieron los rangos (07a o provisorios); no depende del nódulo."""
    _CTX.setdefault("ventana", 1)
    return sortear(1, modo)[1]


def convergencia(datos, k, args, n_boot=400):
    """¿Agregar nódulos todavía cambia el resultado? Para cada anclaje y
    etiqueta se junta lo de todos los nódulos (cada uno pesa igual) y se
    calculan los percentiles 2,5 / 25 / 50 / 75 / 97,5 de 1/f_real. Después se
    remuestrean los NÓDULOS (bootstrap: se sortean k nódulos con reposición) y
    se mira cuánto se mueven: el semiancho relativo de su IC 95 % dice cuánto
    cambiaría ese número con otros nódulos. Converge cuando la mediana, el P25
    y el P75 se mueven menos que la tolerancia. Los extremos (2,5 y 97,5) se
    guardan con su semiancho pero no se exige que converjan: en distribuciones
    de cola larga necesitarían cientos de nódulos (se vio en la primera corrida)."""
    rng = np.random.default_rng(0)
    fila, ok = dict(n_nodulos=k), k >= 2
    pcts = [2.5, 25, 50, 75, 97.5]
    nombres = ("p2.5", "p25", "mediana", "p75", "p97.5")
    exigidos = np.array([False, True, True, True, False])
    peor_central, peor_colas = 0.0, 0.0
    for m, d in datos.items():
        for f in ETIQUETAS:
            col = f"inv_f_real_{f:g}"
            vals = [x[col].to_numpy() for x in d["mc"]]
            q = np.percentile(np.concatenate(vals), pcts)
            if k >= 2:
                boot = np.array([np.percentile(np.concatenate([vals[i] for i in rng.integers(0, k, k)]), pcts)
                                 for _ in range(n_boot)])
                semi = (np.percentile(boot, 97.5, axis=0) - np.percentile(boot, 2.5, axis=0)) / 2 / q
                ok = ok and bool(np.all(semi[exigidos] <= args.tol))
                peor_central = max(peor_central, float(np.max(semi[exigidos])))
                peor_colas = max(peor_colas, float(np.max(semi[~exigidos])))
            else:
                semi = np.full(len(pcts), np.nan)
            for nombre, v, e in zip(nombres, q, semi):
                fila[f"{m}_{f:g}_{nombre}"] = v
                fila[f"{m}_{f:g}_{nombre}_semiancho_rel"] = e
    if k >= 2:
        print(f"    convergencia: mediana/P25/P75 se mueven hasta ±{100 * peor_central:.1f} % "
              f"(objetivo ±{100 * args.tol:g} %); extremos 2,5/97,5 hasta ±{100 * peor_colas:.1f} %", flush=True)
    return fila, ok


def figura_convergencia(hist, modos, ruta):
    """Evidencia para el informe: cómo se estabilizan los límites al sumar nódulos."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, len(modos), figsize=(5.5 * len(modos), 4.4), squeeze=False)
    for ax, m in zip(axs[0], modos):
        for f, color in zip((0.25, 0.1), ("#4a7ab5", "#d9534f")):
            for nombre, ls in (("p2.5", ":"), ("p25", "--"), ("mediana", "-"), ("p75", "--"), ("p97.5", ":")):
                c = f"{m}_{f:g}_{nombre}"
                y = hist[c]
                e = hist[c + "_semiancho_rel"].fillna(0) * y
                ax.plot(hist.n_nodulos, y, ls, color=color, label=f"×1/{1 / f:g} {nombre}")
                ax.fill_between(hist.n_nodulos, y - e, y + e, color=color, alpha=0.12)
        ax.set_yscale("log")
        ax.set_xlabel("nódulos incluidos")
        ax.set_ylabel("1 / fracción real (mediana y límites del IC 95 %)")
        ax.set_title(f"Convergencia, anclaje {m}\n(banda: cuánto cambiaría con otros nódulos)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("->", ruta)


def finalizar(modo, d, fuentes, suf_esc, dir_res):
    suf = f"_{modo}{suf_esc}"
    tor, mc = pd.concat(d["tor"]), pd.concat(d["mc"])
    tor.to_csv(dir_res / f"incertidumbre_tornado{suf}.csv", index=False)
    mc.to_csv(dir_res / f"incertidumbre_montecarlo{suf}.csv", index=False)
    filas = []
    for f in ETIQUETAS:
        x = mc[f"inv_f_real_{f:g}"]
        filas.append(dict(etiqueta=f"×1/{1 / f:g}", nominal=1 / f, mediana=x.median(),
                          ic95_inf=x.quantile(.025), ic95_sup=x.quantile(.975),
                          p25=x.quantile(.25), p75=x.quantile(.75), n_nodulos=mc.caso.nunique()))
    res = pd.DataFrame(filas)
    res.to_csv(dir_res / f"incertidumbre_resumen{suf}.csv", index=False)
    # ¿Cambia la incertidumbre de un nódulo a otro? (si sí, conviene reportarla por grupo)
    pc = mc.groupby("caso")["inv_f_real_0.25"].agg(mediana="median", ic95_inf=lambda v: v.quantile(.025),
                                                   ic95_sup=lambda v: v.quantile(.975))
    pc.to_csv(dir_res / f"incertidumbre_por_nodulo{suf}.csv")
    origen = {k: ("medido en LIDC (07a)" if v else "PROVISORIO (correr el 07a)") for k, v in fuentes.items()}
    print(f"\n== Dosis REAL de cada etiqueta (anclaje {modo}, {mc.caso.nunique()} nódulos) ==")
    print(f"   error de Φ: {origen['phi']}; error de medir el ruido: {origen['error_ruido']}; "
          f"ruido no cuántico: {origen['no_cuantico']}")
    for _, r in res.iterrows():
        print(f"   {r.etiqueta:7s} -> mediana 1/{r.mediana:.1f};  95 % entre 1/{r.ic95_inf:.1f} y 1/{r.ic95_sup:.1f}"
              f"  (mitad central 1/{r.p25:.1f} a 1/{r.p75:.1f})")
    print(f"   entre nódulos, la mediana de ×1/4 va de 1/{pc.mediana.min():.1f} a 1/{pc.mediana.max():.1f}")
    imp = importancia(mc, "inv_f_real_0.25", modo)
    imp.to_csv(dir_res / f"incertidumbre_importancia{suf}.csv", index=False)
    print(f"\n== Qué explica la incertidumbre de ×1/4, anclaje {modo} (% de la variabilidad, R² de rangos) ==")
    print(imp.round(1).to_string(index=False))
    figura(res, tor, imp, C.DIR_FIGURAS / "comparaciones" / f"incertidumbre_dosis{suf}.png")
    return res


def comparar_modos(resumenes, ruta):
    """Un gráfico de barras: para cada etiqueta, la dosis real con su IC 95 %
    según cada anclaje. El mejor anclaje es el de intervalo más angosto Y
    centrado en la etiqueta."""
    nombres = {"mas": "mAs × Φ (paper)", "ruido": "ruido de la imagen", "hibrido": "híbrido"}
    colores = {"mas": "#4a7ab5", "ruido": "#f0ad4e", "hibrido": "#5cb85c"}
    fig, ax = plt.subplots(figsize=(9, 5))
    etiquetas = next(iter(resumenes.values())).etiqueta.tolist()
    x = np.arange(len(etiquetas))
    w = 0.8 / (len(resumenes) + 1)
    ref = next(iter(resumenes.values())).nominal
    ax.bar(x - 0.4 + w / 2, ref, w, color="#bbbbbb", label="lo que dice la etiqueta")
    filas = []
    for i, (m, r) in enumerate(resumenes.items(), 1):
        ax.bar(x - 0.4 + w / 2 + i * w, r.mediana, w, color=colores[m], label=nombres[m],
               yerr=[r.mediana - r.ic95_inf, r.ic95_sup - r.mediana], capsize=3)
        for _, f in r.iterrows():
            filas.append(dict(anclaje=m, etiqueta=f.etiqueta, mediana=f.mediana, ic95_inf=f.ic95_inf,
                              ic95_sup=f.ic95_sup, ancho_relativo=f.ic95_sup / f.ic95_inf,
                              sesgo_relativo=f.mediana / f.nominal - 1))
    ax.set_xticks(x, etiquetas)
    ax.set_yscale("log")
    ax.set_ylabel("1 / fracción de dosis real (mediana e IC 95 %)")
    ax.set_title("¿Qué anclaje deja menos incertidumbre sobre la dosis real?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    t = pd.DataFrame(filas)
    t.to_csv(C.DIR_RESULTADOS / "comparaciones" / "incertidumbre_anclajes.csv", index=False)
    print("\n== Comparación de anclajes: ancho del IC 95 % (máx/mín; 1 = sin incertidumbre) y sesgo de la mediana ==")
    print(t.pivot(index="etiqueta", columns="anclaje", values="ancho_relativo").round(2).to_string())
    print(t.pivot(index="etiqueta", columns="anclaje", values="sesgo_relativo").mul(100).round(0).to_string())
    print("->", ruta)


def importancia(mc, col, modo="mas"):
    """Correlación de rangos de Spearman al cuadrado entre cada parámetro y
    log(1/f_real), normalizada para sumar 100 %. Es una medida simple de
    'cuánto de la variabilidad va con este parámetro' (los parámetros se
    sortean independientes, así que no se pisan entre sí)."""
    y = np.log(mc[col])
    filas = []
    for p in PARAMETROS:
        if p in NO_APLICA[modo]:
            continue
        x = mc[p]
        if x.nunique() < 2:
            continue
        rho = stats.spearmanr(x, y).statistic
        filas.append(dict(parametro=p, descripcion=PARAMETROS[p][4], rho=rho, r2=rho ** 2))
    t = pd.DataFrame(filas)
    t["porcentaje"] = 100 * t.r2 / t.r2.sum()
    return t.sort_values("porcentaje", ascending=False)[["parametro", "descripcion", "rho", "porcentaje"]]


def figura(res, tor, imp, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1, 1.4, 1.2]))
    # a. barras: etiqueta vs dosis real (intervalo 95 %)
    x = np.arange(len(res))
    ax[0].bar(x - 0.2, res.nominal, 0.38, label="lo que dice la etiqueta", color="#bbbbbb")
    ax[0].bar(x + 0.2, res.mediana, 0.38, yerr=[res.mediana - res.ic95_inf, res.ic95_sup - res.mediana],
              capsize=4, label="dosis real (mediana e IC 95 %)", color="#4a7ab5")
    ax[0].set_xticks(x, res.etiqueta)
    ax[0].set_ylabel("1 / fracción de dosis  (4 = un cuarto)")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=7)
    ax[0].set_title("a. ¿A qué dosis corresponde cada etiqueta?")
    # b. tornado para ×1/4
    col = "inv_f_real_0.25"
    ref = float(tor["ref_" + col].iloc[0])
    g = tor.groupby(["parametro", "descripcion"])[col].agg(["min", "max"]).reset_index()
    g["rango"] = g["max"] - g["min"]
    g = g.sort_values("rango")
    y = np.arange(len(g))
    # una barra de min a max: dónde puede caer la dosis real si SOLO este parámetro está mal
    ax[1].barh(y, g["rango"], left=g["min"], color="#d9534f", alpha=0.8)
    ax[1].axvline(ref, color="k", lw=1, ls="--", label=f"con los valores por defecto (1/{ref:.1f})")
    ax[1].axvline(4, color="gray", lw=0.8, ls=":", label="lo que dice la etiqueta (1/4)")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_yticks(y, g.descripcion, fontsize=7)
    ax[1].set_xlabel("1 / fracción real de \"×1/4\" con el parámetro en su mínimo y en su máximo")
    ax[1].set_title("b. Uno por vez: cuánto mueve cada parámetro")
    # c. importancia en el Monte Carlo
    ax[2].barh(imp.descripcion[::-1], imp.porcentaje[::-1], color="#5cb85c")
    ax[2].set_xlabel("% de la variabilidad de la dosis real de ×1/4")
    ax[2].tick_params(axis="y", labelsize=7)
    ax[2].set_title("c. Todos a la vez: de dónde\nviene la incertidumbre")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("->", ruta)


if __name__ == "__main__":
    main()
