"""Etapa 7a - Calibrar y validar el anclaje de la dosis con la propia LIDC.

Naziroglu et al. (2017) calibran el rendimiento del tubo, el bowtie, el ruido
de lectura y la ventana del kernel escaneando un cilindro de agua en el MISMO
equipo a varias corrientes. En LIDC no hay esos barridos, pero hay algo
parecido: ~cien estudios de distintos equipos, cada uno con su mAs en el
encabezado y su ruido medible en la imagen. Si la física de la simulación es
correcta, la "constante del tubo" Φ (fotones por mAs y por mm²) que explica el
ruido de cada estudio tiene que salir PARECIDA en todos (dentro de un mismo
fabricante): la dispersión de Φ mide cuánto falla el modelo.

Para cada corte de trabajo:
  σ0      ruido medido en la imagen (ruido global de Christianson et al.
          2015: moda del desvío local en tejido blando);
  σ_ref   ruido que da la simulación con N_ref fotones por rayo, con el
          mismo paciente (sinograma virtual), la misma ventana de kernel y el
          mismo estimador;
  N       = N_ref (σ_ref/σ0)²     fotones que explican ese ruido (σ ∝ 1/√N);
  Φ_i     = N / [mAs (kVp/120)^a Δs espesor / vistas]   constante del tubo.

Calibración: Φ = mediana de Φ_i en los estudios de MÁS dosis (tercio superior
de mAs), por fabricante si hay suficientes. Validación (lo que pide la
devolución): con ese Φ y el mAs del encabezado, se PREDICE el ruido de los
estudios de LIDC que ya son de baja dosis (tercio inferior) y se compara con
el medido. Se predice con y sin ruido electrónico: si sin él la predicción
se queda corta en los de baja dosis, el ruido electrónico importa.

Uso:  python scripts/07a_calibrar_dosis.py [--procesos N] [--limite N]
Salidas:
  resultados/07a_calibracion.csv   una fila por nódulo
  resultados/07a_constante.csv     Φ por fabricante (lo lee el 07)
  resultados/07a_estudios.csv      por estudio: Φ_i, k = Φ_fab/Φ_i, fracción real de cada etiqueta,
                                   error de medición del ruido
  resultados/07a_no_cuantico.csv   parte del ruido que no baja con la dosis, por fabricante (con IC 95 %)
  resultados/07a_dispersion_por_modelo.csv  dispersión de Φ dentro de cada modelo vs error de medición
  figuras/07a_calibracion.png
"""
import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import baja_dosis as B  # noqa: E402
from tpi import config as C  # noqa: E402
from tpi.datos import cargar_corte  # noqa: E402
from tpi.encabezados import familia_kernel, mas_efectivo, sff_bowtie  # noqa: E402

PATRON = re.compile(r"(?P<paciente>.+)_s(?P<scan_id>\d+)_n(?P<nodulo>\d+)$")
N_REF = 1e5
MIN_ESTUDIOS_FABRICANTE = 8


def procesar(args):
    ruta, enc = args
    d = cargar_corte(Path(ruta))
    px = float(d["pixel_mm_original"])
    adq = B.Adquisicion(d["corte_completo"], px)
    adq.ventana = enc["ventana_fbp"]
    adq.usar_bowtie(modelo="kramer", sff=sff_bowtie(enc.get("fabricante"), enc.get("modelo"))[0])
    tejido = B.mascara_tejido_blando(adq.corte, px)
    s0 = B.ruido_global(adq.corte, px, tejido)
    fila = dict(paciente=enc["paciente"], scan_id=enc["scan_id"], nodulo=enc["nodulo"],
                fabricante=enc["fabricante"], modelo=enc.get("modelo"), kernel=enc["kernel"],
                familia_kernel=enc["familia_kernel"],
                error_ruido_rel=B.error_medicion_ruido(adq.corte, px, tejido),
                ventana_fbp=enc["ventana_fbp"], kvp=enc["kvp"], mas=enc["mas_efectivo"],
                espesor_mm=enc["espesor_mm"], px_mm=px, pixeles_tejido=int(tejido.sum()), sigma0_hu=s0)
    if not np.isfinite(s0) or not np.isfinite(enc["mas_efectivo"]):
        return fila
    s_ref = adq.sigma_ruido(N_REF, tejido, rng=np.random.default_rng(C.SEMILLA))
    esp = enc["espesor_mm"] if np.isfinite(enc["espesor_mm"]) else C.MAX_ESPESOR_MM
    g = B.fotones_por_rayo(enc["mas_efectivo"], enc["kvp"], px, esp, adq.n_vistas, phi=1.0)
    N = N_REF * (s_ref / s0) ** 2
    e_ref = ruido_electronico_unitario(adq, esp, tejido)
    fila.update(sigma_ref_hu=s_ref, fotones_por_rayo=N, phi_i=N / g, geom=g, e_ref_hu2=e_ref)
    return fila


def ruido_electronico_unitario(adq, espesor_mm, tejido):
    """Varianza (HU², con el mismo estimador que el ruido medido) que aporta
    el ruido electrónico con σe² = 1 por lectura y N_REF fotones en el centro
    del bowtie, para ESTE paciente y este equipo. Se usa la aproximación lineal
    del logaritmo (Var(p) = σe² / cuentas²), que es la misma que hace la
    compensación. Como la varianza electrónica en la imagen es ∝ σe² / N²,
    con cualquier otro σe² y N alcanza con escalar este número."""
    L = B.lecturas_por_rayo(adq.px, espesor_mm, adq.n_vistas)
    N = (N_REF * np.asarray(adq.bowtie, float)).reshape(-1, 1)
    cuentas = np.maximum(N * np.exp(-adq.p), C.CUENTAS_MIN)
    rng = np.random.default_rng(C.SEMILLA + 7)
    ruido_p = rng.normal(0.0, np.sqrt(L), adq.p.shape) / cuentas
    r = B.reconstruir_ruido(ruido_p, adq.px, adq.corte.shape[0], adq.ventana)
    return B.ruido_global(r, adq.px, tejido) ** 2


def diagnostico_exposicion(enc):
    """Para el informe: por qué no se usan los mAs del DICOM. Muestra, por
    fabricante, qué traen Exposure, ExposureTime y la corriente."""
    est = enc.drop_duplicates("scan_id")
    cols = [c for c in ("exposicion_mas", "tiempo_exposicion_ms", "corriente_ma") if c in est]
    t = est.groupby("fabricante")[cols].describe(percentiles=[.5])
    t = t.loc[:, (slice(None), ["count", "min", "50%", "max"])]
    print("\n== Tags de exposición por fabricante (por qué se ancla con la corriente, mA) ==")
    print(t.to_string(float_format=lambda v: f"{v:.4g}"))


def errores_del_ruido(df, est):
    """Los dos errores del ruido medido, cuantificados con LIDC.

    1. Error de MEDICIÓN en cada imagen (mitad izquierda vs derecha del tejido
       blando, ver B.error_medicion_ruido). Como fotones ∝ 1/σ², un error
       relativo e en σ es un error ~2e en los fotones.

    2. Ruido que NO es cuántico (estructura del tejido, artefactos): no baja
       con la dosis. Con y = (σ0/σ_ref)² (ruido medido relativo al que daría la
       simulación con N_ref fotones para ESE paciente, así el tamaño del
       paciente se cancela) y x = 1/(mAs × geometría), la física dice
       y = pendiente·x + b: la pendiente es el ruido cuántico, b el que no baja
       con la dosis. Se ajusta por fabricante (y global) con bootstrap por
       estudio; la fracción no cuántica de cada estudio es b / y.

    3. ¿Promediar (Φ por modelo) o anclar cada estudio con su ruido? Si la
       dispersión de Φ_i dentro de un MISMO modelo de tomógrafo es la que da
       el error de medición (~2e), los equipos son iguales y promediar ayuda;
       si es mucho mayor, los equipos o protocolos difieren y conviene anclar
       por el ruido. Se reporta el cociente, no se decide a ojo."""
    e = df.error_ruido_rel.dropna()
    print(f"\n== Error de medición del ruido en cada imagen (mitades del tejido blando) ==")
    print(f"   mediana {100 * e.median():.1f} %, P90 {100 * e.quantile(.9):.1f} %  "
          f"(literatura, Christianson 2015: 3,4-4,7 %) -> en fotones ~{200 * e.median():.0f} %")

    est["x"] = 1 / est.geom
    est["y"] = (est.sigma0_hu / est.sigma_ref_hu) ** 2
    rng = np.random.default_rng(0)
    filas = []
    for fab, sub in [("TODOS", est)] + [(f, g) for f, g in est.groupby("fabricante")
                                         if len(g) >= MIN_ESTUDIOS_FABRICANTE]:
        sub = sub.dropna(subset=["x", "y"])
        if len(sub) < 5:
            continue
        def ajuste(t):
            A = np.column_stack([t.x, np.ones(len(t))])
            return np.linalg.lstsq(A, t.y.to_numpy(), rcond=None)[0]
        pend, b = ajuste(sub)
        boots = np.array([ajuste(sub.iloc[rng.integers(0, len(sub), len(sub))]) for _ in range(1000)])
        b_lo, b_hi = np.percentile(boots[:, 1], [2.5, 97.5])
        frac = np.clip(b / sub.y, 0, 1)
        filas.append(dict(fabricante=fab, n_estudios=len(sub), b=b, b_ic95_inf=b_lo, b_ic95_sup=b_hi,
                          frac_no_cuantica_mediana=float(np.median(frac)),
                          frac_no_cuantica_ic95_inf=float(np.clip(np.median(b_lo / sub.y), 0, 1)),
                          frac_no_cuantica_ic95_sup=float(np.clip(np.median(b_hi / sub.y), 0, 1))))
    nq = pd.DataFrame(filas)
    nq.to_csv(C.DIR_RESULTADOS / "07a_no_cuantico.csv", index=False)
    print("\n== Ruido que no baja con la dosis (σ0² = cuántico/mA + b) ==")
    print("   fracción del ruido medido que NO es cuántica (mediana entre estudios, IC 95 % por bootstrap):")
    print(nq.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    filas = []
    for (fab, mod), sub in est.groupby(["fabricante", "modelo"], dropna=False):
        if len(sub) < 3:
            continue
        disp = float(np.std(np.log(sub.phi_i), ddof=1))          # dispersión relativa de Φ_i
        esperada = float(2 * sub.error_ruido_rel.median())        # la que daría solo el error de medición
        filas.append(dict(fabricante=fab, modelo=mod, n_estudios=len(sub), dispersion_phi=disp,
                          dispersion_esperada_por_medicion=esperada,
                          cociente=disp / esperada if esperada > 0 else np.nan))
    dm = pd.DataFrame(filas)
    dm.to_csv(C.DIR_RESULTADOS / "07a_dispersion_por_modelo.csv", index=False)
    if len(dm):
        print("\n== Φ dentro de cada modelo de tomógrafo: dispersión observada vs la que daría solo medir mal el ruido ==")
        print(dm.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        c = dm.cociente.median()
        if not np.isfinite(c):
            print("   no se puede calcular (error de medición nulo o faltante)")
            return
        print(f"   cociente mediano {c:.1f}: "
              + ("≈ 1 -> los equipos de un modelo son parecidos; el anclaje por corriente × Φ (promedia) no pierde nada"
                 if c < 1.5 else "> 1 -> los estudios de un mismo modelo difieren de verdad; el anclaje por "
                 "ruido/híbrido evita ese error (ver incertidumbre_dosis.py para el intervalo de cada uno)"))


def predecir(args):
    """Ruido que predice el modelo con Φ calibrado, con y sin electrónico."""
    ruta, enc, phi = args
    d = cargar_corte(Path(ruta))
    px = float(d["pixel_mm_original"])
    out = {}
    for elec in (False, True):
        adq = B.Adquisicion(d["corte_completo"], px).desde_dicom(enc, phi=phi, electronico=elec)
        tejido = B.mascara_tejido_blando(adq.corte, px)
        r = adq.ruido_original(np.random.default_rng(C.SEMILLA))
        out["sigma_pred_" + ("con_e" if elec else "sin_e")] = B.ruido_global(r, px, tejido)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limite", type=int, default=0)
    args = ap.parse_args()
    enc = pd.read_csv(C.DIR_RESULTADOS / "encabezados_dicom.csv")
    enc["fabricante"] = enc.fabricante.fillna("?").str.split().str[0].str.upper()
    # Índice de dosis: la corriente del corte (ver encabezados.mas_efectivo).
    # Se recalcula acá por si el CSV del 03b tiene la columna vieja (mAs).
    enc["mas_efectivo"] = [mas_efectivo(r) for r in enc.to_dict("records")]
    # Familia y ventana del kernel: se recalculan con la regla ACTUAL (el CSV
    # guarda las del momento en que se leyeron los encabezados).
    fam = [familia_kernel(k) for k in enc.kernel]
    enc["familia_kernel"] = [f for f, _ in fam]
    enc["ventana_fbp"] = [v for _, v in fam]
    diagnostico_exposicion(enc)
    trabajos = []
    for _, e in enc.iterrows():
        ruta = Path(args.carpeta) / f"{e.paciente}_s{e.scan_id}_n{e.nodulo}.npz"
        if ruta.exists():
            trabajos.append((str(ruta), e.to_dict()))
    if args.limite:
        trabajos = trabajos[:args.limite]
    print(f"Calibrando con {len(trabajos)} cortes ({args.procesos} procesos)...")
    with ProcessPoolExecutor(args.procesos) as ex:
        df = pd.DataFrame(list(ex.map(procesar, trabajos)))

    # Un valor por estudio (los nódulos del mismo estudio comparten dosis)
    ok = df.dropna(subset=["phi_i"])
    est = ok.groupby("scan_id").agg(fabricante=("fabricante", "first"), modelo=("modelo", "first"),
                                    mas=("mas", "median"), geom=("geom", "median"),
                                    sigma_ref_hu=("sigma_ref_hu", "median"),
                                    error_ruido_rel=("error_ruido_rel", "median"),
                                    phi_i=("phi_i", "median"), sigma0_hu=("sigma0_hu", "median"),
                                    familia_kernel=("familia_kernel", "first"),
                                    e_ref_hu2=("e_ref_hu2", "median")).reset_index()
    q_lo, q_hi = est.mas.quantile([1 / 3, 2 / 3])
    est["rol"] = np.where(est.mas >= q_hi, "calibración", np.where(est.mas <= q_lo, "validación", "-"))
    print(f"\nEstudios con corriente y ruido medible: {len(est)}  |  calibración: ≥ {q_hi:.0f} mA, "
          f"validación (baja dosis de LIDC): ≤ {q_lo:.0f} mA")
    describir_grupos(est, df)

    cal = est[est.rol == "calibración"]
    phi_global = float(cal.phi_i.median())
    constantes = [dict(fabricante="TODOS", phi=phi_global, n_estudios=len(cal),
                       iqr_relativo=float(cal.phi_i.quantile(.75) / cal.phi_i.quantile(.25)))]
    for fab, sub in cal.groupby("fabricante"):
        if len(sub) >= MIN_ESTUDIOS_FABRICANTE:
            constantes.append(dict(fabricante=fab, phi=float(sub.phi_i.median()), n_estudios=len(sub),
                                   iqr_relativo=float(sub.phi_i.quantile(.75) / sub.phi_i.quantile(.25))))
    cons = pd.DataFrame(constantes)
    cons.to_csv(C.DIR_RESULTADOS / "07a_constante.csv", index=False)
    print("\nConstante del tubo Φ (fotones / (mA·mm²), absorbe tiempo de rotación y pitch); iqr_relativo = P75/P25 (1 = todos iguales):")
    print(cons.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"(valor inicial de config: {C.FOTONES_POR_MAS_MM2:.2g})")
    print("\nΦ_i por familia de kernel (si la ventana elegida para un kernel estuviera mal, se nota acá):")
    print(est.groupby("familia_kernel").phi_i.describe()[["count", "25%", "50%", "75%"]]
          .to_string(float_format=lambda v: f"{v:.3g}"))

    # Error de usar UN Φ por fabricante en vez del de cada estudio. Si el
    # anclaje por el ruido de cada estudio fuera la verdad, usar Φ_fab es
    # equivocarse por k_i = Φ_fab / Φ_i, y la etiqueta f simula de verdad
    # f_real = 1 / (1 + (1/f - 1) / k_i)   (ver comparaciones/sensibilidad_dosis.py)
    phi_fab = est.fabricante.map(dict(zip(cons.fabricante, cons.phi))).fillna(phi_global)
    est["k_phi"] = phi_fab / est.phi_i
    print("\nAnclaje por corriente × Φ del fabricante vs. por el ruido de cada estudio:")
    print(f"   k = Φ_fab/Φ_i: mediana {est.k_phi.median():.2f}, P25 {est.k_phi.quantile(.25):.2f}, "
          f"P75 {est.k_phi.quantile(.75):.2f}")
    for f in C.FRACCIONES_DOSIS:
        fr = 1 / (1 + (1 / f - 1) / est.k_phi)
        est[f"f_real_{f:g}"] = fr
        print(f"   etiqueta 1/{1 / f:g}: en la mitad central de los estudios es de verdad entre "
              f"1/{1 / fr.quantile(.75):.1f} y 1/{1 / fr.quantile(.25):.1f}")
    errores_del_ruido(df, est)
    estimar_electronico(est)
    est.to_csv(C.DIR_RESULTADOS / "07a_estudios.csv", index=False)

    # Validación en los estudios de baja dosis de LIDC
    phi_de = dict(zip(cons.fabricante, cons.phi))
    val_ids = set(est.scan_id[est.rol == "validación"])
    val = [(r, e, phi_de.get(e["fabricante"], phi_global)) for r, e in trabajos
           if e["scan_id"] in val_ids and np.isfinite(e["mas_efectivo"])]
    with ProcessPoolExecutor(args.procesos) as ex:
        pred = list(ex.map(predecir, val))
    pv = pd.DataFrame([dict(scan_id=e["scan_id"], nodulo=e["nodulo"], **p) for (_, e, _), p in zip(val, pred)])
    df = df.merge(pv, on=["scan_id", "nodulo"], how="left")
    df["rol"] = df.scan_id.map(dict(zip(est.scan_id, est.rol)))
    df.to_csv(C.DIR_RESULTADOS / "07a_calibracion.csv", index=False)
    v = df[df.rol == "validación"].dropna(subset=["sigma_pred_sin_e"])
    if len(v):
        for col, nombre in (("sigma_pred_sin_e", "sin electrónico"), ("sigma_pred_con_e", "con electrónico")):
            err = (v[col] - v.sigma0_hu) / v.sigma0_hu
            print(f"\nValidación ({len(v)} cortes de baja dosis), {nombre}: error relativo mediano "
                  f"{100 * err.median():+.0f} %  (P25 {100 * err.quantile(.25):+.0f} %, P75 {100 * err.quantile(.75):+.0f} %)")
    figura(df, est, phi_global)


def describir_grupos(est, df):
    """Qué estudios son los de calibración y cuáles los de validación (cuántos,
    de qué fabricante, con qué corriente). Son tomografías REALES de LIDC: los
    de validación se adquirieron con menos corriente, no son una simulación."""
    cortes = df.dropna(subset=["phi_i"]).groupby("scan_id").size()
    t = est.assign(cortes=est.scan_id.map(cortes)).query("rol != '-'")
    tab = t.groupby(["rol", "fabricante"]).agg(estudios=("scan_id", "size"), cortes=("cortes", "sum"),
                                               ma_min=("mas", "min"), ma_mediana=("mas", "median"),
                                               ma_max=("mas", "max"))
    print("\n== Estudios de calibración y de validación (tomografías reales de LIDC) ==")
    print(tab.to_string(float_format=lambda v: f"{v:.0f}"))
    tot = t.groupby("rol").agg(estudios=("scan_id", "size"), cortes=("cortes", "sum"))
    print(tot.to_string())


def _ajuste_electronico(t, fabs):
    """σ0² = Σ_f 1{f}·u·σ_ref²/Φ_f + σe²·u²·e_ref/Φ_f² + b, con u = N_REF/geom.
    Parámetros: Φ_f por fabricante, σe² por lectura (común) y b (ruido que no
    baja con la dosis, HU²). Residuos en logaritmo: la diferencia de Φ entre
    equipos (que el 07a mide grande) es multiplicativa, y en un ensayo con datos
    sintéticos de verdad conocida el ajuste en log no tiene sesgo y el de
    residuos relativos sí (subestima σe² cuando Φ varía entre equipos)."""
    from scipy.optimize import least_squares
    y = t.sigma0_hu.to_numpy() ** 2
    u = (N_REF / t.geom).to_numpy()
    q = u * t.sigma_ref_hu.to_numpy() ** 2
    e = u ** 2 * t.e_ref_hu2.to_numpy()
    idx = np.array([fabs.index(f) for f in t.fabricante])

    def modelo(par):
        inv_phi = np.exp(par[:len(fabs)])[idx]
        se2, b = par[len(fabs)], par[len(fabs) + 1]
        return q * inv_phi + se2 * e * inv_phi ** 2 + b

    x0 = np.r_[np.full(len(fabs), np.log(1 / 3e5)), 10.0, 0.1 * np.median(y)]
    lo = np.r_[np.full(len(fabs), -np.inf), 0.0, 0.0]
    r = least_squares(lambda p: np.log(np.maximum(modelo(p), 1e-9)) - np.log(y), x0,
                      bounds=(lo, np.inf), x_scale="jac")
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def estimar_electronico(est):
    """El ruido electrónico estimado con los propios datos de LIDC.

    Cada fuente de ruido cambia distinto con la corriente: el cuántico baja
    como 1/corriente, el electrónico como 1/corriente² y el que no es cuántico
    no baja. Con todos los estudios (de ~40 a ~570 mA) se ajustan las tres
    partes a la vez, así el ruido no cuántico no se confunde con el
    electrónico. IC 95 % por bootstrap de estudios. Se compara además el
    ajuste con σe² fijo en 0 y en 30 (el valor de literatura)."""
    t = est.dropna(subset=["sigma0_hu", "sigma_ref_hu", "geom", "e_ref_hu2"])
    t = t[t.e_ref_hu2 > 0]
    cuenta = t.fabricante.value_counts()
    fabs = sorted(cuenta[cuenta >= MIN_ESTUDIOS_FABRICANTE].index)
    t = t[t.fabricante.isin(fabs)]
    if len(t) < 20:
        print("\n== Ruido electrónico: muy pocos estudios para estimarlo ==")
        return
    par, rms = _ajuste_electronico(t, fabs)
    se2, b = par[len(fabs)], par[len(fabs) + 1]
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(500):
        s = t.iloc[rng.integers(0, len(t), len(t))]
        if s.fabricante.nunique() < len(fabs):
            continue
        boots.append(_ajuste_electronico(s, fabs)[0][len(fabs):])
    boots = np.array(boots)
    lo, hi = np.percentile(boots[:, 0], [2.5, 97.5])
    b_lo, b_hi = np.percentile(boots[:, 1], [2.5, 97.5])
    # Peso del electrónico en el ruido de cada estudio, con lo estimado
    u = (N_REF / t.geom).to_numpy()
    inv_phi = np.exp(par[[fabs.index(f) for f in t.fabricante]])
    q = u * t.sigma_ref_hu.to_numpy() ** 2 * inv_phi
    e = se2 * u ** 2 * t.e_ref_hu2.to_numpy() * inv_phi ** 2
    frac_e = e / (q + e + b)
    print("\n== Ruido electrónico estimado con LIDC (σ0² = cuántico/I + electrónico/I² + no cuántico) ==")
    print(f"   {len(t)} estudios ({', '.join(f'{f} {cuenta[f]}' for f in fabs)}), de {t.mas.min():.0f} a "
          f"{t.mas.max():.0f} mA; {len(boots)} remuestreos")
    print(f"   σe² por lectura: {se2:.1f}  (IC 95 % {lo:.1f} - {hi:.1f});  valor de literatura usado: 30")
    print(f"   ruido no cuántico b: {np.sqrt(b):.1f} HU de desvío (IC 95 % {np.sqrt(b_lo):.1f} - {np.sqrt(b_hi):.1f})")
    print(f"   peso del electrónico en el ruido de cada estudio: mediana {100 * np.median(frac_e):.1f} %, "
          f"máximo {100 * frac_e.max():.1f} % (en el de menos corriente)")
    from scipy.optimize import least_squares as ls
    for fijo in (0.0, 30.0):
        y = t.sigma0_hu.to_numpy() ** 2
        idx = np.array([fabs.index(f) for f in t.fabricante])
        qq = u * t.sigma_ref_hu.to_numpy() ** 2
        ee = u ** 2 * t.e_ref_hu2.to_numpy()

        def res(p):
            ip = np.exp(p[:len(fabs)])[idx]
            return np.log(np.maximum(qq * ip + fijo * ee * ip ** 2 + p[-1], 1e-9)) - np.log(y)
        r = ls(res, np.r_[par[:len(fabs)], b], bounds=(np.r_[np.full(len(fabs), -np.inf), 0], np.inf))
        print(f"   ajuste con σe² fijo en {fijo:g}: error RMS {100 * np.sqrt(np.mean(r.fun ** 2)):.1f} %"
              f"  (con σe² estimado: {100 * rms:.1f} %)")
    if hi < 30:
        print("   -> 30 queda FUERA del intervalo: el valor de literatura es alto para estos equipos.")
    elif lo > 30:
        print("   -> 30 queda por DEBAJO del intervalo: estos equipos tenían más ruido electrónico.")
    else:
        print("   -> 30 queda dentro del intervalo: los datos no permiten descartarlo.")
    pd.DataFrame([dict(sigma_e2=se2, ic95_inf=lo, ic95_sup=hi, b_hu2=b, b_ic95_inf=b_lo, b_ic95_sup=b_hi,
                       n_estudios=len(t), rms_relativo=rms, frac_electronico_mediana=float(np.median(frac_e)),
                       frac_electronico_max=float(frac_e.max()),
                       **{f"phi_{f}": float(1 / np.exp(par[i])) for i, f in enumerate(fabs)})]) \
        .to_csv(C.DIR_RESULTADOS / "07a_electronico.csv", index=False)


def figura(df, est, phi):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for (fam, sub), m in zip(est.groupby("familia_kernel"), "osd^"):
        ax[0].loglog(sub.mas, sub.sigma0_hu, m, label=fam, alpha=0.7)
    x = np.geomspace(max(est.mas.min(), 1), est.mas.max(), 50)
    ax[0].loglog(x, est.sigma0_hu.median() * np.sqrt(est.mas.median() / x), "k--", lw=1, label="∝ 1/√mA")
    ax[0].set_xlabel("corriente del tubo, mA (DICOM)")
    ax[0].set_ylabel("ruido medido σ0 (HU)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("a. Ruido real de LIDC vs corriente")

    grupos = [g.phi_i.to_numpy() for _, g in est.groupby("fabricante")]
    ax[1].boxplot(grupos, tick_labels=[f for f, _ in est.groupby("fabricante")])
    ax[1].axhline(phi, color="k", ls="--", lw=1)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Φ_i (fotones / mA·mm²)")
    ax[1].set_title("b. Constante del tubo por estudio")

    v = df[df.rol == "validación"].dropna(subset=["sigma_pred_sin_e"])
    if len(v):
        ax[2].plot(v.sigma0_hu, v.sigma_pred_sin_e, "o", alpha=0.6, label="sin electrónico")
        ax[2].plot(v.sigma0_hu, v.sigma_pred_con_e, "s", alpha=0.6, label="con electrónico")
        lim = [0, float(np.nanmax(v[["sigma0_hu", "sigma_pred_sin_e", "sigma_pred_con_e"]].to_numpy())) * 1.05]
        ax[2].plot(lim, lim, "k-", lw=0.8)
        ax[2].legend(fontsize=7)
    ax[2].set_xlabel("ruido medido (HU)")
    ax[2].set_ylabel("ruido predicho con Φ y la corriente (HU)")
    ax[2].set_title("c. Validación en los estudios de baja dosis")
    fig.tight_layout()
    C.DIR_FIGURAS.mkdir(exist_ok=True)
    ruta = C.DIR_FIGURAS / "07a_calibracion.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("->", ruta)


if __name__ == "__main__":
    main()
