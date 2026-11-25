"""Interfaz gráfica del TPI: recorre el pipeline completo sobre un nódulo.

Cada pestaña es una etapa del trabajo y usa EXACTAMENTE las funciones del
pipeline (paquete tpi/ y scripts), así lo que se ve es lo que calcula el TP:

  1. Imagen        corte, histograma de HU y ruido de la imagen (análisis)
  2. Reconstrucción Radon, sinograma, SBP / FBP (rampa, Hann) / SART
  3. Filtros       mediana, gaussiano y Wiener sobre una imagen con ruido
  4. Segmentación  crecimiento de regiones paso a paso, Dice, Otsu
  5. Características forma, intensidad, GLCM, borde, calcio, entorno
  6. Clasificación puntaje (06) armado sin el paciente de este nódulo
  7. Baja dosis y compresión  simulación de Naziroglu et al. (2017) y JPEG
  8. Resultados    las figuras globales del trabajo

Se abre con abrir_interfaz.py (botón ▶) o con:  streamlit run interfaz/app.py
Necesita datos/cortes/*.npz y resultados/caracteristicas.csv.
"""
import importlib.util
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import morphology  # noqa: E402

from tpi import baja_dosis as B  # noqa: E402
from tpi import compresion as Z  # noqa: E402
from tpi import config as C  # noqa: E402
from tpi import mejoramiento as M  # noqa: E402
from tpi import reconstruccion as R  # noqa: E402
from tpi.caracteristicas import (caracteristicas_interior, fraccion_calcio_robusta,  # noqa: E402
                                 nitidez_borde)
from tpi.datos import _recortar, _remuestrear, cargar_corte  # noqa: E402
from tpi.entorno import anillos, caracteristicas_entorno, mascara_pulmon  # noqa: E402
from tpi.estadistica import META  # noqa: E402
from tpi.puntaje import Puntaje, auc  # noqa: E402
from tpi.segmentacion import (_componente_con_semilla, region_permitida,  # noqa: E402
                              segmentar_nodulo, segmentar_otsu)
from tpi.validacion import concordancia_radiologos, dice, evaluar  # noqa: E402

VENTANA = dict(cmap="gray", vmin=-1000, vmax=400)
CLAVES = ["paciente", "scan_id", "nodulo"]
TAMANO = {"area_mm2", "diametro_eq_mm"}
ENTORNO_EXTRA = {"frac_contacto_no_pulmon", "esqueleto_mm_por_mm_borde", "n_estructuras_convergentes",
                 "convergentes_por_10mm_borde"}
FRACCIONES = {"×1/2": 0.5, "×1/4": 0.25, "×1/10": 0.1, "×1/20": 0.05}

NOMBRES = {  # nombre legible de cada medida
    "area_mm2": "Área (mm²)", "diametro_eq_mm": "Diámetro equivalente (mm)",
    "circularidad": "Circularidad", "solidez": "Solidez", "excentricidad": "Excentricidad",
    "circularidad_cruda": "Circularidad sin apertura", "solidez_cruda": "Solidez sin apertura",
    "frac_fina": "Fracción fina (espículas)", "nitidez_borde_hu_mm": "Nitidez del borde (HU/mm)",
    "hu_media": "HU media", "hu_desvio": "Desvío de HU", "hu_asimetria": "Asimetría de HU",
    "hu_curtosis": "Curtosis de HU", "frac_calcio": "Fracción de calcio (>200 HU)",
    "frac_aire": "Fracción de aire (<−600 HU)", "glcm_energia": "GLCM energía",
    "glcm_entropia": "GLCM entropía", "glcm_contraste": "GLCM contraste",
    "glcm_homogeneidad": "GLCM homogeneidad", "glcm_correlacion": "GLCM correlación",
    "glcm_cluster_shade": "GLCM cluster shade", "glcm_cluster_prominence": "GLCM cluster prominence",
    "frac_contacto_no_pulmon": "Contacto con pleura/mediastino",
    "n_estructuras_convergentes": "Estructuras convergentes",
    "convergentes_por_10mm_borde": "Convergentes por 10 mm de borde",
    "esqueleto_mm_por_mm_borde": "Esqueleto del entorno (mm/mm de borde)",
}


# ---------------------------------------------------------------------------
# Carga de datos (con caché: se calcula una vez)
# ---------------------------------------------------------------------------
def _modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@st.cache_resource
def modulo_07():
    """El script 07 se reusa tal cual (anclaje, degradación y re-recorte)."""
    return _modulo("e07_interfaz", RAIZ / "scripts" / "07_baja_dosis.py")


@st.cache_data
def leer_csv(nombre):
    ruta = C.DIR_RESULTADOS / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data
def catalogo():
    car = leer_csv("caracteristicas.csv")
    if car is None:
        return None
    car = car.copy()
    car["archivo"] = [f"{p}_s{int(s)}_n{int(n)}" for p, s, n in zip(car.paciente, car.scan_id, car.nodulo)]
    car = car[[(C.DIR_CORTES / f"{a}.npz").exists() for a in car.archivo]]
    d12 = leer_csv("12_dice_vs_radiologos.csv")
    if d12 is not None:
        car = car.merge(d12[CLAVES + ["dice_vs_cada_radiologo", "aceptable_justo"]], on=CLAVES, how="left")
    return car.reset_index(drop=True)


@st.cache_data
def parametros_segmentacion():
    t = leer_csv("04_ajuste_umbral.csv")
    if t is not None and "radio_cierre_mm" in t:
        m = t.loc[t.dice_medio.idxmax()]
        return int(m["umbral_hu"]), float(m["radio_cierre_mm"])
    return C.UMBRAL_HU, C.RADIO_CIERRE_PULMON_MM


def cargar(archivo):
    # no se cachea con st.cache_data: el pipeline guarda cachés dentro del dict
    if st.session_state.get("_archivo") != archivo:
        st.session_state["_d"] = cargar_corte(C.DIR_CORTES / f"{archivo}.npz")
        st.session_state["_archivo"] = archivo
    return st.session_state["_d"]


def encabezado(fila):
    t = leer_csv("encabezados_dicom.csv")
    if t is None:
        return None
    m = t[(t.scan_id == fila.scan_id) & (t.nodulo == fila.nodulo)]
    return m.iloc[0].to_dict() if len(m) else None


def sigma_e2_lidc():
    t = leer_csv("07a_electronico.csv")
    if t is None:
        return None, None
    return float(t.sigma_e2.iloc[0]), float(t.ic95_sup.iloc[0])


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def contorno(ax, m, color, lw=1.4, ls="-"):
    if m is not None and m.any():
        ax.contour(m.astype(float), [0.5], colors=color, linewidths=lw, linestyles=ls)


def figura(n, alto=3.6, ancho=None):
    fig, ax = plt.subplots(1, n, figsize=(ancho or 3.6 * n, alto))
    return fig, np.atleast_1d(ax)


def mostrar(fig):
    fig.tight_layout()
    if os.environ.get("TPI_GUARDAR_FIGURAS"):  # para revisar las figuras sin abrir el navegador
        carpeta = Path(os.environ["TPI_GUARDAR_FIGURAS"])
        carpeta.mkdir(parents=True, exist_ok=True)
        fig.savefig(carpeta / f"fig_{len(list(carpeta.glob('*.png'))):02d}.png", dpi=80)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def sin_ejes(axs):
    for a in np.atleast_1d(axs):
        a.axis("off")


def zoom_alrededor(m, margen_px=18, n=None):
    """Recorte cuadrado alrededor de una máscara, para ver el nódulo de cerca."""
    if m is None or not m.any():
        return np.s_[:, :]
    rr, cc = np.nonzero(m)
    r = int(max(np.ptp(rr), np.ptp(cc)) / 2 + margen_px)
    cy, cx = int(rr.mean()), int(cc.mean())
    n = n or m.shape[0]
    return np.s_[max(cy - r, 0):min(cy + r, n), max(cx - r, 0):min(cx + r, n)]


# ---------------------------------------------------------------------------
# Cálculos compartidos entre pestañas
# ---------------------------------------------------------------------------
def segmentar(d, T, Rc):
    seg, cruda = segmentar_nodulo(d, umbral=T, devolver_cruda=True, radio_cierre_mm=Rc)
    return seg, cruda


def pulmon_roi(d):
    p = mascara_pulmon(d["corte_completo_rem"])
    return _recortar(p, tuple(d["semilla_completo"]), d["img"].shape[0], False)


def medir(d, seg, cruda):
    if seg.sum() < 5:
        return {}
    f = caracteristicas_interior(d["img"], seg, mascara_cruda=cruda)
    f["frac_calcio_robusto"] = fraccion_calcio_robusta(d["img"], seg)
    f.update(caracteristicas_entorno(d["img"], seg, pulmon_roi(d))[0])
    return f


def conjuntos_puntaje(car):
    medidas = [c for c in car.select_dtypes("number").columns
               if c not in META and c not in {"dice_vs_cada_radiologo"}]
    entorno = [c for c in medidas if c.startswith("anillo_") or c in ENTORNO_EXTRA]
    interior = [c for c in medidas if c not in entorno]
    return {"diámetro solo": (["diametro_eq_mm"], 1),
            "interior": (interior, 5),
            "interior sin tamaño": ([c for c in interior if c not in TAMANO], 5),
            "interior + entorno": (interior + entorno, 5)}


@st.cache_data
def puntaje_sin_paciente(paciente, nombre_conjunto, k_override=None):
    """Puntaje del 06 armado con los nódulos de desarrollo aceptables, SIN el
    paciente del nódulo elegido (así se lo puntúa como a un caso nuevo)."""
    car = catalogo_completo()
    dev = car[(car.conjunto == "entrenamiento") & car.aceptable & car.grupo.isin(["benigno", "sospechoso"])]
    cols, k = conjuntos_puntaje(car)[nombre_conjunto]
    tr = dev[dev.paciente != paciente]
    y = (tr.grupo == "sospechoso").to_numpy(int)
    modelo = Puntaje(k_override or k).ajustar(tr[cols], y)
    return modelo, cols, tr


@st.cache_data
def catalogo_completo():
    return leer_csv("caracteristicas.csv")


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TPI PSIB · Nódulos pulmonares", layout="wide")
st.title("Biomarcadores de nódulos pulmonares en tomografía")
st.caption("TPI de Procesamiento de Señales e Imágenes Biomédicas (ITBA) · Castex, González Barreiro, Sabatte · "
           "LIDC-IDRI")

cat = catalogo()
if cat is None or cat.empty:
    st.error("No encuentro los datos. Hacen falta `resultados/caracteristicas.csv` (script 04) y los cortes "
             "en `datos/cortes/` (scripts 01 a 03).")
    st.stop()

# ----- Barra lateral: elegir nódulo -----
with st.sidebar:
    st.header("Nódulo")
    grupos = st.multiselect("Grupo", ["benigno", "sospechoso"], default=["benigno", "sospechoso"])
    conjunto = st.radio("Conjunto", ["desarrollo", "evaluación", "todos"], horizontal=True)
    solo_ok = st.checkbox("Solo segmentación aceptable", value=False)
    sub = cat[cat.grupo.isin(grupos)]
    if conjunto != "todos":
        sub = sub[sub.conjunto == ("entrenamiento" if conjunto == "desarrollo" else "evaluacion")]
    if solo_ok:
        sub = sub[sub.aceptable]
    if sub.empty:
        st.warning("Ningún nódulo con esos filtros.")
        st.stop()
    etiquetas = [f"{r.paciente} · n{int(r.nodulo)} · {r.grupo} · {r.diametro_medio_mm:.1f} mm"
                 for r in sub.itertuples()]
    i = st.selectbox(f"Nódulo ({len(sub)})", range(len(sub)), format_func=lambda k: etiquetas[k])
    fila = sub.iloc[i]
    st.divider()
    st.markdown(f"**Grupo (radiólogos):** {fila.grupo}  \n"
                f"**Malignidad (mediana):** {fila.malignidad_mediana:g}  \n"
                f"**Diámetro (radiólogos):** {fila.diametro_medio_mm:.1f} mm  \n"
                f"**Conjunto:** {fila.conjunto}")
    enc = encabezado(fila)
    if enc:
        st.markdown(f"**Equipo:** {enc.get('fabricante', '?')} {enc.get('modelo', '')}  \n"
                    f"**kVp:** {enc.get('kvp', '?')} · **corriente:** {enc.get('corriente_ma', '?')} mA  \n"
                    f"**Kernel:** {enc.get('kernel', '?')} · **espesor:** {enc.get('espesor_mm', '?')} mm")

d = cargar(fila.archivo)
T0, R0 = parametros_segmentacion()
px = float(d["pixel_mm_original"])

ETAPAS = ["1 · Imagen", "2 · Reconstrucción", "3 · Filtros", "4 · Segmentación",
          "5 · Características", "6 · Clasificación", "7 · Baja dosis y compresión", "8 · Resultados"]
# Solo se calcula la etapa elegida (con pestañas, Streamlit calcularía todas en cada cambio)
etapa = st.radio("Etapa del trabajo", ETAPAS, horizontal=True, label_visibility="collapsed")
st.divider()

# ===========================================================================
# 1. Imagen
# ===========================================================================
if etapa == ETAPAS[0]:
    st.subheader("El corte de trabajo")
    st.write("El corte de LIDC con mayor área del nódulo, remuestreado a 0,7 mm y recortado a "
             f"{C.ROI_MM:g} mm alrededor del nódulo. Los contornos son de cada radiólogo (colores) y el "
             "consenso (≥ 50 % de los radiólogos, blanco).")
    fig, ax = figura(3, 3.8)
    ax[0].imshow(d["corte_completo"], **VENTANA)
    sc = d["semilla_completo"] * C.PIXEL_MM / px
    lado = C.ROI_MM / px
    ax[0].add_patch(plt.Rectangle((sc[1] - lado / 2, sc[0] - lado / 2), lado, lado, fill=False, ec="y", lw=1.2))
    ax[0].set_title(f"Corte completo ({d['corte_completo'].shape[0]} px de {px:.2f} mm)", fontsize=9)
    ax[1].imshow(d["img"], **VENTANA)
    for k, m in enumerate(d["radiologos"]):
        contorno(ax[1], m, plt.cm.tab10(k), 1.0)
    contorno(ax[1], d["consenso"], "w", 1.6)
    ax[1].set_title("Recorte con los radiólogos", fontsize=9)
    z = zoom_alrededor(d["consenso"])
    ax[2].imshow(d["img"][z], **VENTANA)
    contorno(ax[2], d["consenso"][z], "w", 1.6)
    ax[2].set_title("Zoom al nódulo", fontsize=9)
    sin_ejes(ax)
    mostrar(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Histograma de HU** (análisis antes de elegir métodos)")
        fig, ax = figura(1, 3.0, 5)
        ax[0].hist(d["img"].ravel(), bins=150, range=(-1100, 500), color="0.4")
        if d["consenso"].any():
            ax2 = ax[0].twinx()
            ax2.hist(d["img"][d["consenso"]], bins=60, range=(-1100, 500), color="tab:red", alpha=0.6)
            ax2.set_yticks([])
        for x, t in ((-1000, "aire"), (-850, "pulmón"), (0, "agua"), (C.HU_CALCIO, "calcio")):
            ax[0].axvline(x, color="k", lw=0.6, ls=":")
            ax[0].text(x, ax[0].get_ylim()[1] * 0.9, t, fontsize=7, rotation=90, ha="right")
        ax[0].set_xlabel("HU")
        ax[0].set_ylabel("píxeles del recorte")
        ax[0].set_title("gris: todo el recorte · rojo: el nódulo", fontsize=8)
        mostrar(fig)
    with c2:
        st.markdown("**Ruido de la imagen**")
        tejido = B.mascara_tejido_blando(d["corte_completo_rem"], C.PIXEL_MM)
        s0 = B.ruido_global(d["corte_completo_rem"], C.PIXEL_MM, tejido)
        st.metric("Ruido global (moda del desvío local en tejido blando)", f"{s0:.1f} HU")
        st.write("Se mide como Christianson et al. (2015): el desvío en ventanas de ~6 mm sobre tejido blando "
                 "homogéneo, y la moda de esos desvíos. En el parénquima no sirve, porque la textura del "
                 "pulmón se confunde con ruido. Este valor ancla la simulación de baja dosis (pestaña 7).")
        fig, ax = figura(1, 2.6, 5)
        ax[0].imshow(d["corte_completo_rem"], **VENTANA)
        ax[0].imshow(np.ma.masked_where(~tejido, tejido), cmap="autumn", alpha=0.5)
        ax[0].set_title("Tejido blando donde se mide (naranja)", fontsize=8)
        sin_ejes(ax)
        mostrar(fig)

# ===========================================================================
# 2. Reconstrucción
# ===========================================================================
if etapa == ETAPAS[1]:
    st.subheader("Reconstrucción tomográfica")
    st.write("Se simula la adquisición del corte: HU → μ, transformada de Radon (sinograma, Beer-Lambert) "
             "y se reconstruye con cada método de la materia. Sirve para comparar MÉTODOS de reconstrucción; "
             "la simulación de dosis del trabajo está en la pestaña 7.")
    c1, c2, c3, c4 = st.columns(4)
    metodo = c1.selectbox("Método", ["FBP (rampa)", "FBP (Hann)", "SBP", "SART"])
    n_ang = c2.select_slider("Proyecciones (en 180°)", [30, 60, 90, 180, 360], value=180)
    it = c3.slider("Iteraciones de SART", 1, 5, 1, disabled=metodo != "SART")
    ruido_rec = c4.select_slider("Fotones por rayo (I0)", ["sin ruido", "1e6", "1e5", "2.5e4", "1e4"],
                                 value="sin ruido")
    if st.button("Reconstruir", type="primary"):
        with st.spinner("Reconstruyendo (SART con muchas proyecciones tarda unos segundos)..."):
            met, filtro = {"FBP (rampa)": ("fbp", "ramp"), "FBP (Hann)": ("fbp", "hann"),
                           "SBP": ("sbp", "ramp"), "SART": ("sart", "ramp")}[metodo]
            I0 = None if ruido_rec == "sin ruido" else float(ruido_rec)
            # recorte cuadrado del corte remuestreado alrededor del nódulo (más rápido que 512 x 512)
            lado = 256
            ref = _recortar(d["corte_completo_rem"], tuple(d["semilla_completo"]), lado, -1000.0)
            rec, p = R.simular(ref, C.PIXEL_MM, n_angulos=n_ang, I0=I0, metodo=met, iteraciones=it,
                               filtro=filtro, rng=np.random.default_rng(C.SEMILLA))
            fov = R.mascara_fov(lado)
            st.session_state["rec"] = (ref, rec, p, fov, metodo, n_ang, ruido_rec)
    if "rec" in st.session_state:
        ref, rec, p, fov, met_txt, n_txt, r_txt = st.session_state["rec"]
        fig, ax = figura(4, 3.4)
        ax[0].imshow(np.where(fov, ref, -1000), **VENTANA)
        ax[0].set_title("Original", fontsize=9)
        ax[1].imshow(p, cmap="gray", aspect="auto")
        ax[1].set_title(f"Sinograma ({n_txt} proyecciones)", fontsize=9)
        ax[1].set_xlabel("ángulo")
        ax[1].set_ylabel("posición del detector")
        if met_txt == "SBP":
            ax[2].imshow(rec, cmap="gray")
        else:
            ax[2].imshow(rec, **VENTANA)
        ax[2].set_title(f"{met_txt} · {r_txt}", fontsize=9)
        err = np.where(fov, rec - ref, 0)
        lim = np.percentile(np.abs(err[fov]), 99) or 1
        ax[3].imshow(err, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax[3].set_title("Error (reconstruida − original)", fontsize=9)
        for a in (ax[0], ax[2], ax[3]):
            a.axis("off")
        mostrar(fig)
        if met_txt == "SBP":
            st.info("La SBP no está escalada (es el adjunto de Radon, no su inversa): sirve para ver por qué "
                    "hace falta el filtro, no para medir HU.")
        else:
            rmse = float(np.sqrt(np.mean((rec - ref)[fov] ** 2)))
            st.metric("Error RMS dentro del campo de visión", f"{rmse:.1f} HU")
        st.caption("En el trabajo, con 60 proyecciones SART da ~74 HU de error aun sin ruido, contra ~25 HU de "
                   "FBP con 180 (Tabla A3): por eso la baja dosis se simula con menos fotones y no con menos "
                   "proyecciones (Byl et al. 2023). ART no se implementa: requiere armar la matriz del sistema.")

# ===========================================================================
# 3. Filtros
# ===========================================================================
if etapa == ETAPAS[2]:
    st.subheader("Mejoramiento: filtros convencionales y óptimos")
    st.write("Los filtros se prueban sobre una versión ruidosa del corte (baja dosis simulada), porque la "
             "imagen original ya tiene poco ruido. En el pipeline NO se filtra antes de medir.")
    c1, c2, c3 = st.columns(3)
    base = c1.selectbox("Imagen", ["original", "×1/4 de la dosis", "×1/10 de la dosis"], index=2)
    filtro = c2.selectbox("Filtro", ["mediana", "gaussiano", "Wiener adaptativo"])
    if filtro == "mediana":
        param = c3.select_slider("Tamaño (px)", [3, 5, 7], value=3)
    elif filtro == "gaussiano":
        param = c3.select_slider("σ (px)", [0.5, 1.0, 1.5, 2.0], value=1.0)
    else:
        param = c3.select_slider("Ventana (px)", [3, 5, 7], value=5)

    @st.cache_data
    def imagen_ruidosa(archivo, f):
        d_ = cargar_corte(C.DIR_CORTES / f"{archivo}.npz")
        if f >= 1:
            return d_["img"]
        E7 = modulo_07()
        C.SIGMA_E2_LECTURA = sigma_e2_lidc()[0] or 30.0   # el resultado principal
        adq = E7.adquisicion(d_, encabezado(fila), {}, electronico=True, anclaje="hibrido")
        img, _ = adq.a_fraccion(f, np.random.default_rng(C.SEMILLA))
        return E7.como_si_fuera_original(d_, img)["img"]

    f = {"original": 1.0, "×1/4 de la dosis": 0.25, "×1/10 de la dosis": 0.1}[base]
    with st.spinner("Preparando la imagen..."):
        img_r = imagen_ruidosa(fila.archivo, f)
    salida = {"mediana": lambda x: M.mediana(x, int(param)), "gaussiano": lambda x: M.gaussiano(x, param),
              "Wiener adaptativo": lambda x: M.filtro_wiener(x, int(param))}[filtro](img_r.astype(np.float32))
    seg0, _ = segmentar(d, T0, R0)
    pul = pulmon_roi(d) & ~ndi.binary_dilation(seg0, morphology.disk(6))
    pul = pul & (d["img"] < -600)  # parénquima, sin vasos grandes

    def ruido_parenquima(x):
        return float(np.std((x - ndi.median_filter(x, 5))[pul])) if pul.any() else np.nan

    fig, ax = figura(3, 3.6)
    z = zoom_alrededor(seg0, 30)
    ax[0].imshow(img_r[z], **VENTANA)
    ax[0].set_title(f"Antes ({base})", fontsize=9)
    ax[1].imshow(salida[z], **VENTANA)
    ax[1].set_title(f"Después ({filtro})", fontsize=9)
    dif = (salida - img_r)[z]
    lim = np.percentile(np.abs(dif), 99) or 1
    ax[2].imshow(dif, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[2].set_title("Lo que sacó el filtro", fontsize=9)
    sin_ejes(ax)
    mostrar(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("Ruido en el parénquima", f"{ruido_parenquima(salida):.1f} HU",
              f"{ruido_parenquima(salida) - ruido_parenquima(img_r):+.1f} HU", delta_color="inverse")
    nb0, nb1 = nitidez_borde(img_r, seg0), nitidez_borde(salida, seg0)
    c2.metric("Nitidez del borde del nódulo", f"{nb1:.0f} HU/mm", f"{nb1 - nb0:+.0f} HU/mm")
    rmse = float(np.sqrt(np.mean((salida - d["img"]) ** 2)))
    c3.metric("RMSE contra la original", f"{rmse:.1f} HU")
    st.caption("Un buen filtro baja el ruido sin bajar la nitidez del borde. Una métrica global como el RMSE "
               "puede premiar un suavizado que destruye bordes: por eso se miran las tres.")

# ===========================================================================
# 4. Segmentación
# ===========================================================================
if etapa == ETAPAS[3]:
    st.subheader("Segmentación por crecimiento de regiones")
    c1, c2 = st.columns(2)
    T = c1.slider("Umbral T (HU): crece por los píxeles con HU > T", -700, -250, int(T0), 25)
    Rc = c2.select_slider("Radio del cierre del pulmón (mm)", [10.0, 15.0, 20.0, 30.0],
                          value=R0 if R0 in (10.0, 15.0, 20.0, 30.0) else 20.0)
    st.caption(f"Los valores elegidos en el ajuste con el conjunto de desarrollo son T = {T0} HU y cierre de "
               f"{R0:g} mm.")
    perm = region_permitida(d, Rc)
    binaria = d["img"] > T
    if perm is not None:
        binaria_p = binaria & perm
    else:
        binaria_p = binaria
    rr, cc = np.ogrid[:d["img"].shape[0], :d["img"].shape[1]]
    dentro = (rr - d["semilla"][0]) ** 2 + (cc - d["semilla"][1]) ** 2 <= (C.RADIO_MAX_MM / C.PIXEL_MM) ** 2
    region = _componente_con_semilla(binaria_p & dentro, tuple(d["semilla"]))
    seg, cruda = segmentar(d, T, Rc)
    otsu, t_otsu = segmentar_otsu(d["img"], tuple(d["semilla"]))

    fig, ax = figura(5, 3.2)
    ax[0].imshow(binaria, cmap="gray")
    ax[0].plot(d["semilla"][1], d["semilla"][0], "r+", ms=10)
    ax[0].set_title(f"1. Umbral HU > {T}", fontsize=9)
    ax[1].imshow(d["img"], **VENTANA)
    if perm is not None:
        contorno(ax[1], perm, "c", 1.2)
    ax[1].set_title("2. Pulmón cerrado (cian):\ndonde se permite crecer", fontsize=9)
    ax[2].imshow(region, cmap="gray")
    ax[2].set_title("3. Región conectada\na la semilla", fontsize=9)
    z = zoom_alrededor(seg | d["consenso"], 15)
    ax[3].imshow(d["img"][z], **VENTANA)
    contorno(ax[3], cruda[z], "y", 1.0, "--")
    contorno(ax[3], seg[z], "r", 1.6)
    ax[3].set_title("4. Apertura + relleno\n(amarillo: antes de la apertura)", fontsize=9)
    ax[4].imshow(d["img"][z], **VENTANA)
    for k, m in enumerate(d["radiologos"]):
        contorno(ax[4], m[z], plt.cm.tab10(k), 0.9)
    contorno(ax[4], d["consenso"][z], "w", 1.4)
    contorno(ax[4], seg[z], "r", 1.6)
    ax[4].set_title("Algoritmo (rojo) vs radiólogos", fontsize=9)
    sin_ejes(ax)
    mostrar(fig)

    ev = evaluar(seg, d["consenso"], d["radiologos"])
    rads = [m for m in d["radiologos"] if m.any()]
    dice_cada = float(np.mean([dice(seg, m) for m in rads])) if rads else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dice vs consenso", f"{ev['dice']:.3f}")
    c2.metric("Dice medio vs cada radiólogo", f"{dice_cada:.3f}")
    c3.metric("Concordancia entre radiólogos", f"{ev['concordancia_radiologos']:.3f}")
    c4.metric("Dice de Otsu vs consenso", f"{dice(otsu, d['consenso']):.3f}", f"umbral Otsu {t_otsu:.0f} HU",
              delta_color="off")
    ok_orig = ev["aceptable"]
    ok_justo = bool(np.isfinite(dice_cada) and dice_cada >= ev["concordancia_radiologos"])
    st.write(f"**Aceptable** con el criterio original (vs consenso): {'sí' if ok_orig else 'no'} · con el "
             f"criterio justo (vs cada radiólogo, script 12): {'sí' if ok_justo else 'no'}")
    st.caption("Una segmentación es aceptable si se parece a los radiólogos al menos tanto como ellos entre sí. "
               "Contra el consenso el algoritmo tiene ventaja, porque el consenso se arma con los mismos "
               "radiólogos; la comparación justa es contra cada uno.")

# ===========================================================================
# 5. Características
# ===========================================================================
if etapa == ETAPAS[4]:
    st.subheader("Características del nódulo")
    seg, cruda = segmentar(d, T0, R0)
    if seg.sum() < 5:
        st.warning("La segmentación quedó vacía: no se pueden medir características.")
    else:
        feats = medir(d, seg, cruda)
        car_all = catalogo_completo()
        dev = car_all[(car_all.conjunto == "entrenamiento") & car_all.aceptable]
        filas = []
        for k, nombre in NOMBRES.items():
            if k not in feats or k not in dev:
                continue
            v = feats[k]
            b = dev.loc[dev.grupo == "benigno", k].dropna()
            s = dev.loc[dev.grupo == "sospechoso", k].dropna()
            filas.append({"medida": nombre, "este nódulo": v,
                          "mediana benignos": b.median(), "mediana sospechosos": s.median(),
                          "se parece más a": ("benignos" if abs(v - b.median()) < abs(v - s.median())
                                              else "sospechosos")})
        st.dataframe(pd.DataFrame(filas).style.format(precision=3), hide_index=True, width="stretch")
        st.caption("Medianas de los 112 nódulos de desarrollo con segmentación aceptable. Que este nódulo se "
                   "parezca a un grupo en una medida no lo clasifica: eso lo hace el puntaje (pestaña 6).")

        st.markdown("**Cómo se mide cada grupo de características**")
        z = zoom_alrededor(seg, 26)
        pul = pulmon_roi(d)
        a_int, a_ext = anillos(seg, pul)
        borde = ndi.binary_dilation(seg & ~ndi.binary_erosion(seg))
        g = np.hypot(ndi.sobel(d["img"], 0), ndi.sobel(d["img"], 1)) / (8 * C.PIXEL_MM)
        calcio = (d["img"] > C.HU_CALCIO) & seg
        fig, ax = figura(4, 3.4)
        ax[0].imshow(d["img"][z], **VENTANA)
        contorno(ax[0], seg[z], "r", 1.4)
        contorno(ax[0], cruda[z], "y", 1.0, "--")
        ax[0].set_title(f"Forma: diámetro {feats['diametro_eq_mm']:.1f} mm\n"
                        f"circularidad {feats['circularidad']:.2f} · solidez {feats['solidez']:.2f}", fontsize=8)
        ax[1].imshow(g[z], cmap="viridis", vmin=0, vmax=np.percentile(g[z], 99))
        contorno(ax[1], borde[z], "w", 0.8)
        ax[1].set_title(f"Borde: gradiente de Sobel en una banda\nnitidez {feats['nitidez_borde_hu_mm']:.0f} HU/mm",
                        fontsize=8)
        ax[2].imshow(d["img"][z], **VENTANA)
        ax[2].imshow(np.ma.masked_where(~calcio[z], calcio[z]), cmap="autumn", alpha=0.9)
        contorno(ax[2], seg[z], "r", 1.0)
        ax[2].set_title(f"Calcio (> {C.HU_CALCIO} HU): {100 * feats['frac_calcio']:.1f} % del área\n"
                        f"robusto (mediana + manchas ≥ 3 px): {100 * feats['frac_calcio_robusto']:.1f} %", fontsize=8)
        ax[3].imshow(d["img"][z], **VENTANA)
        ax[3].imshow(np.ma.masked_where(~a_int[z], a_int[z]), cmap="cool", alpha=0.5)
        ax[3].imshow(np.ma.masked_where(~a_ext[z], a_ext[z]), cmap="spring", alpha=0.35)
        contorno(ax[3], seg[z], "r", 1.0)
        ax[3].set_title(f"Entorno: anillos de {C.ANILLO_INTERNO_MM:g} y {C.ANILLO_EXTERNO_MM:g} mm\n"
                        f"{feats.get('n_estructuras_convergentes', 0)} estructuras convergentes", fontsize=8)
        sin_ejes(ax)
        mostrar(fig)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = figura(1, 2.8, 5)
            ax[0].hist(d["img"][seg], bins=40, color="tab:red")
            ax[0].axvline(C.HU_CALCIO, color="k", ls=":", lw=0.8)
            ax[0].set_xlabel("HU dentro del nódulo")
            ax[0].set_title(f"Histograma · media {feats['hu_media']:.0f} · desvío {feats['hu_desvio']:.0f} · "
                            f"asimetría {feats['hu_asimetria']:.2f}", fontsize=8)
            mostrar(fig)
        with c2:
            from tpi.caracteristicas import glcm as _glcm
            P = _glcm(d["img"], seg)
            fig, ax = figura(1, 2.8, 3.4)
            ax[0].imshow(P, cmap="magma")
            ax[0].set_title(f"GLCM (16 niveles, {C.VENTANA_TEXTURA_NODULO[0]} a {C.VENTANA_TEXTURA_NODULO[1]} HU)\n"
                            f"contraste {feats['glcm_contraste']:.1f} · entropía {feats['glcm_entropia']:.2f}",
                            fontsize=8)
            ax[0].set_xlabel("nivel del vecino")
            ax[0].set_ylabel("nivel del píxel")
            mostrar(fig)

# ===========================================================================
# 6. Clasificación
# ===========================================================================
if etapa == ETAPAS[5]:
    st.subheader("Clasificación: el puntaje combinado")
    st.write("El puntaje del script 06 elige las k medidas que mejor separan (sin repetir medidas muy "
             "correlacionadas), las estandariza y las promedia con su signo. Acá se arma con los nódulos de "
             "desarrollo **sin el paciente de este nódulo**, para puntuarlo como a un caso nuevo.")
    nombre_conj = st.radio("Medidas disponibles", ["diámetro solo", "interior", "interior sin tamaño",
                                                   "interior + entorno"], index=1, horizontal=True)
    modelo, cols, tr = puntaje_sin_paciente(fila.paciente, nombre_conj)
    car_all = catalogo_completo()
    yo = car_all[(car_all.paciente == fila.paciente) & (car_all.scan_id == fila.scan_id)
                 & (car_all.nodulo == fila.nodulo)]
    s_yo = float(modelo.puntuar(yo[cols])[0])
    s_tr = modelo.puntuar(tr[cols])
    y_tr = (tr.grupo == "sospechoso").to_numpy(int)
    # corte: el que maximiza sensibilidad + especificidad en el armado
    cortes = np.unique(s_tr)
    j = [((s_tr >= c) & (y_tr == 1)).sum() / max(y_tr.sum(), 1) + ((s_tr < c) & (y_tr == 0)).sum()
         / max((1 - y_tr).sum(), 1) for c in cortes]
    corte = float(cortes[int(np.argmax(j))])
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("**Medidas elegidas** (signo +1: mayor en sospechosos)")
        med = modelo.medidas.assign(medida=lambda x: x.medida.map(lambda m: NOMBRES.get(m, m)))
        st.dataframe(med.rename(columns={"auc_entrenamiento": "AUC al armar"}).style.format(precision=3),
                     hide_index=True, width="stretch")
        st.metric("Puntaje de este nódulo", f"{s_yo:+.2f}",
                  "lado sospechoso" if s_yo >= corte else "lado benigno", delta_color="off")
        st.write(f"Grupo según los radiólogos: **{fila.grupo}**.")
    with c2:
        fig, ax = figura(1, 3.2, 5.5)
        bins = np.linspace(min(s_tr.min(), s_yo) - 0.1, max(s_tr.max(), s_yo) + 0.1, 30)
        ax[0].hist(s_tr[y_tr == 0], bins=bins, alpha=0.6, label="benignos", color="tab:blue")
        ax[0].hist(s_tr[y_tr == 1], bins=bins, alpha=0.6, label="sospechosos", color="tab:red")
        ax[0].axvline(corte, color="k", ls=":", lw=1, label="corte")
        ax[0].axvline(s_yo, color="k", lw=2.2, label="este nódulo")
        ax[0].set_xlabel("puntaje")
        ax[0].legend(fontsize=7)
        ax[0].set_title(f"Puntajes de los nódulos con los que se armó (AUC al armar "
                        f"{auc(y_tr, s_tr):.2f})", fontsize=8)
        mostrar(fig)
    p06 = leer_csv("06_puntaje.csv")
    if p06 is not None:
        st.markdown("**Resultado del trabajo (validación cruzada por paciente, script 06)**")
        cols06 = [c for c in ("puntaje", "auc_validacion_cruzada", "ic95_inf", "ic95_sup") if c in p06]
        st.dataframe(p06[cols06].style.format(precision=3), hide_index=True, width="stretch")
    st.caption("El AUC es la probabilidad de que un sospechoso tomado al azar tenga un puntaje mayor que un "
               "benigno tomado al azar (0,5 = azar, 1 = perfecto). El AUC del armado de arriba es optimista; "
               "el que vale es el de validación cruzada.")

# ===========================================================================
# 7. Baja dosis y compresión
# ===========================================================================
if etapa == ETAPAS[6]:
    st.subheader("Robustez: baja dosis (Naziroglu et al. 2017) y compresión")
    se_lidc, se_sup = sigma_e2_lidc()
    c1, c2 = st.columns([1.2, 1])
    cond = c1.radio("Condición", ["original"] + list(FRACCIONES) + ["JPEG calidad 50", "JPEG calidad 25"],
                    horizontal=True, index=3)
    escenarios = {"sin ruido electrónico (σe² = 0)": 0.0}
    if se_lidc is not None:
        escenarios[f"el más probable en LIDC (σe² = {se_lidc:.0f})"] = se_lidc
        escenarios[f"el máximo compatible (σe² = {se_sup:.0f})"] = se_sup
    else:
        escenarios["valor de literatura (σe² = 30)"] = 30.0
    esc = c2.selectbox("Ruido electrónico", list(escenarios), index=min(1, len(escenarios) - 1),
                       disabled=cond not in FRACCIONES)

    @st.cache_data
    def condicion(archivo, cond, sigma_e2):
        d_ = cargar_corte(C.DIR_CORTES / f"{archivo}.npz")
        E7 = modulo_07()
        if cond == "original":
            return np.maximum(d_["corte_completo"], -1000).astype(np.float32), None, None
        if cond.startswith("JPEG"):
            img, _, _ = E7.degradar(d_, None, "jpeg", int(cond.split()[-1]), 0)
            return img, None, None
        C.SIGMA_E2_LECTURA = sigma_e2
        adq = E7.adquisicion(d_, encabezado(fila), {}, electronico=sigma_e2 > 0, anclaje="hibrido")
        img, ruido = adq.a_fraccion(FRACCIONES[cond], np.random.default_rng(C.SEMILLA))
        return img, ruido, adq.N_alta

    with st.spinner("Simulando (sinograma virtual, ruido de Poisson + electrónico, FBP del ruido)..."):
        img_c, ruido_c, n_alta = condicion(fila.archivo, cond, escenarios[esc])
    E7 = modulo_07()
    d2 = E7.como_si_fuera_original(d, img_c)
    seg_o, cruda_o = segmentar(d, T0, R0)
    seg_c, cruda_c = segmentar(d2, T0, R0)
    f_o, f_c = medir(d, seg_o, cruda_o), medir(d2, seg_c, cruda_c)

    z = zoom_alrededor(seg_o | seg_c, 26)
    fig, ax = figura(3, 3.6)
    ax[0].imshow(d["img"][z], **VENTANA)
    contorno(ax[0], seg_o[z], "r", 1.4)
    ax[0].set_title("Original", fontsize=9)
    ax[1].imshow(d2["img"][z], **VENTANA)
    contorno(ax[1], seg_c[z], "r", 1.4)
    contorno(ax[1], (d2["img"] > C.HU_CALCIO)[z] & seg_c[z], "y", 0.8)
    ax[1].set_title(f"{cond}" + (f" · {esc.split('(')[1][:-1]}" if cond in FRACCIONES else ""), fontsize=9)
    if ruido_c is not None:
        rr_ = _remuestrear(ruido_c, px, 1)
        rr_ = _recortar(rr_, tuple(d["semilla_completo"]), d["img"].shape[0], 0.0)
        lim = np.percentile(np.abs(rr_), 99) or 1
        ax[2].imshow(rr_[z], cmap="coolwarm", vmin=-lim, vmax=lim)
        ax[2].set_title("Ruido agregado (solo el ruido se\nreconstruye y se suma a la original)", fontsize=9)
        agregado = float(np.std(rr_))
    else:
        dif = (d2["img"] - d["img"])[z]
        lim = np.percentile(np.abs(dif), 99) or 1
        ax[2].imshow(dif, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax[2].set_title("Diferencia con la original", fontsize=9)
        agregado = float(np.std(d2["img"] - d["img"]))
    sin_ejes(ax)
    mostrar(fig)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ruido agregado", f"{agregado:.0f} HU")
    c2.metric("Dice vs consenso", f"{dice(seg_c, d['consenso']):.3f}",
              f"{dice(seg_c, d['consenso']) - dice(seg_o, d['consenso']):+.3f}")
    if f_o and f_c:
        c3.metric("Calcio, umbral simple", f"{100 * f_c['frac_calcio']:.1f} %",
                  f"{100 * (f_c['frac_calcio'] - f_o['frac_calcio']):+.1f} pp", delta_color="inverse")
        c4.metric("Calcio, medida robusta", f"{100 * f_c['frac_calcio_robusto']:.1f} %",
                  f"{100 * (f_c['frac_calcio_robusto'] - f_o['frac_calcio_robusto']):+.1f} pp",
                  delta_color="inverse")
        tabla = pd.DataFrame([{"medida": NOMBRES.get(k, k), "original": f_o[k], cond: f_c.get(k, np.nan),
                               "cambio %": 100 * (f_c.get(k, np.nan) - f_o[k]) / abs(f_o[k]) if f_o[k] else np.nan}
                              for k in NOMBRES if k in f_o])
        st.dataframe(tabla.style.format(precision=3), hide_index=True, width="stretch")
        # puntaje transferido: armado con las imágenes originales, aplicado a esta condición
        modelo, cols, _ = puntaje_sin_paciente(fila.paciente, "interior")
        s_o = float(modelo.puntuar(pd.DataFrame([f_o]).reindex(columns=cols))[0])
        s_c = float(modelo.puntuar(pd.DataFrame([f_c]).reindex(columns=cols))[0])
        st.metric("Puntaje 'interior' (armado con imágenes originales)", f"{s_c:+.2f}", f"{s_c - s_o:+.2f}",
                  delta_color="off")
    st.caption("La simulación sigue a Naziroglu et al. (2017): sinograma virtual de la imagen, ruido que falta "
               "para llegar a esa fracción de la dosis (compensando el que la imagen ya trae: N·f/(1−f)), ruido "
               "electrónico y reconstrucción SOLO del ruido con la ventana del kernel del equipo. Los fotones se "
               "anclan con el ruido medido en la imagen (anclaje híbrido, el resultado principal).")

    res = leer_csv("07_clasificacion_anclaje_hibrido_se6.csv")
    seg07 = leer_csv("07_segmentacion_anclaje_hibrido_se6.csv")
    if seg07 is not None:
        with st.expander("Resultados del trabajo con todos los nódulos (script 07, resultado principal)"):
            st.dataframe(seg07, hide_index=True, width="stretch")
            if res is not None:
                st.dataframe(res, hide_index=True, width="stretch")

# ===========================================================================
# 8. Resultados
# ===========================================================================
if etapa == ETAPAS[7]:
    st.subheader("Resultados globales del trabajo")
    figuras = [
        ("05_boxplots.png", "Medidas que separan benignos de sospechosos (Mann-Whitney con Holm)"),
        ("06_roc.png", "Curvas ROC de los puntajes (validación cruzada por paciente)"),
        ("06_roc_sincalcio.png", "ROC sin los nódulos calcificados"),
        ("07_baja_dosis_anclaje_hibrido_se6.png", "Baja dosis, resultado principal (híbrido, σe² estimado en LIDC)"),
        ("08_propuesta_anclaje_hibrido_se6.png", "Propuesta: puntaje robusto y calcio robusto"),
        ("07a_calibracion.png", "Calibración de la simulación con LIDC (07a)"),
        ("comparaciones/incertidumbre_anclajes.png", "Incertidumbre de la dosis real con cada anclaje"),
        ("14_nitidez_borde.png", "Cómo se mide la nitidez del borde"),
    ]
    hay = [(f, t) for f, t in figuras if (C.DIR_FIGURAS / f).exists()]
    if not hay:
        st.info("Todavía no hay figuras en figuras/ (se generan con los scripts 05 a 08).")
    for f, t in hay:
        st.markdown(f"**{t}**")
        st.image(str(C.DIR_FIGURAS / f), width="stretch")
