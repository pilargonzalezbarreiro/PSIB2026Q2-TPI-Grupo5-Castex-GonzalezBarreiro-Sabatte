"""Simulación de baja dosis a partir de la imagen ya reconstruida.

Método: Naziroglu et al. (2017), Physica Medica 36:12-23. Es el que aplica a
LIDC porque no necesita los datos crudos del tomógrafo (Zeng et al. 2015 y
Yu et al. 2012 trabajan sobre las proyecciones reales, que LIDC no trae).

Por qué no alcanza con "agregar ruido de Poisson con un I0":
  La imagen original YA tiene el ruido de su propia dosis. Si bajamos a la
  fracción f, la varianza del ruido cuántico pasa de σ0² a σ0²/f. Hay que
  AGREGAR solo la diferencia, σ0²(1/f - 1). Como la varianza ∝ 1/fotones,
  sumar varianzas es sumar inversas de fotones:

      1/(f·N) = 1/N + 1/N_eq   =>   N_eq = N · f / (1 - f)

  (con corrientes: I_eq = I_alta·I_baja / (I_alta - I_baja); 200 -> 50 mAs
  se logra agregando el ruido de 66,7 mAs, no el de 50: si se agregara el de
  50 quedaría 1/(1/200 + 1/50) = 40 mAs). Con f = 1, N_eq = ∞: no se agrega
  nada, que es lo que tiene que pasar.

Pasos (los mismos del paper):
  1. sinograma virtual: Radon de la imagen en μ (problema directo);
  2. fotones de la adquisición ORIGINAL por rayo, anclados al DICOM:
     N(s) = Φ · mAs · (kVp/120)^a · Δs · espesor / n_vistas · bowtie(s);
  3. sinograma de baja dosis: Poisson con N_eq·e^{-p} + ruido electrónico
     (compensado, ver abajo), logaritmo;
  4. se resta el sinograma virtual sin ruido -> SOLO el ruido;
  5. FBP del ruido con la ventana del kernel del equipo;
  6. se suma esa imagen de ruido a la original.
  Como la original no se re-reconstruye, no pasa dos veces por la
  interpolación de Radon y del FBP: con f = 1 sale idéntica.

Ruido electrónico compensado: una lectura con N fotones medios tiene
varianza N + σe² (Poisson + gaussiano). En el logaritmo, Var ≈ (N + σe²)/N².
Para pasar de N a fN hay que agregar
  ΔVar = (1/f - 1)/N + σe² (1/f² - 1)/N².
Con Poisson de media N_eq = N f/(1-f) más un gaussiano de varianza σa² la
varianza agregada es 1/N_eq + σa²/N_eq²; igualando, σa² = σe² (1+f)/(1-f)
(no depende del rayo).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import iradon, radon

from . import config as C
from .encabezados import familia_kernel, mas_efectivo, sff_bowtie


def hu_a_mu(hu):
    return np.maximum(C.MU_AGUA * (1 + np.asarray(hu, float) / 1000.0), 0.0)


def angulos(n):
    return np.linspace(0.0, 180.0, n, endpoint=False)


# ---------------------------------------------------------------------------
# 1. Sinograma virtual
# ---------------------------------------------------------------------------
def sinograma_virtual(corte_hu, pixel_mm, n_vistas=C.N_VISTAS_DOSIS):
    """Integrales de línea p = ∫μ dl (adimensional). circle=False: el cuerpo
    del paciente y la camilla llegan a las esquinas del corte, y con
    circle=True Radon los trataría como aire (camino más corto = menos ruido)."""
    mu = hu_a_mu(np.maximum(corte_hu, -1000))
    return radon(mu, theta=angulos(n_vistas), circle=False) * pixel_mm


# ---------------------------------------------------------------------------
# 2. Fotones de la adquisición original
# ---------------------------------------------------------------------------
def perfil_bowtie(n_bins, pixel_mm, modelo=None, **par):
    """Transmisión del bowtie relativa al rayo central, por bin del sinograma.
    modelo "cilindro": exp(-α μ_agua (D - camino(s))), camino(s) = 2√(R² - s²),
    con piso b_min; α = 0 es sin bowtie y α = 1 compensa todo el agua que le
    falta a un rayo lateral de un cilindro de diámetro D.
    modelo "gauss": b_min + (1 - b_min) exp(-(s/ancho)²).
    modelo "plano": sin bowtie.
    modelo "kramer": Kramer et al. (2017), ver bowtie_kramer."""
    modelo = modelo or C.BOWTIE_MODELO
    s = (np.arange(n_bins) - (n_bins - 1) / 2) * pixel_mm
    b_min = par.get("b_min", C.BOWTIE_MIN)
    if modelo == "kramer":
        return bowtie_kramer(s, par.get("sff", C.BOWTIE_SFF), par.get("dsi_mm", C.BOWTIE_DSI_MM),
                             par.get("mu_pmma", C.MU_PMMA_MM))
    if modelo == "plano":
        return np.ones(n_bins)
    if modelo == "gauss":
        ancho = par.get("ancho_mm", C.BOWTIE_ANCHO_MM)
        return b_min + (1 - b_min) * np.exp(-(s / ancho) ** 2)
    if modelo == "cilindro":
        D = par.get("diametro_mm", C.BOWTIE_DIAMETRO_MM)
        a = par.get("compensacion", C.BOWTIE_COMPENSACION)
        R = D / 2
        camino = 2 * np.sqrt(np.maximum(R ** 2 - s ** 2, 0))
        return np.maximum(np.exp(-a * C.MU_AGUA * (D - camino)), b_min)
    raise ValueError(modelo)


# Coeficientes del filtro de cuerpo "básico" de Kramer et al. (2017), Tabla 2:
# espesor de PMMA t(θ) = (a - b) / (1 + exp((AC·θ - c)/d)) + b, θ en radianes
# (ángulo del abanico), t en cm. Sigue el filtro ideal para un fantoma de 32 cm
# hasta ~12° y después se aplana en ~8 cm.
_KRAMER_CUERPO = dict(a=8.3015550, b=-0.1228647, c=0.1992510, d=-0.0472935)


def bowtie_kramer(s_mm, sff=C.BOWTIE_SFF, dsi_mm=C.BOWTIE_DSI_MM, mu_pmma=C.MU_PMMA_MM):
    """Transmisión relativa al centro de un rayo que pasa a s del isocentro.
    En un haz en abanico, ese rayo sale con ángulo θ = arcsen(s/DSI). AC corrige
    la geometría para fuentes a otra distancia que los 54 cm del modelo; SFF
    escala el espesor (W = exp(-μ t SFF), como en el paper). Se usa un μ
    efectivo (no se modela el endurecimiento del haz: entra como incertidumbre
    de μ_PMMA)."""
    k = _KRAMER_CUERPO
    th = np.arcsin(np.clip(np.abs(s_mm) / dsi_mm, 0, 1))
    ac = np.arctan(160 / 540) / np.arctan(160 / dsi_mm)
    t_cm = (k["a"] - k["b"]) / (1 + np.exp((ac * th - k["c"]) / k["d"])) + k["b"]
    t0 = (k["a"] - k["b"]) / (1 + np.exp(-k["c"] / k["d"])) + k["b"]
    return np.exp(-mu_pmma * 10 * (np.maximum(t_cm, 0) - max(t0, 0)) * sff)


def lecturas_por_rayo(pixel_mm, espesor_mm, n_vistas=C.N_VISTAS_DOSIS):
    """Cuántas lecturas reales del detector junta un rayo de la simulación
    (más vistas por rotación, canales más angostos, filas más finas que el
    corte). Las cuentas se suman, la varianza electrónica también."""
    return ((C.VISTAS_POR_ROTACION / n_vistas) * (pixel_mm / C.ANCHO_DETECTOR_ISO_MM)
            * (espesor_mm / C.ANCHO_FILA_MM))


def fotones_por_rayo(mas, kvp, pixel_mm, espesor_mm, n_vistas=C.N_VISTAS_DOSIS, phi=None):
    """Fotones incidentes (centro del bowtie) en un rayo de la simulación:
    Φ · mAs · (kVp/120)^a · ancho del rayo · espesor / vistas."""
    phi = C.FOTONES_POR_MAS_MM2 if phi is None else phi
    kvp = 120.0 if not np.isfinite(kvp) else kvp
    return phi * mas * (kvp / 120.0) ** C.EXPONENTE_KVP * pixel_mm * espesor_mm / n_vistas


# ---------------------------------------------------------------------------
# 3-4. Sinograma de ruido
# ---------------------------------------------------------------------------
def ruido_de_adquisicion(p, N, sigma_e2, rng):
    """Ruido (en p) de una adquisición con N fotones incidentes por rayo
    (escalar o por bin): -ln(cuentas/N) - p."""
    N = np.broadcast_to(np.asarray(N, float).reshape(-1, 1) if np.ndim(N) == 1 else N, p.shape)
    cuentas = rng.poisson(N * np.exp(-p)).astype(float)
    if sigma_e2 > 0:
        cuentas += rng.normal(0.0, np.sqrt(sigma_e2), p.shape)
    return -np.log(np.maximum(cuentas, C.CUENTAS_MIN) / N) - p


def ruido_a_agregar(p, N_alta, f, sigma_e2, rng):
    """Ruido que falta para pasar de la dosis original (N_alta fotones) a la
    fracción f, con la compensación del ruido cuántico y electrónico que la
    imagen ya trae."""
    if f >= 1:
        return np.zeros_like(p)
    N_eq = np.asarray(N_alta, float) * f / (1 - f)
    return ruido_de_adquisicion(p, N_eq, sigma_e2 * (1 + f) / (1 - f), rng)


# ---------------------------------------------------------------------------
# 5. FBP del ruido
# ---------------------------------------------------------------------------
def reconstruir_ruido(ruido_p, pixel_mm, n, ventana):
    """Sinograma de ruido -> imagen de ruido en HU (ΔHU = 1000 Δμ / μ_agua)."""
    th = angulos(ruido_p.shape[1])
    dmu = iradon(ruido_p / pixel_mm, theta=th, filter_name=ventana, circle=False, output_size=n)
    return (1000.0 * dmu / C.MU_AGUA).astype(np.float32)


# ---------------------------------------------------------------------------
# Todo junto
# ---------------------------------------------------------------------------
class Adquisicion:
    """Lo que sabemos de cómo se adquirió UN corte: fotones por rayo de la
    dosis original, ruido electrónico (en unidades de la simulación) y la
    ventana del kernel. Se arma desde el DICOM (anclaje 'mAs') o, si falta
    el mAs, desde el ruido medido en la imagen (anclaje 'ruido')."""

    def __init__(self, corte_hu, pixel_mm, n_vistas=C.N_VISTAS_DOSIS, bowtie=None):
        self.corte = np.maximum(np.asarray(corte_hu, np.float32), -1000)
        self.px = float(pixel_mm)
        self.n_vistas = n_vistas
        self.p = sinograma_virtual(self.corte, self.px, n_vistas)
        # bowtie: None (el de config) o dict(modelo=..., parámetros)
        self.bowtie = perfil_bowtie(self.p.shape[0], self.px, **(bowtie or {}))
        self.N_alta = None      # fotones por rayo en el centro del bowtie
        self.sigma_e2 = 0.0
        self.ventana = "shepp-logan"
        self.anclaje = None

    # -- anclajes -------------------------------------------------------
    def usar_bowtie(self, **par):
        self.bowtie = perfil_bowtie(self.p.shape[0], self.px, **par)
        return self

    def desde_dicom(self, enc, phi=None, electronico=True):
        sff, _ = sff_bowtie(enc.get("fabricante"), enc.get("modelo"))
        self.usar_bowtie(modelo="kramer", sff=sff)
        mas = mas_efectivo(enc)
        if not np.isfinite(mas):
            raise ValueError("el encabezado no trae mAs")
        esp = enc.get("espesor_mm", np.nan)
        esp = float(esp) if isinstance(esp, (int, float)) and np.isfinite(esp) else C.MAX_ESPESOR_MM
        self.N_alta = fotones_por_rayo(mas, float(enc.get("kvp", np.nan)), self.px, esp, self.n_vistas, phi)
        self.sigma_e2 = C.SIGMA_E2_LECTURA * lecturas_por_rayo(self.px, esp, self.n_vistas) if electronico else 0.0
        self.ventana = familia_kernel(enc.get("kernel"))[1]
        self.anclaje = "mAs"
        return self

    def desde_ruido(self, sigma0_hu, ventana=None, rng=None, n_ref=1e5):
        """Sin mAs: busca los fotones que reproducen el ruido medido en la
        imagen (σ ∝ 1/√N). Supone que todo ese ruido es cuántico."""
        if not (np.isfinite(sigma0_hu) and sigma0_hu > 0):
            raise ValueError("ruido de la imagen no medible (¿hay tejido blando en el corte?)")
        if ventana:
            self.ventana = ventana
        s_ref = self.sigma_ruido(n_ref, rng=rng)
        self.N_alta = n_ref * (s_ref / sigma0_hu) ** 2
        self.sigma_e2 = 0.0
        self.anclaje = "ruido"
        return self

    def desde_ruido_hibrido(self, sigma0_hu, enc, rng=None, electronico=True):
        """Híbrido: fotones calibrados con el ruido medido en la imagen (no usa
        Φ) y ruido electrónico como en desde_dicom (su tamaño depende del valor
        por lectura y de la geometría del detector, no de Φ). Bowtie y ventana
        del modelo de equipo del DICOM."""
        if enc is not None:
            sff, _ = sff_bowtie(enc.get("fabricante"), enc.get("modelo"))
            self.usar_bowtie(modelo="kramer", sff=sff)
            ventana = enc.get("ventana_fbp") if isinstance(enc.get("ventana_fbp"), str) else None
        else:
            ventana = None
        self.desde_ruido(sigma0_hu, ventana=ventana, rng=rng)
        esp = enc.get("espesor_mm", np.nan) if enc is not None else np.nan
        esp = float(esp) if isinstance(esp, (int, float)) and np.isfinite(esp) else C.MAX_ESPESOR_MM
        self.sigma_e2 = C.SIGMA_E2_LECTURA * lecturas_por_rayo(self.px, esp, self.n_vistas) if electronico else 0.0
        self.anclaje = "híbrido"
        return self

    # -- simulación -----------------------------------------------------
    def ruido_original(self, rng=None):
        """Imagen del ruido cuántico que, según el modelo, ya trae la original
        (una realización). Sirve para calibrar y validar el anclaje."""
        rng = np.random.default_rng(C.SEMILLA) if rng is None else rng
        rp = ruido_de_adquisicion(self.p, self.N_alta * self.bowtie, self.sigma_e2, rng)
        return reconstruir_ruido(rp, self.px, self.corte.shape[0], self.ventana)

    def sigma_ruido(self, N_centro, mascara=None, rng=None):
        rng = np.random.default_rng(C.SEMILLA) if rng is None else rng
        rp = ruido_de_adquisicion(self.p, N_centro * self.bowtie, 0.0, rng)
        r = reconstruir_ruido(rp, self.px, self.corte.shape[0], self.ventana)
        m = mascara_tejido_blando(self.corte, self.px) if mascara is None else mascara
        return ruido_global(r, self.px, m)

    def a_fraccion(self, f, rng=None):
        """Imagen a la fracción f de la dosis original, y la imagen de ruido agregada."""
        if self.N_alta is None:
            raise RuntimeError("falta anclar la adquisición (desde_dicom o desde_ruido)")
        rng = np.random.default_rng(C.SEMILLA) if rng is None else rng
        rp = ruido_a_agregar(self.p, self.N_alta * self.bowtie, f, self.sigma_e2, rng)
        ruido = reconstruir_ruido(rp, self.px, self.corte.shape[0], self.ventana)
        return self.corte + ruido, ruido


# ---------------------------------------------------------------------------
# Ruido de una imagen real (para anclar y para validar)
# ---------------------------------------------------------------------------
def mascara_tejido_blando(img, pixel_mm):
    """Tejido blando homogéneo (músculo, mediastino) lejos de bordes. El umbral
    se aplica a la imagen SUAVIZADA (promedio local de ~6 mm): sobre la imagen
    con ruido, con σ ~ 25 HU casi ningún píxel tiene toda su ventana dentro de
    [0, 100] HU y la máscara quedaría vacía. Después se erosiona para alejarse
    de los bordes con grasa, pulmón o hueso."""
    k = _lado(pixel_mm)
    lo, hi = C.HU_TEJIDO_BLANDO
    suave = ndi.uniform_filter(np.asarray(img, np.float64), k)
    m = (suave >= lo) & (suave <= hi)
    return ndi.binary_erosion(m, np.ones((k, k), bool))


def ruido_global(img, pixel_mm, mascara):
    """Christianson et al. (2015): moda del desvío estándar local (ventana de
    ~6 mm) dentro de la máscara. La moda, y no la media, para que los pocos
    lugares con estructura (bordes, vasos) no inflen el valor. En el
    parénquima pulmonar esto falla: la textura del pulmón se confunde con
    ruido; por eso se mide en tejido blando."""
    if mascara is None or mascara.sum() < 200:
        return np.nan
    k = _lado(pixel_mm)
    x = np.asarray(img, np.float64)
    m1 = ndi.uniform_filter(x, k)
    m2 = ndi.uniform_filter(x * x, k)
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))[mascara]
    hi = np.percentile(sd, 99)
    if hi <= 0:
        return 0.0
    # Histograma fino (200 intervalos hasta el P99) suavizado, y el pico
    # ajustado con una parábola por los 3 intervalos de alrededor: así la moda
    # no queda redondeada al ancho del intervalo (con intervalos fijos de
    # 0,5 HU, diferencias de pocos % entre mediciones desaparecían).
    h, bordes = np.histogram(sd, bins=200, range=(0, hi))
    h = ndi.gaussian_filter1d(h.astype(float), 4)
    i = int(np.clip(np.argmax(h), 1, len(h) - 2))
    y0, y1, y2 = h[i - 1], h[i], h[i + 1]
    den = y0 - 2 * y1 + y2
    corr = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    ancho = bordes[1] - bordes[0]
    return float(bordes[0] + (i + 0.5 + np.clip(corr, -1, 1)) * ancho)


def error_medicion_ruido(img, pixel_mm, mascara):
    """Error de medir el ruido EN ESTA imagen, estimado con la propia imagen:
    se mide el ruido global por separado en la mitad izquierda y la derecha del
    tejido blando. Si la medición fuera perfecta darían igual; la diferencia
    estima el error. Con dos mitades de desvío s_m cada una, la diferencia
    tiene desvío √2·s_m y la medición completa ≈ s_m/√2, o sea |g1 - g2|/2.
    Con un solo par es una estimación ruidosa (en el fantoma dio 0,1 % con
    una variación real entre repeticiones de 0,3 %): por eso se usa la
    DISTRIBUCIÓN de estos errores sobre todos los estudios, no el de cada uno.
    Mide solo el error al azar; el sesgo sistemático (estructura del tejido
    que se cuenta como ruido) lo mide el 07a aparte (ruido no cuántico).
    Devuelve el error relativo (o nan si no hay tejido)."""
    cols = np.nonzero(mascara.any(axis=0))[0]
    if cols.size < 10:
        return np.nan
    medio = int(np.median(np.nonzero(mascara)[1]))
    izq, der = mascara.copy(), mascara.copy()
    izq[:, medio:] = False
    der[:, :medio] = False
    g1, g2 = ruido_global(img, pixel_mm, izq), ruido_global(img, pixel_mm, der)
    g = ruido_global(img, pixel_mm, mascara)
    if not (np.isfinite(g1) and np.isfinite(g2) and g > 0):
        return np.nan
    return float(abs(g1 - g2) / 2 / g)


def _lado(pixel_mm):
    k = max(3, int(round(C.LADO_RUIDO_LOCAL_MM / pixel_mm)))
    return k + (k % 2 == 0)
