"""Verifica la simulación de baja dosis (tpi/baja_dosis.py) en un fantoma
donde SÍ conocemos la verdad: una imagen sin ruido, una adquisición "de dosis
completa" simulada sobre ella, y la adquisición real a la fracción f.

Chequea lo que la devolución marcó:
  1. ×1 no agrega ruido (y la imagen no cambia: no se re-reconstruye).
  2. Con la compensación, original + ruido agregado tiene el MISMO ruido que
     adquirir directamente con f·N fotones. Sin compensación, no.
  3. El ruido total sigue la ley σ0/√f.
  4. Cuánto pesa el ruido electrónico en tórax.
  5. La ventana (kernel) cambia la textura del ruido, no solo su tamaño.

Uso:  python tests/test_baja_dosis.py      (~1-2 min)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tpi import baja_dosis as B  # noqa: E402
from tpi import reconstruccion as R  # noqa: E402

N, PX, ESP, MAS = 512, 0.7, 1.25, 100.0


def fantoma_limpio():
    rr, cc = np.mgrid[:N, :N]
    img = np.full((N, N), -1000.0)
    img[((rr - 256) / 230) ** 2 + ((cc - 256) / 250) ** 2 <= 1] = 40     # cuerpo (35 x 32 cm)
    for c0 in (165, 347):
        img[((rr - 250) / 150) ** 2 + ((cc - c0) / 75) ** 2 <= 1] = -850  # pulmones
    img[(rr - 250) ** 2 + (cc - 165) ** 2 <= 12 ** 2] = 30                # nódulo de ~17 mm
    img[(rr - 250) ** 2 + (cc - 165) ** 2 <= 3 ** 2] = 600                # calcio
    img[(rr - 420) ** 2 + (cc - 256) ** 2 <= 25 ** 2] = 700               # "vértebra"
    return img.astype(np.float32)


def sd_en(img, m):
    return float(np.std(img[m]))


def main():
    rng = np.random.default_rng(0)
    verdad = fantoma_limpio()
    tejido = B.mascara_tejido_blando(verdad, PX)
    rr, cc = np.mgrid[:N, :N]
    pulmon = ((rr - 250) / 140) ** 2 + ((cc - 347) / 65) ** 2 <= 1        # pulmón derecho, sin nódulo

    enc = dict(corriente_ma=MAS, kvp=120.0, espesor_mm=ESP, kernel="B45f")   # índice de dosis = corriente
    adq = B.Adquisicion(verdad, PX).desde_dicom(enc)
    print(f"Fotones por rayo (centro): {adq.N_alta:,.0f}   σe² por rayo: {adq.sigma_e2:.0f}   ventana {adq.ventana}")

    # "Tomografía original" = verdad + su propio ruido (una realización)
    ruido0 = adq.ruido_original(rng)
    original = verdad + ruido0
    s0_t, s0_p = sd_en(ruido0, tejido), sd_en(ruido0, pulmon)
    print(f"Ruido de la 'original': tejido {s0_t:.1f} HU, pulmón {s0_p:.1f} HU")
    # El ruido global (moda del desvío local) da algo menos que el desvío de
    # todos los píxeles juntos (el ruido no es igual en todo el cuerpo y la
    # ventana local no ve las frecuencias más bajas). No importa: para anclar
    # y validar se usa el MISMO estimador sobre la imagen real y sobre el
    # ruido simulado. Lo que tiene que dar es que medido sobre la imagen
    # (con anatomía) coincida con medido sobre el ruido solo.
    gni = B.ruido_global(original, PX, tejido)
    gni_ruido = B.ruido_global(ruido0, PX, tejido)
    print(f"  ruido global (Christianson): sobre la imagen {gni:.1f} HU, sobre el ruido solo {gni_ruido:.1f} HU")
    assert abs(gni - gni_ruido) / gni_ruido < 0.05, "la anatomía contamina el ruido global"

    # 1. ×1 no cambia nada
    sim = B.Adquisicion(original, PX).desde_dicom(enc)
    img1, r1 = sim.a_fraccion(1.0, rng)
    assert np.array_equal(img1, np.maximum(original, -1000)), "×1 tiene que devolver la original"
    vieja, _ = R.simular(original, PX, I0=None)
    cuerpo = verdad > -900
    print(f"\n1. ×1: ruido agregado = {np.abs(r1).max():.1f} HU (método anterior, re-reconstruir "
          f"sin ruido, ya cambia la imagen: RMSE {np.sqrt(np.mean((vieja - original)[cuerpo] ** 2)):.1f} HU)")

    # 2-3. compensación y ley 1/√f
    print("\n2-3.  f     total compensado | adquisición real f·N | sin compensar | σ0/√f   (pulmón, HU)")
    for f in (0.5, 0.25, 0.1, 0.05):
        _, ra = sim.a_fraccion(f, rng)
        tot = sd_en(ruido0 + ra, pulmon)
        # adquisición real a f·N (lo que queremos imitar)
        rp = B.ruido_de_adquisicion(adq.p, f * adq.N_alta * adq.bowtie, adq.sigma_e2, rng)
        real = sd_en(B.reconstruir_ruido(rp, PX, N, adq.ventana), pulmon)
        # método anterior: sumar el ruido de f·N a la original, sin compensar
        rp_sc = B.ruido_de_adquisicion(adq.p, f * adq.N_alta * adq.bowtie, adq.sigma_e2, rng)
        sin_comp = sd_en(ruido0 + B.reconstruir_ruido(rp_sc, PX, N, adq.ventana), pulmon)
        print(f"     {f:<5g} {tot:12.1f}     | {real:12.1f}         | {sin_comp:10.1f}    | {s0_p / np.sqrt(f):6.1f}")
        assert abs(tot - real) / real < 0.08, f"compensación falla en f={f}"

    # 4. ruido electrónico
    sin_e = B.Adquisicion(original, PX).desde_dicom(enc, electronico=False)
    print("\n4. Ruido electrónico (σ total en pulmón / tejido, HU):")
    for f in (0.25, 0.05):
        _, a = sim.a_fraccion(f, np.random.default_rng(1))
        _, b = sin_e.a_fraccion(f, np.random.default_rng(1))
        print(f"   f={f:<5g} con {sd_en(ruido0 + a, pulmon):6.1f} / {sd_en(ruido0 + a, tejido):6.1f}"
              f"   sin {sd_en(ruido0 + b, pulmon):6.1f} / {sd_en(ruido0 + b, tejido):6.1f}")

    # 5. textura según la ventana: frecuencia media del espectro del ruido
    print("\n5. Ventana del FBP -> textura del ruido (misma dosis, f = 0,25):")
    for ventana in ("ramp", "shepp-logan", "hann"):
        sim.ventana = ventana
        _, r = sim.a_fraccion(0.25, np.random.default_rng(2))
        parche = r[200:300, 320:380]
        ps = np.abs(np.fft.fftshift(np.fft.fft2(parche - parche.mean()))) ** 2
        fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(parche.shape[0], PX)),
                             np.fft.fftshift(np.fft.fftfreq(parche.shape[1], PX)), indexing="ij")
        fmed = float((np.hypot(fy, fx) * ps).sum() / ps.sum())
        print(f"   {ventana:12s} σ {r[pulmon].std():6.1f} HU   frecuencia media del NPS {fmed:.2f} ciclos/mm")
    print("\nOK baja dosis")


if __name__ == "__main__":
    main()
