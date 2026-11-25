"""Corre toda la parte de baja dosis en orden, sin escribir comandos.

Cómo usarlo: abrir este archivo en VS Code (o el editor que usen) y apretar
el botón ▶ ("Run Python File"). Todo lo que se imprime queda también guardado
en resultados/log_baja_dosis.txt.

Pensado para dejarlo corriendo solo (por ejemplo, de noche): con
HASTA = "prueba_y_todo" corre primero la prueba chica y, SOLO si salió bien,
sigue con la corrida completa. Mientras corre, la Mac no se suspende
(caffeinate), pero tiene que estar enchufada y con la tapa ABIERTA: con la tapa
cerrada macOS la duerme igual.

HASTA elige hasta dónde correr:
  "07a"    corre los encabezados, el test y la calibración, y para.
  "prueba" corre TODA la segunda parte pero chiquita (pocos nódulos, pocos
           sorteos): ~10-20 min. Sirve para ver que nada se rompe antes de
           dejar la corrida larga. Sus resultados NO valen (se pisan después).
  "todo"   la corrida completa: incertidumbre, 07 con los tres anclajes,
           comparación y 08 con los tres anclajes. Varias horas.
  "prueba_y_todo"  la prueba y, si anduvo, "todo" (para dejarlo de noche).
  "resto"  para cuando el 07 y el 08 ya corrieron: el 08 con la dosis de uso
           ×1/10 (sensibilidad del criterio) y la incertidumbre en versión
           acotada (ruido: 110 nódulos; corriente × Φ e híbrido: 30 cada uno).
           Retoma los nódulos que ya estaban hechos. Unas 4-5 h.
  "electronico"  estima el ruido electrónico con los datos de LIDC (07a) y
           corre el 07 híbrido con ese valor y con el límite superior de su
           intervalo. Correrlo DESPUÉS de que termine la incertidumbre. ~30 min.

Requisito: los cortes ya extraídos en datos/cortes (scripts 01 a 04 de antes).
"""
import subprocess
import sys
import time
from pathlib import Path

HASTA = "electronico"   # "07a", "prueba", "todo", "prueba_y_todo", "resto" o "electronico"
# Nódulos del análisis de incertidumbre: "auto" = se agregan de a uno hasta que
# sumar más ya no cambia el resultado (ver comparaciones/incertidumbre_dosis.py)
CASOS_INCERTIDUMBRE = "auto"

RAIZ = Path(__file__).resolve().parent
LOG = RAIZ / "resultados" / "log_baja_dosis.txt"

PRIMERA_PARTE = [
    ("Encabezados DICOM (baja de nuevo las series si se borraron)",
     ["scripts/03b_encabezados_dicom.py", "--descargar", "--borrar-dicom"]),
    ("Test de la simulación en el fantoma", ["tests/test_baja_dosis.py"]),
    ("Calibración y errores del ruido (07a)", ["scripts/07a_calibrar_dosis.py"]),
]
def segunda_parte(prueba):
    """Primero lo principal (07 y 08, minutos); al final la incertidumbre, que
    es lo largo y guarda cada nódulo apenas termina (si se corta, se retoma)."""
    lim = ["--limite", "12"] if prueba else []
    pasos = []
    for a, nombre in (("mas", "por corriente × Φ"), ("ruido", "por ruido"), ("hibrido", "híbrido")):
        pasos.append((f"07 con anclaje {nombre}", ["scripts/07_baja_dosis.py", "--anclaje", a, *lim]))
    pasos.append(("Comparación de anclajes", ["comparaciones/comparar_anclajes.py"]))
    for a in ("mas", "ruido", "hibrido"):
        pasos.append((f"08 propuesta (anclaje {a})", ["scripts/08_propuesta.py", "--anclaje", a]))
    base = ["comparaciones/incertidumbre_dosis.py", "--carpeta", "datos/cortes"]
    if prueba:
        pasos.append(("Incertidumbre (prueba)", base + ["--anclaje", "todos", "--casos", "2", "--muestras", "8"]))
    else:
        pasos += [
            ("Incertidumbre, anclaje por ruido, TODOS los nódulos",
             base + ["--anclaje", "ruido", "--casos", "todos", "--muestras", "50"]),
            ("Incertidumbre, anclaje por corriente × Φ (hasta converger)",
             base + ["--anclaje", "mas", "--casos", "auto", "--muestras", "150"]),
            ("Incertidumbre, anclaje híbrido (hasta converger)",
             base + ["--anclaje", "hibrido", "--casos", "auto", "--muestras", "150"]),
            ("Gráfico comparando los tres anclajes", base + ["--solo-comparar"]),
        ]
    return pasos


def correr(nombre, args, log):
    titulo = f"\n{'=' * 70}\n{nombre}\n  python {' '.join(args)}\n{'=' * 70}\n"
    print(titulo, flush=True)
    log.write(titulo)
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, *args], cwd=RAIZ, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for linea in proc.stdout:
        print(linea, end="", flush=True)
        log.write(linea)
    proc.wait()
    fin = f"\n[{nombre}: {'OK' if proc.returncode == 0 else 'ERROR ' + str(proc.returncode)}, " \
          f"{(time.time() - t0) / 60:.1f} min]\n"
    print(fin, flush=True)
    log.write(fin)
    log.flush()
    return proc.returncode == 0


def no_dormir():
    """En Mac, que la compu no se suspenda mientras corre este programa."""
    if sys.platform == "darwin":
        import os
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            print("La Mac no se va a suspender mientras corre (dejarla enchufada y con la tapa abierta).")
        except OSError:
            print("AVISO: no se pudo evitar la suspensión; desactivarla a mano en Ajustes.")


def resto():
    """El 08 con otra dosis de uso y la incertidumbre acotada (ver HASTA)."""
    pasos = [(f"08 propuesta (anclaje {a}, dosis de uso ×1/10)",
              ["scripts/08_propuesta.py", "--anclaje", a, "--dosis-uso", "10"])
             for a in ("mas", "ruido", "hibrido")]
    base = ["comparaciones/incertidumbre_dosis.py", "--carpeta", "datos/cortes", "--muestras", "50"]
    pasos += [
        ("Incertidumbre, anclaje por ruido (110 nódulos; retoma los hechos)",
         base + ["--anclaje", "ruido", "--casos", "110"]),
        ("Incertidumbre, anclaje por corriente × Φ (30 nódulos)", base + ["--anclaje", "mas", "--casos", "30"]),
        ("Incertidumbre, anclaje híbrido (30 nódulos)", base + ["--anclaje", "hibrido", "--casos", "30"]),
        ("Gráfico comparando los tres anclajes", base + ["--solo-comparar"]),
    ]
    return pasos


def electronico():
    """σe² medido en LIDC y el 07 híbrido con ese valor (ver HASTA)."""
    return [
        ("Calibración, errores del ruido y ruido electrónico estimado con LIDC (07a)",
         ["scripts/07a_calibrar_dosis.py"]),
        ("07 híbrido con el σe² estimado en LIDC",
         ["scripts/07_baja_dosis.py", "--anclaje", "hibrido", "--sigma-e2", "lidc"]),
        ("07 híbrido con el límite superior del IC 95 % de σe²",
         ["scripts/07_baja_dosis.py", "--anclaje", "hibrido", "--sigma-e2", "lidc_sup"]),
    ]


def pasos_de(hasta):
    if hasta == "resto":
        return resto()
    if hasta == "electronico":
        return electronico()
    pasos = PRIMERA_PARTE + (segunda_parte(hasta == "prueba") if hasta in ("prueba", "todo") else [])
    if (RAIZ / "resultados" / "encabezados_dicom.csv").exists():
        # ya se leyeron (tarda ~45 min): no volver a bajar las tomografías
        pasos = [p for p in pasos if not p[1][0].startswith("scripts/03b")]
    return pasos


def correr_etapa(hasta, log):
    print(f"\n{'#' * 70}\n# ETAPA: {hasta}\n{'#' * 70}", flush=True)
    log.write(f"\n{'#' * 70}\n# ETAPA: {hasta}\n{'#' * 70}\n")
    if (RAIZ / "resultados" / "encabezados_dicom.csv").exists():
        print("Encabezados ya leídos (resultados/encabezados_dicom.csv): se saltea el 03b.")
    for nombre, args in pasos_de(hasta):
        if not correr(nombre, args, log):
            print(f"\nSe frenó en: {nombre} (etapa {hasta}). Mandar {LOG} para ver qué pasó.")
            log.write(f"\nSe frenó en: {nombre} (etapa {hasta})\n")
            return False
    return True


def main():
    LOG.parent.mkdir(exist_ok=True)
    no_dormir()
    t0 = time.time()
    etapas = ["prueba", "todo"] if HASTA == "prueba_y_todo" else [HASTA]
    with open(LOG, "a", encoding="utf-8") as log:  # "a": no borra el log de corridas anteriores
        for etapa in etapas:
            if not correr_etapa(etapa, log):
                return
        fin = f"\nListo ({(time.time() - t0) / 3600:.1f} h). Todo quedó en {LOG}"
        print(fin)
        log.write(fin + "\n")
    if HASTA == "07a":
        print("Siguiente: mandar ese archivo.")
    elif HASTA == "prueba":
        print("La prueba anduvo. Siguiente: cambiar HASTA = \"todo\" (o \"prueba_y_todo\") y dejarlo corriendo.")


if __name__ == "__main__":
    main()
