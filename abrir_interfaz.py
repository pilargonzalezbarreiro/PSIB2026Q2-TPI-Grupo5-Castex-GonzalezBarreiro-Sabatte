"""Abre la interfaz gráfica del TPI en el navegador.

Cómo usarlo: abrir este archivo en VS Code y apretar ▶ ("Run Python File").
La primera vez instala lo que falte (Streamlit y las librerías del TP; tarda un par de minutos). Después se abre sola una
pestaña del navegador con la interfaz. Para cerrarla: en la terminal de
VS Code, Control + C (o el ícono del tacho de basura).

Si el navegador no se abre solo, entrar a la dirección que aparece en la
terminal (http://localhost:8501).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


# módulo que se importa -> paquete que hay que instalar
NECESARIOS = {"numpy": "numpy", "scipy": "scipy", "skimage": "scikit-image", "pandas": "pandas",
              "matplotlib": "matplotlib", "PIL": "Pillow", "streamlit": "streamlit"}


def main():
    faltan_paquetes = [paq for mod, paq in NECESARIOS.items() if importlib.util.find_spec(mod) is None]
    if faltan_paquetes:
        print("Instalando lo que falta (solo la primera vez): " + ", ".join(faltan_paquetes), flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", *faltan_paquetes])
        if r.returncode != 0:
            sys.exit("No se pudieron instalar las librerías. Mandar este mensaje completo.")
    faltan = [p for p in ("resultados/caracteristicas.csv", "datos/cortes") if not (RAIZ / p).exists()]
    if faltan:
        print("AVISO: faltan", ", ".join(faltan), "(la interfaz lo va a avisar).")
    print("Abriendo la interfaz... (para cerrarla: Control + C en esta terminal)", flush=True)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(RAIZ / "interfaz" / "app.py"),
                        "--browser.gatherUsageStats", "false"], cwd=RAIZ)
    except KeyboardInterrupt:
        print("\nInterfaz cerrada.")


if __name__ == "__main__":
    main()
