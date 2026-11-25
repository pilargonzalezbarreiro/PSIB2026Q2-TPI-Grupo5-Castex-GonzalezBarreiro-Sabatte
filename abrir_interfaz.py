"""Abre la interfaz gráfica del TPI en el navegador.

Cómo usarlo: abrir este archivo en VS Code y apretar ▶ ("Run Python File").
La primera vez instala Streamlit (tarda un minuto). Después se abre sola una
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


def main():
    if importlib.util.find_spec("streamlit") is None:
        print("Instalando Streamlit (solo la primera vez)...", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
        if r.returncode != 0:
            sys.exit("No se pudo instalar Streamlit. Mandar este mensaje.")
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
