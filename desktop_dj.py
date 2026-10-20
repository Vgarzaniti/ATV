"""ATV Autonomous DJ Desktop Application Entrypoint."""

import sys
print("Iniciando ATV Desktop Studio...")
from atv.ui.desktop_app import run_desktop_app

if __name__ == "__main__":
    run_desktop_app()
