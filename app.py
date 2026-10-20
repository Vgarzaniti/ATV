"""ATV Web Application launcher (Local Desktop UI & Server)."""

import webbrowser
import uvicorn

if __name__ == "__main__":
    print("=" * 75)
    print("🚀 [ATV AUTONOMOUS DJ STUDIO - WEB APP]")
    print("Abriendo la aplicación en tu navegador: http://127.0.0.1:8000")
    print("=" * 75)
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run("atv.web.server:app", host="127.0.0.1", port=8000, log_level="info")
