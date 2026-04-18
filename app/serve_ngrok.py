"""
Launch the Streamlit app behind an ngrok tunnel for quick demos.

Usage:
    poetry run python app/serve_ngrok.py [--port 8501]

Requires NGROK_AUTHTOKEN in the environment (or in a .env file at project root).
Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pyngrok import conf, ngrok

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = Path(__file__).resolve().parent / "app.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    parser.add_argument("--region", default="sa", help="ngrok region (sa, us, eu, ...)")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    authtoken = os.getenv("NGROK_AUTHTOKEN")
    if not authtoken:
        print("ERROR: NGROK_AUTHTOKEN não definido. Adicione ao .env ou exporte-o.")
        return 1

    conf.get_default().auth_token = authtoken
    conf.get_default().region = args.region

    streamlit_proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=PROJECT_ROOT,
    )

    # Give Streamlit a moment to bind before opening the tunnel
    time.sleep(3)

    try:
        tunnel = ngrok.connect(args.port, "http")
        print("\n" + "=" * 60)
        print(f"  Streamlit local: http://localhost:{args.port}")
        print(f"  ngrok public:    {tunnel.public_url}")
        print("=" * 60 + "\n  Ctrl+C para encerrar\n")

        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\nEncerrando...")
    finally:
        ngrok.kill()
        if streamlit_proc.poll() is None:
            streamlit_proc.send_signal(signal.SIGTERM)
            try:
                streamlit_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_proc.kill()

    return 0


if __name__ == "__main__":
    sys.exit(main())
