from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path


APP_TITLE = "작업자 안전모 착용 자동 탐지 시스템"
HOST = "127.0.0.1"
PORT = 8501
ROOT = Path(__file__).resolve().parent


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def _wait_for_streamlit(timeout_seconds: int = 40) -> None:
    start = time.time()
    while time.time() - start < timeout_seconds:
        if _is_port_open(HOST, PORT):
            return
        time.sleep(0.5)
    raise RuntimeError("Streamlit server did not start in time.")


def _start_streamlit() -> subprocess.Popen:
    if _is_port_open(HOST, PORT):
        return None

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.address",
        HOST,
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0,
    )


def main() -> None:
    try:
        import webview
    except ImportError as exc:
        raise SystemExit(
            "pywebview가 설치되어 있지 않습니다.\n"
            "먼저 실행하세요: python -m pip install pywebview"
        ) from exc

    process = _start_streamlit()
    _wait_for_streamlit()

    try:
        webview.create_window(
            APP_TITLE,
            f"http://{HOST}:{PORT}",
            width=1440,
            height=920,
            min_size=(1180, 760),
        )
        webview.start()
    finally:
        if process is not None:
            process.terminate()


if __name__ == "__main__":
    main()
