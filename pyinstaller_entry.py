"""Entry point for PyInstaller Windows executable.

This script wraps the Shiny application so it can be bundled into a
standalone Windows executable. It launches the local web server and
automatically opens the user's default web browser.
"""

import threading
import time
import webbrowser

from app import app
from shiny import run_app


def _open_browser():
    """Open the default web browser after a short delay."""
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    print("=" * 60)
    print("Spike Doctor is starting...")
    print("Please keep this window open while using the application.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)

    # Open browser in a background thread so it doesn't block the server
    threading.Thread(target=_open_browser, daemon=True).start()

    run_app(app, host="127.0.0.1", port=8000)
