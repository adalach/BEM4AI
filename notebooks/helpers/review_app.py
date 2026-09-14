"""Launch the local review apps from notebook cells."""

from __future__ import annotations

import importlib
import threading
from html import escape

__all__ = ["launch", "link", "shutdown"]

_RUNNING: dict[str, tuple[object, threading.Thread]] = {}


def shutdown(name: str) -> None:
    """Stop a review server started earlier under `name`, if it is still up."""
    server, thread = _RUNNING.pop(name, (None, None))
    if server is None:
        return
    if thread.is_alive():
        server.shutdown()
    server.server_close()


def launch(module_name: str, *, port: int = 0, **server_kwargs) -> str:
    """Start a review app and return the URL to open.

    `module_name` is the backend module, for example
    ``scripts.review_apps.serve_footprint_selector``. `port` of 0 lets the
    operating system pick one. Any other keyword argument is passed straight to
    that module's ``start_server``.

    The backend is reloaded on every launch, so editing the app and re-running
    the cell is enough to see the change without restarting the kernel.
    """
    shutdown(module_name)

    backend = importlib.reload(importlib.import_module(module_name))
    server, thread, url = backend.start_server(port=port, **server_kwargs)
    _RUNNING[module_name] = (server, thread)
    return url


def link(url: str, text: str = "Open the review app"):
    """The URL as a clickable link in the notebook output."""
    from IPython.display import HTML

    href = escape(url, quote=True)
    label = escape(text)
    return HTML(f'<a href="{href}" target="_blank" rel="noopener">{label}</a>')
