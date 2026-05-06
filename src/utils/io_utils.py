"""Small I/O / stdout helpers."""
from __future__ import annotations

import contextlib
import io
from typing import Any, Callable


def silent(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Run ``fn(*args, **kwargs)`` while suppressing anything it prints to stdout.

    Useful around very chatty third-party calls (e.g. Wikidata helpers) so the
    notebook stays readable. The function's return value is forwarded unchanged.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@contextlib.contextmanager
def silent_stdout():
    """Context-manager flavour of :func:`silent` for ``with`` blocks."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield
