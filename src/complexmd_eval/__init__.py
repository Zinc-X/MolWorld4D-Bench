"""ComplexMD-Eval public API."""

from ._version import __version__

__all__ = ["ProtocolError", "__version__", "evaluate_protocol", "load_protocol"]


def __getattr__(name: str):
    """Load the scientific pipeline lazily so diagnostics remain lightweight."""
    if name == "evaluate_protocol":
        from .pipeline import evaluate_protocol

        return evaluate_protocol
    if name in {"ProtocolError", "load_protocol"}:
        from .protocol import ProtocolError, load_protocol

        return {"ProtocolError": ProtocolError, "load_protocol": load_protocol}[name]
    raise AttributeError(name)
