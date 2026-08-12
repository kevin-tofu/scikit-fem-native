"""Exceptions shared by typed-form tracing and assembly dispatch."""


class UnsupportedNativeForm(Exception):
    """Raised when a form cannot be assembled by the native backend."""


__all__ = ["UnsupportedNativeForm"]
