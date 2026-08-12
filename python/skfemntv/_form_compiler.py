"""Shared tracing boundary between user form functions and typed nodes."""

from __future__ import annotations

from ._errors import UnsupportedNativeForm
from ._form_parameters import Parameters,parameter_values


def trace_expression(function,fields,geometry,parameters,*,context):
    """Evaluate a form with typed fields and consistently preserve diagnostics."""
    try:
        return function(
            *fields,
            Parameters(parameter_values(geometry,parameters)),
        )
    except (UnsupportedNativeForm,ValueError):
        raise
    except Exception as error:
        raise UnsupportedNativeForm(
            f"{context} contains an unsupported operation: {error}"
        ) from error


def extract_terms(expression,term_type,sum_type,*,message):
    """Normalize one typed term or a typed sum to an immutable term tuple."""
    if isinstance(expression,sum_type):
        return expression.terms
    if isinstance(expression,term_type):
        return (expression,)
    raise UnsupportedNativeForm(message)


__all__=["extract_terms","trace_expression"]
