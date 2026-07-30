from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterfaceField:
    """Values and optional gradients/normals on both sides of an interface."""

    master: np.ndarray
    slave: np.ndarray
    master_gradient: np.ndarray | None = None
    slave_gradient: np.ndarray | None = None
    master_normal: np.ndarray | None = None
    slave_normal: np.ndarray | None = None


def jump(field: InterfaceField):
    """Return the oriented jump ``master - slave``."""
    if hasattr(field, "_interface_transform"):
        return field._interface_transform("weights",(1.,-1.))
    if not isinstance(field, InterfaceField):
        raise TypeError("jump expects an InterfaceField")
    return field.master - field.slave


def avg(field: InterfaceField, weights=(0.5, 0.5)):
    """Return a user-weighted average of the two traces."""
    if hasattr(field, "_interface_transform"):
        return field._interface_transform("weights",weights)
    if not isinstance(field, InterfaceField):
        raise TypeError("avg expects an InterfaceField")
    master_weight, slave_weight = weights
    return master_weight * field.master + slave_weight * field.slave


def normal_grad(field: InterfaceField):
    """Return paired outward-normal derivatives as an InterfaceField."""
    if hasattr(field, "_interface_transform"):
        return field._interface_transform("kind","normal_gradient")
    if not isinstance(field, InterfaceField):
        raise TypeError("normal_grad expects an InterfaceField")
    if (
        field.master_gradient is None
        or field.slave_gradient is None
        or field.master_normal is None
        or field.slave_normal is None
    ):
        raise ValueError(
            "normal_grad requires gradients and normals on both sides"
        )
    master = np.einsum(
        "i... ,i...->...", field.master_gradient, field.master_normal
    )
    slave = np.einsum(
        "i... ,i...->...", field.slave_gradient, field.slave_normal
    )
    return InterfaceField(master=master, slave=slave)
