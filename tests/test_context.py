"""
Unit tests for the Context class.

Each test is commented to explain the goal. When an exact numeric
expectation is not obvious (depends on internal constants or random
generation), a #TODO is left so the maintainer can fill the expected value
later.
"""
from __future__ import annotations

import numpy as np
import astropy.units as u

import pytest

from phise.classes.context import Context


def test_factory_creates_context():
    """Ensure the `get_VLTI` factory returns a Context instance.

    Goal: make sure a full construction (with Interferometer, Camera,
    Chip...) does not raise and that basic attributes are initialized.
    """
    ctx = Context.get_VLTI()
    assert isinstance(ctx, Context)
    assert isinstance(ctx.name, str)
    
    ctx = Context.get_LIFE()
    assert isinstance(ctx, Context)
    assert isinstance(ctx.name, str)

def test_str_and_repr_contains_name():
    """Ensure __str__ and __repr__ include the context name.

    This helps with debugging and readability.
    """
    ctx = Context.get_VLTI()
    s = str(ctx)
    r = repr(ctx)
    assert ctx.name in s
    assert ctx.name in r

def test_property_types_and_units():
    """Check types/units of main properties (h, Δh, Γ, pf, p).

    We assert these properties return Quantities (when appropriate) and
    have consistent units.
    """
    ctx = Context.get_VLTI()

    assert hasattr(ctx, 'h') and hasattr(ctx, 'Δh') and hasattr(ctx, 'Γ')
    assert isinstance(ctx.h, u.Quantity)
    assert isinstance(ctx.Δh, u.Quantity)
    assert isinstance(ctx.Γ, u.Quantity)

    # p and pf should be Quantities (p in meters, pf in photons/s)
    assert isinstance(ctx.p, u.Quantity)
    assert isinstance(ctx.pf, u.Quantity)


def test_setters_type_validation_and_errors():
    """Verify that setters validate types and raise errors.

    - interferometer and target expect specific objects -> TypeError
    - h and Δh must be Quantities with compatible units
    - Γ must be a length Quantity
    - p is read-only and assigning to it should raise ValueError
    """
    ctx = Context.get_VLTI()

    # wrong type for interferometer/target
    with pytest.raises(TypeError):
        ctx.interferometer = object()

    with pytest.raises(TypeError):
        ctx.target = object()

    # wrong type for h
    with pytest.raises(TypeError):
        ctx.h = 1  # not a Quantity

    # Quantity but wrong unit for h
    with pytest.raises(ValueError):
        ctx.h = (1 * u.m)  # should be an hour-angle

    # Δh must be a Quantity
    with pytest.raises(TypeError):
        ctx.Δh = 1

    # Δh too small (smaller than camera.e exposure time) -> ValueError
    with pytest.raises(ValueError):
        ctx.Δh = (1 * u.s)

    # Γ must be a Quantity
    with pytest.raises(TypeError):
        ctx.Γ = 1

    # Γ wrong unit (e.g. second) -> ValueError
    with pytest.raises(ValueError):
        ctx.Γ = (1 * u.s)

    # p is read-only
    with pytest.raises(ValueError):
        ctx.p = 0

    with pytest.raises(TypeError):
        ctx.monochromatic = 'yes'

    with pytest.raises(TypeError):
        ctx.name = 123

def test_get_input_fields_shape_and_dtype():
    """Check that `get_input_fields` returns a complex array shaped
    (n_objects, n_telescopes).

    We verify shape and type; numeric values depend on random draws (noise),
    so we don't assert exact values here.
    """
    ctx = Context.get_VLTI()
    fields = ctx.get_input_fields()

    nb_objects = 1 + len(ctx.target.companions)
    nb_tel = len(ctx.interferometer.telescopes)

    assert isinstance(fields, np.ndarray)
    assert fields.dtype == np.complex128
    assert fields.shape == (nb_objects, nb_tel)


def test_get_h_range_properties():
    """Ensure `get_h_range` returns a numpy array of values in radians and
    length >= 1.
    """
    ctx = Context.get_VLTI()
    h_range = ctx.get_h_range()
    assert isinstance(h_range, np.ndarray)
    assert h_range.ndim == 1
    assert h_range.size >= 1


def test_plot_projected_positions_returns_image_bytes():
    """Light test for `plot_projected_positions` requesting the buffer.

    Goal: ensure that calling with `return_image=True` returns a bytes buffer
    (PNG image). We don't inspect binary content.
    """
    ctx = Context.get_VLTI()
    img = ctx.plot_projected_positions(N=5, return_image=True)
    assert isinstance(img, (bytes, bytearray))


def test_pf_is_quantity_and_length_matches_telescopes():
    """Ensure `pf` is a Quantity and its length matches the number of
    telescopes.
    """
    ctx = Context.get_VLTI()
    pf = ctx.pf
    assert isinstance(pf, u.Quantity)
    assert pf.shape[0] == len(ctx.interferometer.telescopes)
    # Exact numeric value depends on instrumental parameters -> #TODO

