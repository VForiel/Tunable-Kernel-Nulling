"""Interfaces et affichage des cartes de transmission du nuller.

Contient des widgets et fonctions pour calculer/afficher les cartes de
transmission, gradients et exporter des images.
"""
from copy import deepcopy as copy
import astropy.units as u
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from phise import Context

def gui(ctx: Context=None, N: int=100):
    """
    GUI to visualize the transmission maps of the VLTI.
    Parameters
    ----------
    ctx : Context, optional
        The context to use for the transmission maps. If None, the default context is used.
        The default is None.
    N : int, optional
        Resolution of the maps. The default is 100 (for 100x100).
    """
    if ctx is None:
        ref_ctx = Context.get_VLTI()
        ref_ctx.interferometer.chip.σ = np.zeros(14) * u.um
    else:
        ref_ctx = ctx
    h_slider = widgets.FloatSlider(value=0, min=(ref_ctx.h - ref_ctx.Δh / 2).value, max=(ref_ctx.h + ref_ctx.Δh / 2).value, step=0.01, description='Hour angle:')
    l_slider = widgets.FloatSlider(value=ref_ctx.interferometer.l.to(u.deg).value, min=-90, max=90, step=0.01, description='Latitude:')
    δ_slider = widgets.FloatSlider(value=ref_ctx.target.δ.to(u.deg).value, min=-90, max=90, step=0.01, description='Declination:')
    reset = widgets.Button(description='Reset values')
    run = widgets.Button(description='Run')
    export = widgets.Button(description='Export')
    plot = widgets.Image()
    plot_gradient = widgets.Image()
    transmission = widgets.HTML()

    def update_plot(*args):
        run.button_style = 'warning'
        ctx = copy(ref_ctx)
        ctx.interferometer.l = l_slider.value * u.deg
        ctx.target.δ = δ_slider.value * u.deg
        ctx.h = h_slider.value * u.hourangle
        (img, txt) = ctx.plot_transmission_maps(N=N, return_plot=True)
        plot.value = img
        transmission.value = txt
        (img, txt) = ctx.plot_transmission_maps(N=N, return_plot=True, grad=True)
        plot_gradient.value = img
        run.button_style = ''

    def export_plot(*args):
        """"export_plot.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        ctx = copy(ref_ctx)
        ctx.interferometer.l = l_slider.value * u.deg
        ctx.target.δ = δ_slider.value * u.deg
        ctx.h = h_slider.value * u.hourangle
        ctx.plot_transmission_maps(N=N, return_plot=False)

    def reset_values(*args):
        """"reset_values.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        l_slider.value = ref_ctx.interferometer.l.to(u.deg).value
        δ_slider.value = ref_ctx.target.δ.to(u.deg).value
        h_slider.value = ref_ctx.h.to(u.deg).value
        run.color = 'blue'
        enable_run()

    def enable_run(*args):
        """"enable_run.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        run.button_style = 'success'
    reset.on_click(reset_values)
    h_slider.observe(enable_run)
    l_slider.observe(enable_run)
    δ_slider.observe(enable_run)
    run.on_click(update_plot)
    export.on_click(export_plot)
    display(widgets.VBox([h_slider, l_slider, δ_slider, widgets.HBox([reset, run, export]), plot, transmission, plot_gradient]))
    update_plot()