"""Analyse de la sensibilité du nuller au bruit d'entrée.

Fonctions pour calibrer, simuler et tracer la dépendance de la
profondeur de nulling à l'OPD et autres paramètres bruités.
"""
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from copy import deepcopy as copy
from phise.classes.context import Context

def plot(ctx: Context=None, β=0.5, n=1000, γ=10*u.nm, figsize=(15, 5)):
    """
    Plot the sensitivity to input noise

    Parameters
    ----------
    ctx : Context
        The context to use for the plot.
    β : float
        The beta parameter for the genetic calibration approach.
    n : int
        The number of observations for the obstruction calibration approach.
    γ : Quantity
        Intrinsic OPD RMS.
    figsize : tuple
        The figure size.

    Returns
    -------
    - None
    """
    if ctx is None:
        ctx_perturbated = Context.get_VLTI()
        ctx_perturbated.monochromatic = True
    else:
        ctx_perturbated = copy(ctx)
    ctx_perturbated.name = f'${γ:.0f}$ intrinsic OPD RMS'
    ctx_perturbated.Δh = ctx_perturbated.interferometer.camera.e.to(u.hour).value * u.hourangle
    ctx_perturbated.target.companions = []
    ctx_perturbated.interferometer.chip.σ = np.random.normal(0, γ.to(u.nm).value, 14) * u.nm

    # Ideal context
    ctx_ideal = copy(ctx_perturbated)
    ctx_ideal.name = 'Ideal case'
    ctx_ideal.interferometer.chip.σ = np.zeros(14) * u.nm
    ctx_ideal.interferometer.chip.φ = np.zeros(14) * u.nm

    # # Trial & Error calibrated context
    print('⌛ Calibrating using trial&error approach...')
    ctx_gen = copy(ctx_perturbated)
    ctx_gen.name = 'Trial & Error calibration'
    ctx_gen.Γ = 0 * u.nm
    ctx_gen.calibrate_gen(β=β)
    print('✅ Done.')

    # Obstruction calibrated context
    print('⌛ Calibrating using obstruction approach...')
    ctx_obs = copy(ctx_perturbated)
    ctx_obs.name = 'Obstruction calibration'
    ctx_obs.Γ = 0 * u.nm
    ctx_obs.calibrate_obs(n=n)
    print('✅ Done.')

    # Prepare OPD RMS range
    (Γ_range, step) = np.linspace(0, ctx_perturbated.Γ.to(u.nm).value, 10, retstep=True)
    Γ_range *= u.nm
    step *= u.nm
    stds = []
    (_, ax) = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    
    # Loop over atmospheric OPD RMS values
    print('⌛ Computing noise sensitivity...')
    for (i, Γ) in enumerate(Γ_range):
        print(f'{i + 1 / len(Γ_range) * 100}% (Γ = {Γ:.1f})', end='\r')

        # Loop over contexts
        context_list = [ctx_ideal, ctx_perturbated, ctx_gen, ctx_obs]
        colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red']
        for (c, ctx) in enumerate(context_list):
            ctx.Γ = Γ
            outs = ctx.observation_serie(n=1000)
            
            data = np.empty(1000)
            for j in range(1000):
                data[j] = ctx.interferometer.chip.process_outputs(outs[j, 0, :])[0] / outs[j, 0, 0]

            stds.append(np.std(data))
            x_dispersion = np.random.normal(Γ.value + (c - 1.5) * step.value / 5, step.value / 20, len(data))
            ax.scatter(x_dispersion, data, color=colors[c], s=5 if i == 0 else 0.1, alpha=1 if i == 0 else 1, label=ctx.name if i == 0 else None)
            ax.boxplot(data, vert=True, positions=[Γ.value + (c - 1.5) * step.value / 5], widths=step.value / 5, showfliers=False, manage_ticks=False)
    print('✅ Done.                      ')
    ax.set_ylim(-max(stds), max(stds))
    ax.set_xlabel(f'Upstream OPD RMS ({Γ_range.unit})')
    ax.set_ylabel('Kernel-Null depth')
    ax.set_title('Sensitivity to noise')
    ax.legend()