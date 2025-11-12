"""Visualisations et transformations des distributions issues des
observations du kernel-nuller.

Fonctions pour calculer et tracer la distribution instantanée, les
évolutions temporelles et autres représentations des données.
"""
import numpy as np
import astropy.units as u
from copy import deepcopy as copy
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from phise import Context

π = np.pi
def get_K(Γ, α=1, β=1, φ2=π/3, φ3=π/4, φ4=π/6, n=10_000):

    # tirage bruit
    σ2 = np.random.normal(0, Γ, n)
    σ3 = np.random.normal(0, Γ, n)
    σ4 = np.random.normal(0, Γ, n)

    # champs étoile
    S1s = np.sqrt(α) * (1
               + np.exp(1j*(np.pi/2 + σ2))
               + np.exp(1j*(np.pi   + σ3))
               + np.exp(1j*(3*np.pi/2 + σ4)))

    S2s = np.sqrt(α) * (1
               + np.exp(1j*(3*np.pi/2 + σ2))
               + np.exp(1j*(np.pi     + σ3))
               + np.exp(1j*(np.pi/2   + σ4)))

    # champs planète (avec phases relatives)
    S1p = np.sqrt(β) * (1
               + np.exp(1j*(np.pi/2 + σ2 + φ2))
               + np.exp(1j*(np.pi   + σ3 + φ3))
               + np.exp(1j*(3*np.pi/2 + σ4 + φ4)))

    S2p = np.sqrt(β) * (1
               + np.exp(1j*(3*np.pi/2 + σ2 + φ2))
               + np.exp(1j*(np.pi     + σ3 + φ3))
               + np.exp(1j*(np.pi/2   + σ4 + φ4)))

    Ks = np.abs(S1s)**2 - np.abs(S2s)**2
    Kp = np.abs(S1p)**2 - np.abs(S2p)**2


    return Ks, Kp, Ks + Kp


def instant_distribution(ctx: Context=None, n=10000, stat=np.median, figsize=(10, 10)) -> np.ndarray:
    """
    Get the instantaneous distribution of the kernel nuller.

    Parameters
    ----------
    ctx : Context
        The context to use.
    n : int, optional
        The number of samples to take, by default 1000
    stat : function, optional
        The function to use to compute the statistic, by default np.median.

    Returns
    -------
    np.ndarray
        The instantaneous distribution of the kernel nuller.
    """

    # Set up context
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.um
        ctx.target.companions[0].c = 0.1
    else:
        ctx = copy(ctx)
        if ctx.target.companions == []:
            raise ValueError('No companion in the context. Please add a companion to the target.')
    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle

    ref_ctx = copy(ctx)
    ref_ctx.target.companions = []

    # Prepare data arrays
    data = np.empty((n, 3))
    ref_data = np.empty((n, 3))

    simple_model_data = np.empty((n,))

    # Sample data
    for i in range(n):

        # Distrib with companion(s)
        outs = ctx.observe()
        data[i, :] = ctx.interferometer.chip.process_outputs(outs)

        # Distrib with star only
        outs_ref = ref_ctx.observe()
        ref_data[i, :] = ref_ctx.interferometer.chip.process_outputs(outs_ref)

    # Distrib with companion on simple model
    ψi = ctx.get_input_fields()
    φi1, φi2, φi3, φi4 = np.angle(ψi[1])
    α = (np.mean(ctx.pf)).value
    β = α * ctx.target.companions[0].c
    Γ = ctx.Γ.to(u.nm).value / ctx.interferometer.λ.to(u.nm).value * 2 * np.pi
    _, _, simple_model_data = get_K(Γ=Γ, α=α, β=β, φ2=φi2-φi1, φ3=φi3-φi1, φ4=φi4-φi1, n=n)

    print(f"Simple: {α:.2e}, {β:.2e}")
    print(f"Full:" + str([f"{i:.2e}, " for i in np.abs(ctx.get_input_fields().flatten())**2]))
    print(f"{outs_ref[0]:.2e}")

    plt.hist(simple_model_data, bins=100)
    plt.title('Simple model kernel distribution')
    plt.xlabel('Kernel output')
    plt.ylabel('Occurrences')
    plt.show()

    # Plot histograms
    (_, axs) = plt.subplots(3, 1, figsize=figsize, constrained_layout=True, sharex=True)

    for k in range(3):

        # Get x limits
        combined = np.concatenate([data[:, k], ref_data[:, k]])
        (xmin_plot, xmax_plot) = np.percentile(combined, [5, 95])

        # Get bins
        bins = np.linspace(xmin_plot, xmax_plot, 2 * int(np.sqrt(n)) + 1)

        # Convert occurence to percentage
        weights_data = np.ones_like(data[:, k]) * (100.0 / data.shape[0])
        weights_ref = np.ones_like(ref_data[:, k]) * (100.0 / ref_data.shape[0])

        # Plot histograms
        axs[k].hist(data[:, k], label='With companion(s)', bins=bins, alpha=0.5, color='blue', weights=weights_data)
        axs[k].hist(ref_data[:, k], label='Star only', bins=bins, alpha=0.5, color='red', weights=weights_ref)

        if k == 0:
            # Plot simple model histogram
            weights_simple_model = np.ones_like(simple_model_data) * (100.0 / simple_model_data.shape[0])
            axs[k].hist(simple_model_data, label='Simple model', bins=bins, alpha=0.5, color='green', weights=weights_simple_model, histtype='stepfilled', linewidth=1.5)
            axs[k].axvline(stat(simple_model_data), color='green', linestyle='--')

        # Plot center line
        axs[k].axvline(stat(data[:, k]), color='blue', linestyle='--')
        axs[k].axvline(stat(ref_data[:, k]), color='red', linestyle='--')

        # Set labels
        axs[k].set_ylabel('Occurrences (%)')
        axs[k].set_title(f'Kernel {k + 1}')
        axs[k].legend()
        axs[k].set_xlim(xmin_plot, xmax_plot)
    axs[2].set_xlabel('Kernel output')
    plt.show()
    return (data, ref_data)

def time_evolution(ctx: Context=None, n=100, map=np.median) -> np.ndarray:
    """
    Get the time evolution of the kernel nuller.

    Parameters
    ----------
    ctx : Context
        The context to use.
    n : int, optional
        The number of samples to take at a given time, by default 1000.
    map : function, optional
        The function to use to map the data, by default np.median.

    Returns
    -------
    np.ndarray
        The time evolution of the kernel nuller. (n_h, 3)
    np.ndarray
        The reference time evolution of the kernel nuller (without input perturbation). (n_h, 3)
    """
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.um
        ctx.Γ = 10 * u.nm
    else:
        ctx = copy(ctx)
    ref_ctx = copy(ctx)
    ref_ctx.Γ = 0 * u.nm
    data = np.empty((len(ctx.get_h_range()), 3))
    ref_data = np.empty((len(ref_ctx.get_h_range()), 3))
    (_, k, b) = ctx.observation_serie(n=n)
    (_, ref_k, ref_b) = ref_ctx.observation_serie(n=1)
    k_depth = np.empty_like(k)
    ref_k_depth = np.empty_like(ref_k)
    for i in range(n):
        for h in range(len(ctx.get_h_range())):
            k_depth[i, h] = k[i, h] / b[i, h]
        for h in range(len(ref_ctx.get_h_range())):
            ref_k_depth[0, h] = ref_k[0, h] / ref_b[0, h]
    for h in range(len(ctx.get_h_range())):
        for k in range(3):
            data[h, k] = map(k_depth[:, h, k])
    for h in range(len(ref_ctx.get_h_range())):
        for k in range(3):
            ref_data[h, k] = ref_k_depth[0, h, k]
    (_, axs) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for k in range(3):
        axs[k].scatter(ctx.get_h_range(), data[:, k], label='Data')
        axs[k].plot(ref_ctx.get_h_range(), ref_data[:, k], label='Reference', linestyle='--')
        axs[k].set_ylabel(f'Kernel output')
        axs[k].set_xlabel('Time (hourangle)')
        axs[k].set_title(f'Kernel {k + 1}')
        axs[k].legend()
    plt.show()
    return (data, ref_data)