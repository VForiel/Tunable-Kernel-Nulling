import numpy as np
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
import astropy.units as u
from scipy.stats import linregress
from copy import deepcopy as copy
from phise import Context

def genetic_approach(ctx: Context=None, β: float=0.9, verbose=False, figsize=(10, 10), σ_rms=None):
    if ctx is None:
        ctx = Context.get_VLTI()
    else:
        ctx = copy(ctx)
    ctx.Γ = 0 * u.nm
    ctx.target.companions = []
    if σ_rms is None:
        σ_rms = ctx.interferometer.λ
    ctx.interferometer.chip.σ = np.abs(np.random.normal(0, 10, 14)) * σ_rms
    print_kernel_null_depth_lab_space_atm(ctx)
    ctx.calibrate_gen(β=β, plot=True, verbose=verbose, figsize=figsize)
    print_kernel_null_depth_lab_space_atm(ctx)
    return ctx

def obstruction_approach(ctx: Context=None, n: int=1000, σ_rms=None, figsize=(10, 10)):
    if ctx is None:
        ctx = Context.get_VLTI()
    else:
        ctx = copy(ctx)
    ctx.Γ = 0 * u.nm
    ctx.target.companions = []
    if σ_rms is None:
        σ_rms = ctx.interferometer.λ
    ctx.interferometer.chip.σ = np.abs(np.random.normal(0, 10, 14)) * σ_rms
    print_kernel_null_depth_lab_space_atm(ctx)
    ctx.calibrate_obs(n=n, plot=True, figsize=(10, 10))
    print_kernel_null_depth_lab_space_atm(ctx)
    return ctx

def print_kernel_null_depth_lab_space_atm(ctx: Context):
    ctx = copy(ctx)
    ctx.Γ = 0 * u.nm
    print('Performances in lab (Γ=0)')
    print_kernel_null_depth(ctx)
    ctx.Γ = 1 * u.nm
    print('\nPerformances in space (Γ=1 nm)')
    print_kernel_null_depth(ctx)
    ctx.Γ = 100 * u.nm
    print('\nPerformances in atmosphere (Γ=100 nm)')
    print_kernel_null_depth(ctx)

def print_kernel_null_depth(ctx: Context, N=100):
    kernels = np.empty((N, 3))
    bright = np.empty(N)
    for i in range(N):
        # observe() now returns raw output intensities; process to get kernels
        outs = ctx.observe()
        k = ctx.interferometer.chip.process_outputs(outs)
        b = outs[0]
        kernels[i] = k
        bright[i] = b
    k_mean = np.mean(kernels, axis=0)
    k_med = np.median(kernels, axis=0)
    k_std = np.std(kernels, axis=0)
    b_mean = np.mean(bright)
    b_med = np.median(bright)
    b_std = np.std(bright)
    print(f'Achieved Kernel-Null depth:')
    print('   Mean: ' + ' | '.join([f'{i / b_mean:.2e}' for i in k_mean]))
    print('   Med:  ' + ' | '.join([f'{i / b_mean:.2e}' for i in k_med]))
    print('   Std:  ' + ' | '.join([f'{i / b_mean:.2e}' for i in k_std]))

def compare_approaches(ctx: Context=None, β: float=0.9, n: int=10000, figsize=(10, 10)):
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.monochromatic = True
    else:
        ctx = copy(ctx)
    kn = ctx.interferometer.chip
    λ = ctx.interferometer.λ
    ctx.Γ = 0 * u.nm
    ctx.target.companions = []
    res = 5
    (βs, dβ) = np.linspace(0.5, β, res, endpoint=True, retstep=True)
    Ns = np.logspace(1, np.log10(n), res, endpoint=True, dtype=int)
    samples = 10
    shots = []
    for β in βs:
        for i in range(samples):
            print(f'Gen.: β={β:.3f}, sample={i + 1}/{samples}          ', end='\r')
            ctx.interferometer.chip.σ = np.abs(np.random.normal(0, 1, len(kn.σ))) * λ
            history = ctx.calibrate_gen(β=β, verbose=False)
            ψ = np.ones(4) * (1 + 0j) * np.sqrt(1 / 4)
            # get_output_fields returns complex fields; square to get intensities
            out_fields = ctx.interferometer.chip.get_output_fields(ψ=ψ, λ=λ)
            raw_outs = np.abs(out_fields) ** 2
            di = raw_outs[1:]
            k = np.array([di[0] - di[1], di[2] - di[3], di[4] - di[5]])
            b = raw_outs[0]
            depth = np.sum(np.abs(k)) / b
            shots.append((len(history['depth']), depth))
    (x, y) = zip(*shots)
    x = np.array(x)
    y = np.array(y)
    plt.figure(figsize=figsize, constrained_layout=True)
    plt.scatter(np.random.uniform(x - x / 10, x + x / 10), y, c='tab:blue', s=5, label='Genetic')
    (slope, intercept, r_value, p_value, std_err) = linregress(np.log10(x), np.log10(y))
    print(slope, intercept)
    x_values = np.linspace(min(x), max(x), 500)
    y_fit = 10 ** intercept * x_values ** slope
    plt.plot(x_values, y_fit, 'tab:cyan', linestyle='--', label=f'Gen. fit')
    print('')
    shots = []
    for (j, n) in enumerate(Ns):
        for i in range(samples):
            print(f'Obs.: n={n}, sample={i + 1}/{samples}          ', end='\r')
            ctx.interferometer.chip.σ = np.abs(np.random.normal(0, 1, len(kn.σ))) * λ
            ctx.calibrate_obs(n=n, plot=False)
            ψ = np.ones(4) * (1 + 0j) * np.sqrt(1 / 4)
            out_fields = ctx.interferometer.chip.get_output_fields(ψ=ψ, λ=λ)
            raw_outs = np.abs(out_fields) ** 2
            di = raw_outs[1:]
            k = np.array([di[0] - di[1], di[2] - di[3], di[4] - di[5]])
            b = raw_outs[0]
            depth = np.sum(np.abs(k)) / b
            shots.append((7 * n, depth))
    (x, y) = zip(*shots)
    x = np.array(x)
    y = np.array(y)
    plt.scatter(np.random.uniform(x - x / 10, x + x / 10), y, c='tab:orange', s=5, label='Obstruction')
    (slope, intercept, r_value, p_value, std_err) = linregress(np.log10(x), np.log10(y))
    print(slope, intercept)
    x_values = np.linspace(min(x), max(x), 500)
    y_fit = 10 ** intercept * x_values ** slope
    plt.plot(x_values, y_fit, 'tab:red', linestyle='--', label=f'Obs. fit')
    plt.xlabel('# of iterations')
    plt.xscale('log')
    plt.ylabel('Depth')
    plt.yscale('log')
    plt.title('Efficiency of the calibration approaches')
    plt.legend()
    plt.show()