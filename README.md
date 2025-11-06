## PHISE

PHISE (PHotonic Interferometry Simulation for Exoplanets) is a Python package for simulation and analysis of interferometric instruments using layered photonic chips. It provides high-level classes (telescopes, interferometer, kernel nuller, camera, target scene) and numerical modules (propagation, coordinates, MMI recombiners, test statistics) to build scenarios, simulate the instrument chain, and visualize responses (transmission maps, projected baselines, null/dark/bright outputs, etc.).

The repository also ships demo notebooks and a complete documentation

> ⚠️ **Important note**
> 
> PHISE is currently under active development and is still a part of a PhD research project. The API and functionalities may be instable, not correctly documented or change at any time.

## Requirements and installation

- Python 3.11 or upper.
- Main dependencies: numpy, astropy, scipy, matplotlib, numba, ipywidgets, sympy, etc. (handled automatically).

Two installation paths:

1) Conda environment (recommended)

```powershell
conda env create -f environment.yml
conda activate phise
```

2) pip editable install (dev mode)

```powershell
pip install -e .
```

## Documentation

A complete documentation is available at https://phise.readthedocs.io/

## Design notes

- Physical quantities are handled with `astropy.units` and validated in property setters to ensure unit consistency.
- Heavy computations rely on `numpy` and `numba` where appropriate.
- High-level methods (`Context`) automatically propagate parameter changes (e.g., recompute projected positions and photon flux).

## Credits

- Lead author: Vincent Foriel.
- If you use PHISE in scientific work, please cite the repository and/or your related publications.

## Questions, bugs, contributions

Issues and contributions are welcome. Feel free to propose improvements (docs, tests, examples, new utilities in `modules/`, etc.) by opening an issue or a pull request.

