import numpy as np
import astropy.units as u
import astropy.constants as const
import numba as nb
from copy import deepcopy as copy
import matplotlib.pyplot as plt

from . import telescope
from .camera import Camera
try:
    import matplotlib.pyplot as plt
    try:
        plt.rcParams['image.origin'] = 'lower'
    except Exception:
        pass
except Exception:
    plt = None
from io import BytesIO
from scipy.optimize import curve_fit

# Internal libs
from .interferometer import Interferometer
from .target import Target
from .companion import Companion
from .archs import superkn, SuperKN

from ..modules import coordinates
from ..modules import signals
from ..modules import phase

class Context:
    """
    Observation context holding instrument, target and acquisition settings.

    Args:
        interferometer (Interferometer): Instrument and geometry.
        target (Target): Target definition (coordinates, flux, companions).
        h (u.Quantity): Local hour angle (central time) of the observation.
        Δh (u.Quantity): Time/Hour-angle span of the observation.
        Γ (u.Quantity): RMS cophasing error (length quantity).
        monochromatic (bool): If ``True``, use monochromatic approximation.
        name (str): Human-readable context name.
    """

    __slots__ = ('_initialized', '_interferometer', '_target', '_h', '_h_unit', '_Δh', '_Δh_unit', '_Γ', '_Γ_unit', '_name', '_p', '_ph', '_monochromatic')

    def __init__(
            self,
            interferometer:Interferometer,
            target:Target,
            h:u.Quantity,
            Δh: u.Quantity,
            Γ: u.Quantity,
            monochromatic = False,
            name:str = "Unnamed Context",
        ):

        self._initialized = False

        self.interferometer = copy(interferometer)
        self.interferometer._parent_ctx = self
        self.target = copy(target)
        self.target._parent_ctx = self
        self.h = h
        self.Δh = Δh
        self.Γ = Γ
        self.monochromatic = monochromatic
        self.name = name
        
        self._update_p() # define self.p
        self._update_pf() # define self.pf

        self._initialized = True

    # To string ---------------------------------------------------------------

    def __str__(self) -> str:
        res = f'Context "{self.name}"\n'
        res += "  " + "\n  ".join(str(self.interferometer).split("\n")) + "\n"
        res += "  " + "\n  ".join(str(self.target).split("\n")) + "\n"
        res += f'  h: {self.h:.2f}\n'
        res += f'  Δh: {self.Δh:.2f}\n'
        res += f'  Γ: {self.Γ:.2f}'
        return res
    
    def __repr__(self) -> str:
        return self.__str__()

    # Interferometer property -------------------------------------------------

    @property
    def interferometer(self) -> Interferometer:
        """
        Interferometer used in this context.
        """
        return self._interferometer
    
    @interferometer.setter
    def interferometer(self, interferometer:Interferometer):
        if not isinstance(interferometer, Interferometer):
            raise TypeError("interferometer must be an Interferometer object")
        self._interferometer = copy(interferometer)
        self.interferometer._parent_ctx = self
        if self._initialized:
            self._update_p()

    # Target property ---------------------------------------------------------
    
    @property
    def target(self) -> Target:
        """
        Target observed in this context.
        """
        return self._target
    
    @target.setter
    def target(self, target: Target):
        if not isinstance(target, Target):
            raise TypeError("target must be a Target object")
        self._target = copy(target)
        self.target._parent_ctx = self
        if self._initialized:
            self._update_p()

    # h property --------------------------------------------------------------

    @property
    def h(self) -> u.Quantity:
        """
        Local hour angle (central time) of the observation.
        """
        return (self._h * u.hourangle).to(self._h_unit)
    
    @h.setter
    def h(self, h: u.Quantity):
        if type(h) != u.Quantity:
            raise TypeError("h must be a Quantity")
        try:
            new_h = h.to(u.hourangle).value
        except u.UnitConversionError:
            raise ValueError("h must be in a hourangle unit")
        self._h_unit = h.unit
        self._h = new_h
        if self._initialized:
            self._update_p()

    # Δh property -------------------------------------------------------------

    @property
    def Δh(self) -> u.Quantity:
        """
        Time/Hour-angle span of the observation.
        """
        return (self._Δh * u.hourangle).to(self._Δh_unit)
    
    @Δh.setter
    def Δh(self, Δh: u.Quantity):
        if type(Δh) != u.Quantity:
            raise TypeError("Δh must be a Quantity")
        try:
            new_Δh = Δh.to(u.hourangle).value
        except u.UnitConversionError:
            raise ValueError("Δh must be in a hourangle unit")
        if new_Δh < (e := self.interferometer.camera.e.to(u.hour).value):
            raise ValueError(f"Δh must be upper or equal to the exposure time {e}")
        self._Δh = new_Δh
        self._Δh_unit = Δh.unit

    # Γ property --------------------------------------------------------------

    @property
    def Γ(self) -> u.Quantity:
        """
        RMS cophasing error (in length units) of the observation.
        """
        return (self._Γ * u.m).to(self._Γ_unit)
    
    @Γ.setter
    def Γ(self, Γ: u.Quantity):
        if type(Γ) != u.Quantity:
            raise TypeError("Γ must be a Quantity")
        try:
            new_Γ = Γ.to(u.m).value
        except u.UnitConversionError:
            raise ValueError("Γ must be in a distance unit")
        self._Γ = new_Γ
        self._Γ_unit = Γ.unit

    # p property --------------------------------------------------------------
    
    @property
    def p(self) -> u.Quantity:
        """
        (Read-only) Projected telescope positions in a plane perpendicular to the line of sight.
        """
        return self._p * u.m
        
    @p.setter
    def p(self, _):
        raise ValueError("p is a read-only property. Use _update_p() to set it accordingly to the other parameters in this context.")
    
    def _update_p(self):
        h = self.h.to(u.rad).value
        l = self.interferometer.l.to(u.rad).value
        δ = self.target.δ.to(u.rad).value
        r = np.array([i.r.to(u.m).value for i in self.interferometer.telescopes])
        
        self._p = project_position_jit(r, h, l, δ)
    
    # monochromatic property --------------------------------------------------

    @property
    def monochromatic(self) -> bool:
        """
        Whether to use the monochromatic approximation.
        """
        return self._monochromatic
    
    @monochromatic.setter
    def monochromatic(self, monochromatic: bool):
        if not isinstance(monochromatic, bool):
            raise TypeError("monochromatic must be a boolean")
        self._monochromatic = monochromatic
    
    # Name property -----------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Human-readable context name.
        """
        return self._name
    
    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self._name = name    

    # Photon flux -------------------------------------------------------------

    @property
    def pf(self) -> u.Quantity:
        """
        (Read-only) Photon flux per telescope. Shape: (n_telescopes,)
        """
        if not hasattr(self, "_ph"):
            raise AttributeError("pf is not defined.")
        return self._ph
    
    @pf.setter
    def pf(self, pf: u.Quantity):
        raise ValueError("pf is a read-only property.")

    def _update_pf(self):
        f = self.target.f.to(u.W / u.m**2 / u.nm)
        λ = self.interferometer.λ.to(u.m)
        η = self.interferometer.η
        Δλ = self.interferometer.Δλ.to(u.nm)
        a = np.array([i.a.to(u.m**2).value for i in self.interferometer.telescopes]) * u.m**2
        h = const.h
        c = const.c

        # Monochromatic case
        if Δλ == 0:
            Δλ = 1 * u.nm

        p = η * f * a * Δλ # Optical power [W]

        self._ph = p * λ / (h*c) # Photon flux [photons/s] (array of (n_telescopes,))
    
    # Plot projected positions over the time ----------------------------------

    def plot_projected_positions(
            self,
            N:int = 11,
            return_image = False,
        ):
        """Plot telescope positions over time.

        Args:
            N (int): Number of positions to plot.
            return_image (bool): If ``True``, return an image buffer instead of
                displaying it.

        Returns:
            Optional[bytes]: PNG image buffer when ``return_image=True``; otherwise ``None``.
        """
        _, ax = plt.subplots()

        h_range = np.linspace(self.h - self.Δh/2, self.h + self.Δh/2, N, endpoint=True)

        # Plot UT trajectory
        for i, h in enumerate(h_range):
            ctx = copy(self)
            ctx.h = h
            for j, (x, y) in enumerate(ctx.p):
                ax.scatter(x, y, label=f"Telescope {j+1}" if i==len(h_range)-1 else None, color=f"C{j}", s=1+14*i/len(h_range))

        print(self.interferometer.l)
        for (x, y) in self.p:
            ax.scatter(x, y, color="black", marker="+")

        ax.set_aspect("equal")
        ax.set_xlabel(f"x [{self.p.unit}]")
        ax.set_ylabel(f"y [{self.p.unit}]")
        ax.set_title(f"Projected telescope positions over the time ({ctx.Δh.to(u.hourangle).value * u.h} long)")
        plt.legend()

        if return_image:
            buffer = BytesIO()
            plt.savefig(buffer,format='png')
            plt.close()
            return buffer.getvalue()
        plt.show()

    # Transmission maps -------------------------------------------------------

    def get_transmission_maps(self, N:int) -> np.ndarray[float]:
        """Generate all kernel nuller transmission maps at a given resolution.

        Args:
            N (int): Map resolution.

        Returns:
            - np.ndarray[float]: Raw outputs transmission maps (nb_raw_outputs x N x N)
            - Optional[np.ndarray[float]]: Processed outputs transmission maps (nb_processed_outputs x N x N)
        """

        N=N
        φ=self.interferometer.chip.φ.to(u.m).value
        σ=self.interferometer.chip.σ.to(u.m).value
        p=self.p.value
        λ=self.interferometer.λ.to(u.m).value
        λ0=self.interferometer.chip.λ0.to(u.m).value
        fov=self.interferometer.fov
        output_order=self.interferometer.chip.output_order
        nb_raw_outputs = self.interferometer.chip.nb_raw_outputs
        nb_processed_outputs = self.interferometer.chip.nb_processed_outputs

        return get_transmission_map_jit(N=N, φ=φ, σ=σ, p=p, λ=λ, λ0=λ0, fov=fov, output_order=output_order, nb_raw_outputs=nb_raw_outputs, nb_processed_outputs=nb_processed_outputs)

    # Get transmission map gradiant norm --------------------------------------

    def get_transmission_map_gradient_norm(self, N:int) -> np.ndarray[float]:
        """Get the gradient norm of the transmission maps.

        Args:
            N (int): Map resolution.

        Returns:
            - np.ndarray[float]: Raw outputs transmission map gradient norms
                (nb_raw_outputs x N x N)
            - Optional[np.ndarray[float]]: Processed outputs transmission map
                gradient norms (nb_processed_outputs x N x N)
        """

        raw_out_maps, processed_out_maps = self.get_transmission_maps(N=N)

        raw_grad_maps = np.empty_like(raw_out_maps)
        processed_grad_maps = np.empty_like(processed_out_maps)

        for i in range(self.interferometer.chip.nb_raw_outputs):
            dnx, dny = np.gradient(raw_out_maps[i])
            raw_grad_maps[i] = np.sqrt(dnx**2 + dny**2)

        for i in range(self.interferometer.chip.nb_processed_outputs):
            ddx, ddy = np.gradient(processed_out_maps[i])
            processed_grad_maps[i] = np.sqrt(ddx**2 + ddy**2)

        return raw_grad_maps, processed_grad_maps

    # Plot transmission maps --------------------------------------------------

    def plot_transmission_maps(self, N:int, return_plot:bool = False, grad=False) -> None:
        
        # Get transmission maps
        if grad:
            raw_out_maps, processed_out_maps = self.get_transmission_map_gradient_norm(N=N)
        else:
            raw_out_maps, processed_out_maps = self.get_transmission_maps(N=N)

        # Get companions position to plot them
        companions_pos = []
        for c in self.target.companions:
            x, y = coordinates.ρθ_to_xy(ρ=c.ρ, θ=c.θ, fov=self.interferometer.fov)
            companions_pos.append((x*self.interferometer.fov/2, y*self.interferometer.fov/2))

        nb_raw_outs = self.interferometer.chip.nb_raw_outputs
        nb_processed_outs = self.interferometer.chip.nb_processed_outputs
        nb_columns = max(nb_raw_outs, nb_processed_outs)
        _, axs = plt.subplots(2, nb_columns, figsize=(5*nb_columns, 10))

        fov = self.interferometer.fov
        extent = (-fov.value/2, fov.value/2, -fov.value/2, fov.value/2)

        for i in range(nb_columns):

            if i >= nb_raw_outs:
                axs[0, i].axis('off')
                continue
            else:
                im = axs[0, i].imshow(raw_out_maps[i], aspect="equal", cmap="hot" if not grad else "gray", extent=extent)
                axs[0, i].set_title(self.interferometer.chip._raw_output_labels[i])
                plt.colorbar(im, ax=axs[0, i])

                axs[0, i].set_xlabel(r"$\theta_x$" + f" ({fov.unit})")
                axs[0, i].set_ylabel(r"$\theta_y$" + f" ({fov.unit})")
                axs[0, i].scatter(0, 0, color="yellow", marker="*", edgecolors="black", s=100)
                for x, y in companions_pos:
                    axs[0, i].scatter(x, y, color="blue", edgecolors="black")

            if i >= nb_processed_outs:
                axs[1, i].axis('off')
            else:
                im = axs[1, i].imshow(processed_out_maps[i], aspect="equal", cmap="bwr" if not grad else "gray", extent=extent)
                axs[1, i].set_title(self.interferometer.chip._processed_output_labels[i])
                axs[1, i].set_aspect("equal")
                plt.colorbar(im, ax=axs[1, i])

                axs[1, i].set_xlabel(r"$\theta_x$" + f" ({fov.unit})")
                axs[1, i].set_ylabel(r"$\theta_y$" + f" ({fov.unit})")
                axs[1, i].scatter(0, 0, color="yellow", marker="*", edgecolors="black", s=100)
                for x, y in companions_pos:
                    axs[1, i].scatter(x, y, color="blue", edgecolors="black")

        # Display companions positions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        transmissions = ""
        companions = [Companion(name=self.target.name + " Star", c=1, θ=0*u.deg, ρ=0*u.mas)] + self.target.companions
        for i, c in enumerate(companions):
            θ = c.θ.to(u.rad)
            ρ = c.ρ.to(u.rad)
            p = self.p.to(u.m)
            λ = self.interferometer.λ.to(u.m)
            ψ = get_unique_source_input_fields_jit(a=1, ρ=ρ.value, θ=θ.value, λ=λ.value, p=p.value)

            out_fields = self.interferometer.chip.get_output_fields(ψ=ψ, λ=self.interferometer.λ)
            raw_outs = np.abs(out_fields)**2
            processed_outs = self.interferometer.chip.process_outputs(raw_outs)

            linebreak = '<br>' if return_plot else '\n   '
            transmissions += '<h2>' if return_plot else ''
            transmissions += f"\n{c.name} throughputs:"
            transmissions += '</h1>' if return_plot else '\n----------' + linebreak
            transmissions += ",   ".join([f"{self.interferometer.chip._raw_output_labels[o]}: {raw_outs[o]*100:.2f}%" for o in range(nb_raw_outs)]) + linebreak
            transmissions += ",   ".join([f"{self.interferometer.chip._processed_output_labels[o]}: {processed_outs[o]*100:.2f}%" for o in range(nb_processed_outs)])

        if return_plot:
            plot = BytesIO()
            plt.savefig(plot, format='png')
            plt.close()
            return plot.getvalue(), transmissions
        plt.show()
        print(transmissions)

    # Input fields ------------------------------------------------------------

    def get_input_fields(self) -> np.ndarray[complex]:
        """Get complex amplitudes of the signals acquired by the telescopes.

        Returns:
            np.ndarray[complex]: Array of shape (n_companions + 1, n_telescopes).
        """
    
        input_fields = []
        λ = self.interferometer.λ.to(u.m).value
        p = self.p.to(u.m).value # Projected telescope positions
        pf = self.pf.to(1/u.s).value # Photon flux from the star for each telescope
        
        # Star input fields
        input_fields.append(get_unique_source_input_fields_jit(a=pf, ρ=0, θ=0, λ=λ, p=p))

        # Companion input fields
        for c in self.target.companions:
            pf_c = pf * c.c # Photon flux from the companion for each telescope
            input_fields.append(get_unique_source_input_fields_jit(a=pf_c, ρ=c.ρ.to(u.rad).value, θ=c.θ.to(u.rad).value, λ=λ, p=p))
        
        # Error OPD
        γ = np.random.normal(0, self.Γ.to(u.m).value, size=len(self.interferometer.telescopes))

        # OPD to phase difference
        phase = 2 * np.pi * γ / λ

        # Add the OPD error to the input fields
        for i in range(len(input_fields)):
            input_fields[i] = input_fields[i] * np.exp(1j * phase)

        return np.array(input_fields, dtype=np.complex128)
    
    # H range -----------------------------------------------------------------

    def get_h_range(self) -> np.ndarray[float]:
        """Get the hour-angle range of the observation.

        Returns:
            np.ndarray[float]: Hour angle values (radians).
        """
        
        nb_obs_per_night = int(self.Δh.to(u.hourangle).value // self.interferometer.camera.e.to(u.hour).value)

        if nb_obs_per_night < 1:
            nb_obs_per_night = 1
        
        h_range = np.linspace(self.h - self.Δh/2, self.h + self.Δh/2, nb_obs_per_night)
        return h_range

    # Observation -------------------------------------------------------------

    def observe_monochromatic(self):
        """Observe the target with monochromatic approximation.

        Returns:
            np.ndarray[float]: Output intensities (photon events).
        """

        nb_outs = self.interferometer.chip.nb_raw_outputs
        nb_objects = len(self.target.companions) + 1

        # Get output fields for all companions & star
        out_fields = np.empty((nb_objects, nb_outs), dtype=np.complex128)
        for companion, ψc in enumerate(self.get_input_fields()):
            out_fields[companion] = self.interferometer.chip.get_output_fields(ψ=ψc, λ=self.interferometer.λ)

            # Scale down companions according to their contrast with target star
            if companion > 0:
                out_fields[companion] *= np.sqrt(self.target.companions[companion - 1].c)

        # Scale up all fields according to the target flux
        out_fields *= np.sqrt(self.target._f)

        # Scale up all fields by the bandwidth
        out_fields *= np.sqrt(self.interferometer._Δλ)

        # Acquire intensity for each output
        outs = np.empty(nb_outs)
        for o in range(nb_outs):
            outs[o] = self.interferometer.camera.acquire(out_fields[:, o])

        return outs
    
    def observe(self, spectral_samples=5):
        """Observe the target in this context.

        Args:
            spectral_samples (int): Number of spectral samples to acquire (default: 5).

        Returns:
            np.ndarray[float]: Output intensities (photon events).
        """

        # If this context use monochromatic approximation
        if self.monochromatic:
            return self.observe_monochromatic()

        # Sampling bandwidth
        λ_range = np.linspace(self.interferometer.λ - self.interferometer.Δλ/2, self.interferometer.λ + self.interferometer.Δλ/2, spectral_samples)

        # Initialize output array
        nb_outs = self.interferometer.chip.nb_raw_outputs
        outs = np.empty((spectral_samples, nb_outs))

        # Monochromatic approximation for each sub-band
        for i, λ in enumerate(λ_range):
            ctx_mono = copy(self)
            ctx_mono.interferometer.λ = λ
            ctx_mono.interferometer.Δλ = 1 * u.nm

            _, outs[i] = ctx_mono.observe_monochromatic()

        # Integrate over the bandwidth
        return np.trapz(outs, λ_range.value, axis=0)

    def observation_serie(
            self,
            n:int = 1,
        ) -> np.ndarray[int]:
        """Generate a series of observations in this context.

        Args:
            n (int): Number of nights (observations per given hour angle).

        Returns:
            np.ndarray[int]: Array of shape (n, nb_hour_angles, nb_outputs)
        """

        # Get hour angle range
        h_range = self.get_h_range()

        # Initialize output array
        nb_outs = self.interferometer.chip.nb_raw_outputs
        outs = np.empty((n, len(h_range), nb_outs))

        # Observe for each hour angle
        for h_i, h in enumerate(h_range):
            ctx = copy(self)
            ctx.h = h

            # Observe n nights at this hour angle
            for n_i in range(n):
                outs[n_i, h_i,:] = ctx.observe()

        return outs
    
    # Genetic calibration -----------------------------------------------------

    def calibrate_gen(
            self,
            β: float,
            verbose: bool = False,
            plot:bool = False,
            figsize:tuple = (10, 10),
        ) -> dict:
        """Optimize phase shifter offsets to maximize nulling performance.

        Args:
            β (float): Decay factor for the step size (0.5 <= β < 1).
            verbose (bool): If ``True``, print optimization progress.
            plot (bool): If ``True``, plot the optimization process.
            figsize (tuple): Figure size for plots.

        Returns:
            dict: Dictionary with optimization history (depth, shifters).
        """

        self.Δh = self.interferometer.camera.e.to(u.hour).value * u.hourangle

        ψ = np.sqrt(self.pf.to(1/self.interferometer.camera.e.unit).value) * (1 + 0j) # Perfectly cophased inputs
        total_execpted_photons = np.sum(np.abs(ψ)**2)

        ε = 1e-6 * self.interferometer.λ.unit # Minimum shift step size

        # Shifters that contribute to redirecting light to the bright output
        φb = [1, 2, 3, 4, 5, 7]

        # Shifters that contribute to the symmetry of the dark outputs
        φk = [6, 8, 9, 10, 11, 12, 13, 14]

        # History of the optimization
        depth_history = []
        shifters_history = []

        Δφ = self.interferometer.λ / 4
        while Δφ > ε:

            if verbose:
                print(f"--- New iteration --- Δφ={Δφ:.2e}")

            for i in φb + φk:
                log = ""

                # Getting observation with different phase shifts
                self.interferometer.chip.φ[i-1] += Δφ
                _, k_pos, b_pos = self.observe()

                self.interferometer.chip.φ[i-1] -= 2*Δφ
                _, k_neg, b_neg = self.observe()

                self.interferometer.chip.φ[i-1] += Δφ
                _, k_old, b_old = self.observe()

                # Computing throughputs
                b_pos = b_pos / total_execpted_photons
                b_neg = b_neg / total_execpted_photons
                b_old = b_old / total_execpted_photons
                k_pos = np.sum(np.abs(k_pos)) / total_execpted_photons
                k_neg = np.sum(np.abs(k_neg)) / total_execpted_photons
                k_old = np.sum(np.abs(k_old)) / total_execpted_photons

                # Save the history
                depth_history.append(np.sum(k_old) / np.sum(b_old))
                shifters_history.append(np.copy(self.interferometer.chip.φ.value))

                # Maximize the bright metric for group 1 shifters
                if i in φb:
                    log += f"Shift {i} Bright: {b_neg:.2e} | {b_old:.2e} | {b_pos:.2e} -> "

                    if b_pos > b_old and b_pos > b_neg:
                        log += " + "
                        self.interferometer.chip.φ[i-1] += Δφ
                    elif b_neg > b_old and b_neg > b_pos:
                        log += " - "
                        self.interferometer.chip.φ[i-1] -= Δφ
                    else:
                        log += " = "

                # Minimize the kernel metric for group 2 shifters
                else:
                    log += f"Shift {i} Kernel: {k_neg:.2e} | {k_old:.2e} | {k_pos:.2e} -> "

                    if k_pos < k_old and k_pos < k_neg:
                        self.interferometer.chip.φ[i-1] += Δφ
                        log += " + "
                    elif k_neg < k_old and k_neg < k_pos:
                        self.interferometer.chip.φ[i-1] -= Δφ
                        log += " - "
                    else:
                        log += " = "
                
                if verbose:
                    print(log)

            Δφ *= β

        self.interferometer.chip.φ = phase.bound(self.interferometer.chip.φ, self.interferometer.λ)

        if plot:

            shifters_history = np.array(shifters_history)

            _, axs = plt.subplots(2,1, figsize=figsize, constrained_layout=True)

            axs[0].plot(depth_history)
            axs[0].set_xlabel("Iterations")
            axs[0].set_ylabel("Kernel-Null depth")
            axs[0].set_yscale("log")
            axs[0].set_title("Performance of the Kernel-Nuller")

            for i in range(shifters_history.shape[1]):
                axs[1].plot(shifters_history[:,i], label=f"Shifter {i+1}")
            axs[1].set_xlabel("Iterations")
            axs[1].set_ylabel("Phase shift")
            axs[1].set_yscale("linear")
            axs[1].set_title("Convergence of the phase shifters")
            # axs[1].legend(loc='upper right')

            plt.show()

        return {
            "depth": np.array(depth_history),
            "shifters": np.array(shifters_history),
        }
    
    # Obstruction calibration -------------------------------------------------

    def calibrate_obs(
            self,
            n: int = 1_000,
            plot: bool = False,
            figsize:tuple[int] = (30,20),
        ):
        """Optimize calibration via least squares sampling.

        Args:
            n (int): Number of sampling points for least squares.
            plot (bool): If ``True``, plot the optimization process.
            figsize (tuple[int]): Figure size for plots.

        Returns:
            None | Context: New context with optimized kernel nuller (if implemented to return).
        """


        chip = self.interferometer.chip
        input_attenuation_backup = chip.input_attenuation.copy()
        λ = self.interferometer.λ
        e = self.interferometer.camera.e
        total_photons = np.sum(self.pf.to(1/e.unit).value) * e.value

        if plot:
            _, axs = plt.subplots(6, 3, figsize=figsize, constrained_layout=True)
            for i in range(7):
                axs.flatten()[i].set_xlabel("Phase shift")
                axs.flatten()[i].set_ylabel("Throughput")

        def maximize_bright(p, plt_coords=None):

            x = np.linspace(0, λ.value,n)
            y = np.empty(n)

            if isinstance(p,list):
                Δp = chip.φ[p[1]-1] - chip.φ[p[0]-1]

            for i in range(n):

                if isinstance(p,list):
                    chip.φ[p[0]-1] = i * λ / n
                    chip.φ[p[1]-1] = (chip.φ[p[0]-1] + Δp) % λ
                else:
                    chip.φ[p-1] = i * λ / n
            
                _, _, b = self.observe()
                y[i] = b / total_photons
            
            def sin(x, x0):
                return (np.sin((x-x0)/λ.value*2*np.pi)+1)/2 * (np.max(y)-np.min(y)) + np.min(y)
            
            popt, _ = curve_fit(sin, x, y, p0=[0], maxfev = 100_000)

            if isinstance(p,list):
                chip.φ[p[0]-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(chip.φ.unit)
                chip.φ[p[1]-1] = (chip.φ[p[0]-1] + Δp) % λ
            else:
                chip.φ[p-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(chip.φ.unit)

            if plot:
                axs[plt_coords].set_title(f"$|B(\phi{p})|$")
                axs[plt_coords].scatter(x, y, label='Data', color='tab:blue')
                axs[plt_coords].plot(x, sin(x, *popt), label='Fit', color='tab:orange')
                axs[plt_coords].axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
                axs[plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
                axs[plt_coords].set_ylabel("Bright throughput")
                axs[plt_coords].legend()

        def minimize_kernel(p, m, plt_coords=None):

            x = np.linspace(0,λ.value,n)
            y = np.empty(n)

            for i in range(n):
                chip.φ[p-1] = i * λ / n
                _, k, b = self.observe()
                y[i] = k[m-1] / b
            
            def sin(x, x0):
                return (np.sin((x-x0)/λ.value*2*np.pi)+1)/2 * (np.max(y)-np.min(y)) + np.min(y)
            
            popt, _ = curve_fit(sin, x, y, p0=[0], maxfev = 100_000)

            chip.φ[p-1] = (np.mod(popt[0], λ.value) * λ.unit).to(chip.φ.unit)

            if plot:
                axs[plt_coords].set_title(f"$K_{m}(\phi{p})$")
                axs[plt_coords].scatter(x, y, label='Data', color='tab:blue')
                axs[plt_coords].plot(x, sin(x, *popt), label='Fit', color='tab:orange')
                axs[plt_coords].axvline(x=np.mod(popt[0], λ.value), color='k', linestyle='--', label='Optimal phase shift')
                axs[plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
                axs[plt_coords].set_ylabel("Kernel throughput")
                axs[plt_coords].legend()

        def maximize_darks(p, ds, plt_coords=None):

            x = np.linspace(0, λ.value, n)
            y = np.empty(n)

            for i in range(n):
                chip.φ[p-1] = i * λ / n
                d, _, b = self.observe()
                y[i] = np.sum(np.abs(d[np.array(ds)-1])) / b

            def sin(x, x0):
                return (np.sin((x-x0)/λ.value*2*np.pi)+1)/2 * (np.max(y)-np.min(y)) + np.min(y)
            
            popt, _ = curve_fit(sin, x, y, p0=[0], maxfev = 100_000)

            chip.φ[p-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(chip.φ.unit)

            if plot:
                axs[plt_coords].set_title(f"$|D_{ds[0]}(\phi{p})| + |D_{ds[1]}(\phi{p})|$")
                axs[plt_coords].scatter(x, y, label='Data', color='tab:blue')
                axs[plt_coords].plot(x, sin(x, *popt), label='Fit', color='tab:orange')
                axs[plt_coords].axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
                axs[plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
                axs[plt_coords].set_ylabel(f"Dark pair {ds} throughput")
                axs[plt_coords].legend()

        # Bright maximization
        self.interferometer.chip.input_attenuation = [1, 1, 0, 0]
        maximize_bright(2, plt_coords=(0,0))
        maximize_bright([1,2], plt_coords=(1,0))

        if plot:
            axs[1,1].axis('off')
            axs[1,2].axis('off')
            plt.show()

        self.interferometer.chip.input_attenuation = [0, 0, 1, 1]
        maximize_bright(4, plt_coords=(0,1))
        maximize_bright([3,4], plt_coords=(1,1))

        self.interferometer.chip.input_attenuation = [1, 0, 1, 0]
        maximize_bright(7, plt_coords=(0,2))
        maximize_bright([5,7], plt_coords=(1,2))

        # Darks maximization
        self.interferometer.chip.input_attenuation = [1, 0, 0, -1]
        maximize_darks(8, [1,2], plt_coords=(1,0))

        # Kernel minimization
        self.interferometer.chip.input_attenuation = [1, 0, 0, 0]
        minimize_kernel(11, 1, plt_coords=(2,0))
        minimize_kernel(13, 2, plt_coords=(2,1))
        minimize_kernel(14, 3, plt_coords=(2,2))

        chip.φ = phase.bound(chip.φ, λ)
        chip.input_attenuation = input_attenuation_backup

        if plot:
            axs[1,1].axis('off')
            axs[1,2].axis('off')
            plt.show()

    #==============================================================================
    # VLTI Context
    #==============================================================================

    def get_VLTI() -> 'Context':
        """Get a default VLTI context for analysis.

        Uses:
            - VLTI with 4 UTs
            - First generation active kernel nuller
            - Vega as target star and a hypothetical 2 mas, 1e-6 contrast companion
        """

        λ = 1.55 * u.um # Central wavelength

        ctx = Context(
            h = 0 * u.hourangle, # Central hour angle
            Δh = 8 * u.hourangle, # Hour angle range
            Γ = 100 * u.nm, # Input cophasing error (RMS)
            monochromatic=False,
            name="Default Context", # Context name
            interferometer = Interferometer(
                l = -24.6275 * u.deg, # Latitude
                λ = λ, # Central wavelength
                Δλ = 1 * u.nm, # Bandwidth
                fov = 10 * u.mas, # Field of view
                η = 0.02, # Optical efficiency
                telescopes = telescope.get_VLTI_UTs(),
                name = "VLTI", # Interferometer name
                chip = SuperKN(
                    φ = np.zeros(14) * u.um, # Injected phase shifts
                    σ = np.abs(np.random.normal(0, 1, 14)) * u.um, # Manufacturing OPD errors
                    λ0 = λ,
                    name = "First Generation Kernel-Nuller", # Kernel nuller name
                ),
                camera = Camera(
                    e = 5 * u.min, # Exposure time
                    name = "Default Camera", # Camera name
                ),
            ),
            target=Target(
                f = (1050 * u.Jy * 2 * np.pi * const.c / λ**2).to(u.W / u.m**2 / u.nm), # Target flux
                δ = -64.71 * u.deg, # Target declination
                name = "Vega", # Target name
                companions = [
                    Companion(
                        c = 1e-6, # Companion contrast
                        ρ = 4 * u.mas, # Companion angular separation
                        θ = 0 * u.deg, # Companion position angle
                        name = "Hypothetical Companion", # Companion name
                    ),
                ],
            ),
        )

        return ctx

    #==============================================================================
    # LIFE Context
    #==============================================================================

    def get_LIFE() -> 'Context':
        """Get a default LIFE context for analysis.

        Uses:
            - 4 telescopes of LIFE
            - First generation active kernel nuller
            - Vega as target star and a hypothetical 2 mas, 1e-6 contrast companion
        """

        λ = 4 * u.um # Central wavelength

        ctx = Context(
            interferometer = Interferometer(
                l = -90 * u.deg, # Latitude
                λ = λ, # Central wavelength
                Δλ = 1 * u.nm, # Bandwidth
                fov = 10 * u.mas, # Field of view
                η = 0.02, # Optical efficiency
                telescopes = telescope.get_VLTI_UTs(),
                name = "LIFE", # Interferometer name
                chip = SuperKN(
                    φ = np.zeros(14) * u.um, # Injected phase shifts
                    σ = np.abs(np.random.normal(0, 1, 14)) * u.um, # Manufacturing OPD errors
                    λ0 = λ,
                    name = "First Generation Kernel-Nuller", # Kernel nuller name
                ),
                camera = Camera(
                    e = 5 * u.min, # Exposure time
                    name = "Default Camera", # Camera name
                ),
            ),
            target=Target(
                f = (1050 * u.Jy * 2 * np.pi * const.c / λ**2).to(u.W / u.m**2 / u.nm), # Target flux
                δ = -90 * u.deg, # Target declination
                name = "Vega", # Target name
                companions = [
                    Companion(
                        c = 1e-6, # Companion contrast
                        ρ = 4 * u.mas, # Companion angular separation
                        θ = 0 * u.deg, # Companion position angle
                        name = "Hypothetical Companion", # Companion name
                    ),
                ],
            ),
            h = 0 * u.hourangle, # Central hour angle
            Δh = 24 * u.hourangle, # Hour angle range
            Γ = 1 * u.nm, # Input cophasing error (RMS)
            name="Default Context", # Context name
        )

        return ctx
            
#==============================================================================
# Number functions
#==============================================================================

# Projected position ----------------------------------------------------------

@nb.njit()
def project_position_jit(
        r: np.ndarray[float],
        h: float,
        l: float,
        δ: float,
    ) -> np.ndarray[float]:
    """
    Project the telescope position in a plane perpendicular to the line of sight.

    Parameters
    ----------
    - r: Array of telescope positions (in meters)
    - h: Hour angle (in radian)
    - l: Latitude (in radian)
    - δ: Declination (in radian)

    Returns
    -------
    - Array of projected telescope positions (same shape and unit as p)
    """

    M = np.array([
        [ -np.sin(l)*np.sin(h),                                np.cos(h)          ],
        [ np.sin(l)*np.cos(h)*np.sin(δ) + np.cos(l)*np.cos(δ), np.sin(h)*np.sin(δ)],
    ])

    p = np.empty_like(r)
    for i, (x,y) in enumerate(r):
        p[i] = M @ np.array([y, x])

    return p

# Transmission maps -----------------------------------------------------------

@nb.njit()
def get_transmission_map_jit(
        N: int,
        φ: np.ndarray[float],
        σ: np.ndarray[float],
        p: np.ndarray[float],
        λ: float,
        λ0: float,
        fov: float,
        output_order: np.ndarray[int],
        nb_raw_outputs: int,
        nb_processed_outputs: int,
    ) -> tuple[np.ndarray[complex], np.ndarray[complex], np.ndarray[float]]:
    """
    Generate the transmission maps of this context with a given resolution

    Parameters
    ----------
    - N: Resolution of the map
    - φ: Array of 14 injected OPD (in meter)
    - σ: Array of 14 intrasic OPD (in meter)
    - p: Projected telescope positions (in meter)
    - λ: Wavelength (in meter)
    - λ0: Reference wavelength (in meter)
    - fov: Field of view in mas
    - output_order: Order of the outputs
    - processed_outputs : If ``True``, also return the processed outputs transmission maps.

    Returns
    -------
    - Raw outputs transmission maps (nb_raw_outputs x resolution x resolution)
    - Processed outputs transmission maps (nb_processed_outputs x resolution x resolution)
    """

    # Get the coordinates of the map
    _, _, θ_map, ρ_map = coordinates.get_maps_jit(N=N, fov=fov)

    # mas to radian
    ρ_map = ρ_map / 1000 / 3600 / 180 * np.pi

    raw_out_maps = np.empty((nb_raw_outputs, N, N))
    processed_out_maps = np.empty((nb_processed_outputs, N, N))

    for x in range(N):
        for y in range(N):
            
            θ = θ_map[x, y]
            ρ = ρ_map[x, y]

            ψ = get_unique_source_input_fields_jit(a=1, ρ=ρ, θ=θ, λ=λ, p=p)
            raw_outs = np.abs(superkn.get_output_fields_jit(ψ, φ, σ, λ, λ0, output_order))**2

            for i in range(nb_raw_outputs):
                raw_out_maps[i, x, y] = raw_outs[i]

            if nb_processed_outputs > 0:
                processed_outs = superkn.process_outputs_jit(raw_outs)

                for i in range(nb_processed_outputs):
                    processed_out_maps[i, x, y] = processed_outs[i]

    return raw_out_maps, processed_out_maps

# Input fields ----------------------------------------------------------------

@nb.njit()
def get_unique_source_input_fields_jit(
    a: float,
    ρ: float,
    θ: float,
    λ: float,
    p: np.ndarray[float],
) -> np.ndarray[complex]:
    """
    Get the complexe amplitude of the input signals according to the object and telescopes positions.

    Parameters
    ----------
    - a: Intensity of the signal (prop. to #photons/s)
    - ρ: Angular separation (in radian)
    - θ: Parallactic angle (in radian)
    - λ: Wavelength (in meter)
    - p: Projected telescope positions (in meter)

    Returns
    -------
    - Array of acquired signals (complex amplitudes).
    """

    # Array of complex signals
    s = np.empty(p.shape[0], dtype=np.complex128)

    for i, t in enumerate(p):

        # Rotate the projected telescope positions by the parallactic angle
        p_rot = t[0] * np.cos(-θ) - t[1] * np.sin(-θ)

        # Compute the phase delay according to the object position
        Φ = 2 * np.pi * p_rot * np.sin(ρ) / λ

        # Build the complex amplitude of the signal
        s[i] = np.exp(1j * Φ)

    return s * np.sqrt(a / p.shape[0])
