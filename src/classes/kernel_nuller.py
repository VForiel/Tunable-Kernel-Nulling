import numpy as np
import numba as nb
import astropy.units as u
from astropy import constants as const
import matplotlib.pyplot as plt
from io import BytesIO
import os
import ipywidgets as widgets
from copy import deepcopy as copy

from ..modules import mmi
from ..modules import phase
from .source import Source

class KernelNuller():
    def __init__(
            self,
            φ: np.ndarray[u.Quantity],
            σ: np.ndarray[u.Quantity],
            output_order: np.ndarray[int] = None,
            name: str = "Unnamed",
        ):
        """Kernel-Nuller object.

        Parameters
        ----------
        - φ: Array of 14 injected OPD
        - σ: Array of 14 intrasic OPD error
        - output_order: Order of the outputs
        - name: Name of the Kernel-Nuller object
        """
        self._φ = φ
        self.σ = σ
        self.output_order = output_order if output_order is not None else np.array([0, 1, 2, 3, 4, 5])
        self.name = name

    def copy(self,
            φ:np.ndarray[u.Quantity] = None,
            σ:np.ndarray[u.Quantity] = None,
            output_order:np.ndarray[int] = None,
            **kwargs
        ) -> "KernelNuller":
        """
        Create a copy of the Kernel-Nuller object with some parameters changed.

        Parameters
        ----------
        - φ: Array of 14 injected OPD
        - σ: Array of 14 intrasic OPD error
        - output_order: Order of the outputs

        Returns
        -------
        - Copied Kernel-Nuller object
        """
        return KernelNuller(
            φ = copy(φ) if φ is not None else copy(self.φ),
            σ = copy(σ) if σ is not None else copy(self.σ),
            output_order = copy(output_order) if output_order is not None else copy(self.output_order),
        )

    @property
    def φ(self):
        return self._φ
    
    @φ.setter
    def φ(self, φ:np.ndarray[u.Quantity]):
        if type(φ) != u.Quantity:
            raise ValueError("φ must be a Quantity")
        try:
            φ.to(u.m)
        except u.UnitConversionError:
            raise ValueError("φ must be in a distance unit")
        if φ.shape != (14,):
            raise ValueError("φ must have a shape of (14,)")
        self._φ = φ

    @property
    def σ(self):
        return self._σ
    
    @σ.setter
    def σ(self, σ:np.ndarray[u.Quantity]):
        if type(σ) != u.Quantity:
            raise ValueError("σ must be a Quantity")
        try:
            σ.to(u.m)
        except u.UnitConversionError:
            raise ValueError("σ must be in a distance unit")
        if σ.shape != (14,):
            raise ValueError("σ must have a shape of (14,)")
        self._σ = σ

    @property
    def output_order(self):
        return self._output_order
    
    @output_order.setter
    def output_order(self, output_order:np.ndarray[int]):
        try:
            output_order = np.array(output_order, dtype=int)
        except:
            raise ValueError(f"output_order must be an array of integers, not {type(output_order)}")
        if output_order.shape != (6,):
            raise ValueError(f"output_order must have a shape of (6,), not {output_order.shape}")
        if not np.all(np.sort(output_order) == np.arange(6)):
            raise ValueError(f"output_order must contain all the integers from 0 to 5, not {output_order}")
        if output_order[0] - output_order[1] not in [-1, 1] \
                or output_order[2] - output_order[3] not in [-1, 1] \
                or output_order[4] - output_order[5] not in [-1, 1]:
            raise ValueError(f"output_order contain an invalid configuration of output pairs. Found {output_order}")
        self._output_order = output_order

    # Electric fields propagation ---------------------------------------------

    def propagate_fields(
            self,
            ψ: np.ndarray[complex],
            λ: u.Quantity,
        ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], float]:
        """
        Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

        Parameters
        ----------
        - ψ: Array of 4 input signals complex amplitudes
        - λ: Wavelength of the light

        Returns
        -------
        - Array of 3 null outputs electric fields
        - Array of 6 dark outputs electric fields
        - Bright output electric fields
        """
        φ = self.φ.to(λ.unit).value
        σ = self.σ.to(λ.unit).value
        λ = λ.value

        return propagate_fields_njit(ψ=ψ, φ=φ, σ=σ, λ=λ, output_order=self.output_order)

    # Observation -------------------------------------------------------------

    def observe(
            self,
            ψ: np.ndarray[complex],
            λ: u.Quantity,
            f: float = None,
            Δt:u.Quantity = 1*u.s,
        ) -> np.ndarray[float]:
        """
        Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

        Parameters
        ----------
        - Ψ: Array of 4 input beams complex amplitudes
        - λ: Wavelength of the light
        - Δt: Exposure time 

        Returns
        -------
        - Array of 6 dark outputs intensities
        - Array of 3 kernels output intensities
        - Bright output intensity
        """
        φ = self.φ.to(λ.unit).value
        σ = self.σ.to(λ.unit).value
        Δt = Δt.to(u.s).value
        return observe_njit(ψ, φ, σ, λ.value, Δt, self.output_order)
    
    # Plotting --------------------------------------------------------------------

    def plot_phase(
            self,
            λ,
            ψ=np.array([0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j]),
            plot=True
        ):

        ψ1 = np.array([ψ[0], 0, 0, 0])
        ψ2 = np.array([0, ψ[1], 0, 0])
        ψ3 = np.array([0, 0, ψ[2], 0])
        ψ4 = np.array([0, 0, 0, ψ[3]])

        n1, d1, b1 = self.propagate_fields(ψ1, λ)
        n2, d2, b2 = self.propagate_fields(ψ2, λ)
        n3, d3, b3 = self.propagate_fields(ψ3, λ)
        n4, d4, b4 = self.propagate_fields(ψ4, λ)

        # Using first signal as reference
        n2 = np.abs(n2) * np.exp(1j * (np.angle(n2) - np.angle(n1)))
        n3 = np.abs(n3) * np.exp(1j * (np.angle(n3) - np.angle(n1)))
        n4 = np.abs(n4) * np.exp(1j * (np.angle(n4) - np.angle(n1)))
        d2 = np.abs(d2) * np.exp(1j * (np.angle(d2) - np.angle(d1)))
        d3 = np.abs(d3) * np.exp(1j * (np.angle(d3) - np.angle(d1)))
        d4 = np.abs(d4) * np.exp(1j * (np.angle(d4) - np.angle(d1)))
        b2 = np.abs(b2) * np.exp(1j * (np.angle(b2) - np.angle(b1)))
        b3 = np.abs(b3) * np.exp(1j * (np.angle(b3) - np.angle(b1)))
        b4 = np.abs(b4) * np.exp(1j * (np.angle(b4) - np.angle(b1)))
        n1 = np.abs(n1) * np.exp(1j * 0)
        d1 = np.abs(d1) * np.exp(1j * 0)
        b1 = np.abs(b1) * np.exp(1j * 0)

        _, axs = plt.subplots(2, 6, figsize=(20, 7.5), subplot_kw={'projection': 'polar'})

        # Bright output
        axs[0,0].scatter(np.angle(b1), np.abs(b1), color="yellow", label='Input 1', alpha=0.5)
        axs[0,0].plot([0, np.angle(b1)], [0, np.abs(b1)], color="yellow", alpha=0.5)
        axs[0,0].scatter(np.angle(b2), np.abs(b2), color="green", label='Input 2', alpha=0.5)
        axs[0,0].plot([0, np.angle(b2)], [0, np.abs(b2)], color="green", alpha=0.5)
        axs[0,0].scatter(np.angle(b3), np.abs(b3), color="red", label='Input 3', alpha=0.5)
        axs[0,0].plot([0, np.angle(b3)], [0, np.abs(b3)], color="red", alpha=0.5)
        axs[0,0].scatter(np.angle(b4), np.abs(b4), color="blue", label='Input 4', alpha=0.5)
        axs[0,0].plot([0, np.angle(b4)], [0, np.abs(b4)], color="blue", alpha=0.5)
        axs[0,0].set_title('Bright output')

        for n in range(3):
            axs[0,n+1].scatter(np.angle(n1[n]), np.abs(n1[n]), color="yellow", label='Input 1', alpha=0.5)
            axs[0,n+1].plot([0, np.angle(n1[n])], [0, np.abs(n1[n])], color="yellow", alpha=0.5)
            axs[0,n+1].scatter(np.angle(n2[n]), np.abs(n2[n]), color="green", label='Input 2', alpha=0.5)
            axs[0,n+1].plot([0, np.angle(n2[n])], [0, np.abs(n2[n])], color="green", alpha=0.5)
            axs[0,n+1].scatter(np.angle(n3[n]), np.abs(n3[n]), color="red", label='Input 3', alpha=0.5)
            axs[0,n+1].plot([0, np.angle(n3[n])], [0, np.abs(n3[n])], color="red", alpha=0.5)
            axs[0,n+1].scatter(np.angle(n4[n]), np.abs(n4[n]), color="blue", label='Input 4', alpha=0.5)
            axs[0,n+1].plot([0, np.angle(n4[n])], [0, np.abs(n4[n])], color="blue", alpha=0.5)
            axs[0,n+1].set_title(f'Null output {n+1}')

        for d in range(6):
            axs[1,d].scatter(np.angle(d1[d]), np.abs(d1[d]), color="yellow", label='I1', alpha=0.5)
            axs[1,d].plot([0, np.angle(d1[d])], [0, np.abs(d1[d])], color="yellow", alpha=0.5)
            axs[1,d].scatter(np.angle(d2[d]), np.abs(d2[d]), color="green", label='I2', alpha=0.5)
            axs[1,d].plot([0, np.angle(d2[d])], [0, np.abs(d2[d])], color="green", alpha=0.5)
            axs[1,d].scatter(np.angle(d3[d]), np.abs(d3[d]), color="red", label='I3', alpha=0.5)
            axs[1,d].plot([0, np.angle(d3[d])], [0, np.abs(d3[d])], color="red", alpha=0.5)
            axs[1,d].scatter(np.angle(d4[d]), np.abs(d4[d]), color="blue", label='I4', alpha=0.5)
            axs[1,d].plot([0, np.angle(d4[d])], [0, np.abs(d4[d])], color="blue", alpha=0.5)
            axs[1,d].set_title(f'Dark output {d+1}')

        m = np.max(np.concatenate([
            np.abs(n1), np.abs(n2), np.abs(n3), np.abs(n4),
            np.abs(d1), np.abs(d2), np.abs(d3), np.abs(d4),
            np.array([np.abs(b1), np.abs(b2), np.abs(b3), np.abs(b4)])
        ]))

        for ax in axs.flatten():
            ax.set_ylim(0, m)

        axs[0, 4].axis("off")
        axs[0, 5].axis("off")

        axs[0,0].legend()

        if not plot:
            plot = BytesIO()
            plt.savefig(plot, format='png')
            plt.close()
            return plot.getvalue()
        plt.show()

    # Shift control GUI -------------------------------------------------------

    def shifts_control_gui(self, λ:u.Quantity):
        step = 1e-20

        # Build sliders -----------------------------------------------------------

        # Input amplitude
        IA_sliders = [
            widgets.FloatSlider(
                value=0.5, min=0, max=0.5, step=step, description=f"I{i+1}",
                continuous_update=False,
            )
            for i in range(4)
        ]

        # Input phase
        IP_sliders = [
            widgets.FloatSlider(
                value=0, min=0, max=λ.value, step=step, description=f"I{i+1}",
                continuous_update=False,
            )
            for i in range(4)
        ]

        # Shifter phase
        P_sliders = [
            widgets.FloatSlider(
                value=0, min=0, max=λ.value, step=step, description=f"P{i+1}",
                continuous_update=False,
            )
            for i in range(14)
        ]

        # for i in range(14):
        #     P_sliders[i].value = CALIBRATED_SHIFTS_IB[i].value


        # Build GUI ---------------------------------------------------------------

        def beam_repr(beam: complex) -> str:
            return f"<b>{np.abs(beam):.2e}</b> * exp(<b>{np.angle(beam)/np.pi:.2f}</b> pi i)"

        inputs = [widgets.HTML(value=f" ") for _ in range(4)]
        null_outputs = [widgets.HTML(value=f" ") for _ in range(4)]
        dark_outputs = [widgets.HTML(value=f" ") for _ in range(6)]
        kernel_outputs = [widgets.HTML(value=f" ") for _ in range(3)]

        def update_gui(*args):

            ψ = np.array([
                IA_sliders[i].value * np.exp(1j * IP_sliders[i].value / λ.value * 2 * np.pi)
                for i in range(4)
            ])

            self.φ = np.array([x.value for x in P_sliders]) * λ.unit
            n, d, b = self.propagate_fields(ψ=ψ, λ=λ)

            k = np.array([
                np.abs(d[2*i])**2 - np.abs(d[2*i+1])**2
            for i in range(3)])

            for i, beam in enumerate(ψ):
                inputs[i].value = (
                    f"<b>Input {i+1} -</b> Amplitude: <code>{beam_repr(beam)}</code> Intensity: <code><b>{np.abs(beam)**2*100:.1f}%</b></code>"
                )
            null_outputs[0].value = (
                f"<b>N3a -</b> Amplitude: <code>{beam_repr(b)}</code> Intensity: <code><b>{np.abs(b)**2*100:.1f}%</b></code> <b><- Bright channel</b>"
            )
            for i, beam in enumerate(n):
                null_outputs[i + 1].value = (
                    f"<b>N{(i-1)//2+4}{['a','b'][(i+1)%2]} -</b> Amplitude: <code>{beam_repr(beam)}</code> Intensity: <code><b>{np.abs(beam)**2*100:.1f}%</b></code>"
                )
            for i, beam in enumerate(d):
                dark_outputs[i].value = (
                    f"<b>Dark {i+1} -</b> Amplitude: <code>{beam_repr(beam)}</code> Intensity: <code><b>{np.abs(beam)**2*100:.1f}%</b></code>"
                )
            for i, beam in enumerate(k):
                kernel_outputs[i].value = (
                    f"<b>Kernel {i+1} -</b> Value: <code>{beam:.2e}</code>"
                )   
                
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            phases.value = self.plot_phase(
                λ=λ,
                ψ = ψ,
                plot = False
                )

            # Plot intensities
            for i in range(len(ψ)):
                plt.imshow([[np.abs(ψ[i])**2,],], cmap="hot", vmin=0, vmax=np.sum(np.abs(ψ)**2))
                plt.savefig(fname=f"img/tmp.png", format="png")
                plt.close()
                with open("img/tmp.png", "rb") as file:
                    image = file.read()
                    photometric_cameras[i].value = image
            for i in range(len(n)+1):
                if i == 0:
                    plt.imshow([[np.abs(b)**2,],], cmap="hot", vmin=0, vmax=np.sum(np.abs(n)**2) + np.abs(b)**2)
                else:
                    plt.imshow([[np.abs(n[i-1])**2,],], cmap="hot", vmin=0, vmax=np.sum(np.abs(n)**2) + np.abs(b)**2)
                plt.savefig(fname=f"img/tmp.png", format="png")
                plt.close()
                with open("img/tmp.png", "rb") as file:
                    image = file.read()
                    null_cameras[i].value = image
            for i in range(len(d)):
                plt.imshow([[np.abs(d[i])**2,],], cmap="hot", vmin=0, vmax=np.sum(np.abs(d)**2))
                plt.savefig(fname=f"img/tmp.png", format="png")
                plt.close()
                with open("img/tmp.png", "rb") as file:
                    image = file.read()
                    dark_cameras[i].value = image
            for i in range(len(k)):
                plt.imshow([[k[i],],], cmap="bwr", vmin=-np.max(np.abs(k)), vmax=np.max(np.abs(k)))
                plt.savefig(fname=f"img/tmp.png", format="png")
                plt.close()
                with open("img/tmp.png", "rb") as file:
                    image = file.read()
                    kernel_cameras[i].value = image

            os.remove("img/tmp.png")

            return b, d
        
        photometric_cameras = [widgets.Image(width=50,height=50) for _ in range(4)]
        null_cameras = [widgets.Image(width=50,height=50) for _ in range(4)]
        dark_cameras = [widgets.Image(width=50,height=50) for _ in range(6)]
        kernel_cameras = [widgets.Image(width=50,height=50) for _ in range(3)]
        phases = widgets.Image()

        vbox = widgets.VBox(
            [
                widgets.HTML("<h1>Inputs</h1>"),
                widgets.HTML("Amplitude:"),
                widgets.HBox(IA_sliders[:4]),
                widgets.HTML("Phase:"),
                widgets.HBox(IP_sliders[:4]),
                *[widgets.HBox([photometric_cameras[i], x]) for i, x in enumerate(inputs)],
                widgets.HTML("<h1>Phases</h1>"),
                phases,
                widgets.HTML("<h1>Nuller</h1>"),
                widgets.HBox(P_sliders[:4]),
                widgets.HBox(P_sliders[4:8]),
                *[widgets.HBox([null_cameras[i], x]) for i, x in enumerate(null_outputs)],
                widgets.HTML("<h1>Recombiner</h1>"),
                widgets.HBox(P_sliders[8:11]),
                widgets.HBox(P_sliders[11:14]),
                *[widgets.HBox([dark_cameras[i], x]) for i, x in enumerate(dark_outputs)],
                widgets.HTML("<h1>Kernels</h1>"),
                *[widgets.HBox([kernel_cameras[i], x]) for i, x in enumerate(kernel_outputs)],
            ]
        )

        # Link sliders to update function ------------------------------------------

        for widget in P_sliders:
            widget.observe(update_gui, "value")
        for widget in IA_sliders:
            widget.observe(update_gui, "value")
        for widget in IP_sliders:
            widget.observe(update_gui, "value")

        update_gui()
        return vbox
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Kernel-Nuller "{self.name}":' + '\n' \
            f" | output_order = {self.output_order}" + '\n' + \
            f" | φ = {self.φ}" + '\n' + \
            f" | σ = {self.σ}"
        

#==============================================================================
# Numba functions
#==============================================================================

# Electric fields propagation -------------------------------------------------

@nb.njit()
def propagate_fields_njit(
        ψ: np.ndarray[complex],
        φ: np.ndarray[float],
        σ: np.ndarray[float],
        λ: float,
        output_order:np.ndarray[int]
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], float]:
    """
    Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

    Parameters
    ----------
    - ψ: Array of 4 input signals complex amplitudes
    - φ: Array of 14 injected OPD (in wavelenght unit)
    - σ: Array of 14 intrasic OPD error (in wavelenght unit)
    - λ: Wavelength of the light
    - output_order: Order of the outputs

    Returns
    -------
    - Array of 3 null outputs electric fields
    - Array of 6 dark outputs electric fields
    - Bright output electric fields
    """

    φ = phase.bound_njit(φ + σ, λ)

    # First layer of pahse shifters
    nuller_inputs = phase.shift_njit(ψ, φ[:4], λ)

    # First layer of nulling
    N1 = mmi.nuller_2x2(nuller_inputs[:2])
    N2 = mmi.nuller_2x2(nuller_inputs[2:])

    # Second layer of phase shifters
    N1_shifted = phase.shift_njit(N1, φ[4:6], λ)
    N2_shifted = phase.shift_njit(N2, φ[6:8], λ)

    # Second layer of nulling
    N3 = mmi.nuller_2x2(np.array([N1_shifted[0], N2_shifted[0]]))
    N4 = mmi.nuller_2x2(np.array([N1_shifted[1], N2_shifted[1]]))

    nulls = np.array([N3[1], N4[0], N4[1]], dtype=np.complex128)
    bright = N3[0]

    # Beam splitting
    R_inputs = np.array([N3[1], N3[1], N4[0], N4[0], N4[1], N4[1]]) * 1 / np.sqrt(2)

    # Last layer of phase shifters
    R_inputs = phase.shift_njit(R_inputs, φ[8:], λ)

    # Beam mixing
    R1_output = mmi.cross_recombiner_2x2(np.array([R_inputs[0], R_inputs[2]]))
    R2_output = mmi.cross_recombiner_2x2(np.array([R_inputs[1], R_inputs[4]]))
    R3_output = mmi.cross_recombiner_2x2(np.array([R_inputs[3], R_inputs[5]]))

    darks = np.array(
        [
            R1_output[0],
            R1_output[1],
            R2_output[0],
            R2_output[1],
            R3_output[0],
            R3_output[1],
        ],
        dtype=np.complex128,
    )

    darks = darks[output_order]

    return nulls, darks, bright

# Observation -----------------------------------------------------------------

@nb.njit()
def observe_njit(
    ψ: np.ndarray[complex],
    φ: u.Quantity,
    σ: u.Quantity,
    λ: u.Quantity,
    Δt:float,
    output_order:np.ndarray[int],
) -> np.ndarray[float]:
    """
    Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

    Parameters
    ----------
    - ψ: Array of 4 input beams complex amplitudes (in photon/[Δt] unit)
    - φ: Array of 14 injected OPD
    - σ: Array of 14 intrasic OPD
    - λ: Wavelength of the light
    - Δt: Exposure time in seconds
    - output_order: Order of the outputs

    Returns
    -------
    - Array of 6 dark outputs intensities
    - Array of 3 kernels outputs intensities
    - Bright output intensity
    """

    _, d, b = propagate_fields_njit(ψ, φ, σ, λ, output_order)

    # Get intensities
    d = np.abs(d) ** 2
    b = np.abs(b) ** 2

    # Add photon noise
    dp = d * (d <= 2147020237)
    dn = d * (d > 2147020237)

    for i in range(d.shape[0]):
        d[i] = int(np.random.poisson(dp[i] * Δt))
        d[i] += int(np.random.normal(dn[i], np.sqrt(dn[i])))

    if b <= 2147020237:
        b = np.random.poisson(b * Δt)
    else:
        b = int(np.random.normal(b, np.sqrt(b)))

    # Create kernels
    k = np.array([d[0]-d[1], d[2]-d[3], d[4]-d[5]])

    return d, k, b