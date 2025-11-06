import numpy as np
import numba as nb
import astropy.units as u
from typing import Tuple, Any, Optional
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from io import BytesIO
from copy import deepcopy as copy
from ...modules import mmi
from ...modules import phase

from ..chip import Chip

class PhaseList(np.ndarray):
    ...

class SuperKN(Chip):
    """Kernel nuller representation for 4 telescopes.

    Args:
        φ (u.Quantity): (14,) array of applied OPDs (length units).
        σ (u.Quantity): (14,) array of intrinsic OPD errors.
        λ0 (u.Quantity): Reference wavelength at which matrices are defined.
        output_order (np.ndarray[int] | None): Output ordering (6 elements)
            defining output pairs.
        input_attenuation (np.ndarray[float] | None): Attenuations on the
            4 optical inputs.
        input_opd (u.Quantity | None): Relative OPDs applied to the 4 inputs.
        name (str): Descriptive name.
    """
    __slots__ = ('_parent_interferometer', '_φ', '_σ', '_λ0', '_output_order', '_input_attenuation', '_input_opd', '_name', '_raw_output_labels', '_processed_output_labels', 'nb_raw_outputs', 'nb_processed_outputs')

    def __init__(
            self,
            φ: np.ndarray[u.Quantity],
            σ: np.ndarray[u.Quantity],
            λ0: u.Quantity,
            output_order:np.ndarray[int]=None,
            input_attenuation:np.ndarray[float]=None,
            input_opd:np.ndarray[u.Quantity]=None,
            name:str='Unnamed Kernel-Nuller'
        ):

        self._raw_output_labels = ['Bright', 'Dark 1', 'Dark 2', 'Dark 3', 'Dark 4', 'Dark 5', 'Dark 6']
        self._processed_output_labels = ['Kernel 1', 'Kernel 2', 'Kernel 3']

        self.nb_raw_outputs = 7
        self.nb_processed_outputs = 3

        self._parent_interferometer = None
        self.φ = φ
        self.σ = σ
        self.λ0 = λ0
        self.output_order = output_order if output_order is not None else np.array([0, 1, 2, 3, 4, 5, 6])
        self.input_attenuation = input_attenuation if input_attenuation is not None else np.array([1.0, 1.0, 1.0, 1.0])
        self.input_opd = input_opd if input_opd is not None else np.zeros(4) * u.m
        self.name = name


        super().__init__()

    #==========================================================================
    # Attributes
    #==========================================================================

    # Phase shifters ----------------------------------------------------------

    @property
    def φ(self):
        """Applied OPD/phase per nuller element.

        Returns:
            u.Quantity: Shape (14,) in length units (e.g., meters).
        """
        return self._φ

    @φ.setter
    def φ(self, φ: np.ndarray[u.Quantity]):
        """Set applied OPDs.

        Args:
            φ (u.Quantity): Shape (14,) in a length unit.

        Raises:
            ValueError: If not a Quantity, not in length units, wrong shape,
                or contains negative values.
        """
        if type(φ) != u.Quantity:
            raise ValueError('φ must be a Quantity')
        try:
            φ.to(u.m)
        except u.UnitConversionError:
            raise ValueError('φ must be in a distance unit')
        if φ.shape != (14,):
            raise ValueError('φ must have a shape of (14,)')
        if np.any(φ < 0):
            raise ValueError('φ must be positive')
        self._φ = φ

    # Perturbations -----------------------------------------------------------

    @property
    def σ(self):
        """Intrinsic OPD errors of the nuller.

        Returns:
            u.Quantity: Shape (14,) in same unit as ``φ``.
        """
        return self._σ

    @σ.setter
    def σ(self, σ: np.ndarray[u.Quantity]):
        """Set intrinsic OPD errors.

        Args:
            σ (u.Quantity): Shape (14,) in a length unit.

        Raises:
            ValueError: If not a Quantity, not in length units, or wrong shape.
        """
        if type(σ) != u.Quantity:
            raise ValueError('σ must be a Quantity')
        try:
            σ.to(u.m)
        except u.UnitConversionError:
            raise ValueError('σ must be in a distance unit')
        if σ.shape != (14,):
            raise ValueError('σ must have a shape of (14,)')
        self._σ = σ

    # Design wavelength -------------------------------------------------------

    @property
    def λ0(self):
        """Reference wavelength of the model.

        Returns:
            u.Quantity: Reference wavelength (e.g., meters).
        """
        return self._λ0

    @λ0.setter
    def λ0(self, λ0: u.Quantity):
        """Set reference wavelength.

        Args:
            λ0 (u.Quantity): Wavelength in a convertible length unit.

        Raises:
            TypeError: If not an ``astropy.units.Quantity``.
            ValueError: If not convertible to a length unit.
        """
        if not isinstance(λ0, u.Quantity):
            raise TypeError('λ0 must be an astropy Quantity')
        try:
            λ0 = λ0.to(u.m)
        except u.UnitConversionError:
            raise ValueError('λ0 must be in a distance unit')
        self._λ0 = λ0

    # Output ordering ---------------------------------------------------------

    @property
    def output_order(self):
        """Output order of the nuller.

        Returns:
            np.ndarray[int]: Length-6 array describing the output order and
                pair structure.
        """
        return self._output_order

    @output_order.setter
    def output_order(self, output_order: np.ndarray[int]):
        """Set output order.

        Args:
            output_order (np.ndarray[int]): Permutation of [0..5] with valid
                pair structure.

        Raises:
            ValueError: If not an integer array, wrong shape, not a permutation
                of 0..5, or invalid pair configuration.
        """
        try:
            output_order = np.array(output_order, dtype=int)
        except:
            raise ValueError(f'output_order must be an array of integers, not {type(output_order)}')
        if output_order.shape != (self.nb_raw_outputs,):
            raise ValueError(f'output_order must have a shape of ({self.nb_raw_outputs},), not {output_order.shape}')
        if not np.all(np.sort(output_order) == np.arange(self.nb_raw_outputs)):
            raise ValueError(f'output_order must contain all the integers from 0 to {self.nb_raw_outputs - 1}, not {output_order}')
        
        # Specitifc criteria for valid output pairs
        if output_order[1] - output_order[2] not in [-1, 1] or output_order[3] - output_order[4] not in [-1, 1] or output_order[5] - output_order[6] not in [-1, 1]:
            raise ValueError(f'output_order contain an invalid configuration of output pairs. Found {output_order}')
        
        self._output_order = output_order

    def rebind_outputs(self, λ):
        """Correct output ordering of the SuperKN object.

        Successively obstruct two inputs and add a π/4 phase over one of the two
        remaining inputs to determine output pairing and ordering.

        Args:
            λ (u.Quantity): Observation wavelength.

        Returns:
            None: Updates ``self.output_order`` in place.
        """
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[3] = (1 + 0j) * np.sqrt(1 / 2)
        d = self.get_output_fields(ψ=ψ, λ=λ)[1:]
        k1 = np.argsort((d * np.conj(d)).real)[:2]
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[2] = (1 + 0j) * np.sqrt(1 / 2)
        d = self.get_output_fields(ψ=ψ, λ=λ)[1:]
        k2 = np.argsort((d * np.conj(d)).real)[:2]
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[1] = (1 + 0j) * np.sqrt(1 / 2)
        d = self.get_output_fields(ψ=ψ, λ=λ)[1:]
        k3 = np.argsort((d * np.conj(d)).real)[:2]
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[1] = (1 + 0j) * np.sqrt(1 / 2)
        ψ[1] *= np.exp(-1j * np.pi / 2)
        d = self.get_output_fields(ψ=ψ, λ=λ)[1:]
        dk1 = d[k1]
        diff = np.abs(dk1[0] - dk1[1])
        if diff < 0:
            k1 = np.flip(k1)
        dk2 = d[k2]
        diff = np.abs(dk2[0] - dk2[1])
        if diff < 0:
            k2 = np.flip(k2)
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[1] = (1 + 0j) * np.sqrt(1 / 2)
        ψ[2] *= np.exp(-1j * np.pi / 2)
        d = self.get_output_fields(ψ=ψ, λ=λ)[1:]
        dk3 = d[k3]
        diff = np.abs(dk3[0] - dk3[1])
        if diff < 0:
            k3 = np.flip(k3)
        self.output_order = np.concatenate([[0], k1+1, k2+1, k3+1])

    # Input properties --------------------------------------------------------

    @property
    def input_attenuation(self):
        """Input attenuations.

        Returns:
            np.ndarray[float]: Length-4 multiplicative attenuation factors.
        """
        return self._input_attenuation

    @input_attenuation.setter
    def input_attenuation(self, input_attenuation: np.ndarray[float]):
        """Set input attenuations.

        Args:
            input_attenuation (np.ndarray[float]): Length-4 attenuation factors.

        Raises:
            ValueError: If not convertible to float array or wrong shape.
        """
        try:
            input_attenuation = np.array(input_attenuation, dtype=float)
        except:
            raise ValueError(f'input_attenuation must be an array of floats, not {type(input_attenuation)}')
        if input_attenuation.shape != (4,):
            raise ValueError(f'input_attenuation must have a shape of (4,), not {input_attenuation.shape}')
        self._input_attenuation = input_attenuation

    @property
    def input_opd(self):
        """Relative OPD applied on each input.

        Returns:
            u.Quantity: Shape (4,) in length units.
        """
        return self._input_opd

    @input_opd.setter
    def input_opd(self, input_opd: np.ndarray[u.Quantity]):
        """Set input OPDs.

        Args:
            input_opd (u.Quantity): Shape (4,) in a length unit.

        Raises:
            ValueError: If not a Quantity, not in length units, or wrong shape.
        """
        if type(input_opd) != u.Quantity:
            raise ValueError('input_opd must be a Quantity')
        try:
            input_opd.to(u.m)
        except u.UnitConversionError:
            raise ValueError('input_opd must be in a distance unit')
        if input_opd.shape != (4,):
            raise ValueError('input_opd must have a shape of (4,)')
        self._input_opd = input_opd

    # Name --------------------------------------------------------------------

    @property
    def name(self):
        """Descriptive instance name.

        Returns:
            str: Kernel nuller name.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Set instance name.

        Args:
            name (str): Readable name.

        Raises:
            ValueError: If not a string.
        """
        if not isinstance(name, str):
            raise ValueError('name must be a string')
        self._name = name

    def __str__(self) -> str:
        res = f'Kernel-Nuller "{self.name}"\n'
        res += f"  φ: [{', '.join([f'{i:.2e}' for i in self.φ.value])}] {self.φ.unit}\n"
        res += f"  σ: [{', '.join([f'{i:.2e}' for i in self.σ.value])}] {self.σ.unit}\n"
        res += f"  Output order: [{', '.join([f'{i}' for i in self.output_order])}]\n"
        res += f"  Input attenuation: [{', '.join([f'{i:.2e}' for i in self.input_attenuation])}]\n"
        res += f"  Input OPD: [{', '.join([f'{i:.2e}' for i in self.input_opd.value])}] {self.input_opd.unit}"
        return res.replace('e+00', '')

    def __repr__(self) -> str:
        return self.__str__()
    
    # Parent interferometer ---------------------------------------------------

    @property
    def parent_interferometer(self):
        """Parent interferometer associated with this kernel nuller.

        Read-only property set during association with an Interferometer object.
        """
        return self._parent_interferometer

    @parent_interferometer.setter
    def parent_interferometer(self, parent_interferometer):
        """Setter is disabled; ``parent_interferometer`` is read-only.

        Raises:
            ValueError: Always raised; property is read-only.
        """
        raise ValueError('parent_interferometer is read-only')
    
    #==========================================================================
    # Methods
    #==========================================================================

    # Wave propagation --------------------------------------------------------

    def get_output_fields(self, ψ: np.ndarray[complex], λ: u.Quantity) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Propagate input fields through the kernel nuller.
        Args:
            ψ (np.ndarray[complex]): Input complex fields for the 4 channels (shape (4,)).
            λ (u.Quantity): Wavelength for propagation.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]: Output complex
            fields (shape (7,)).
        """
        φ = self.φ.to(λ.unit).value
        σ = self.σ.to(λ.unit).value
        λ0 = self.λ0.to(λ.unit).value
        ψ *= self.input_attenuation
        ψ *= np.exp(-1j * 2 * np.pi * self.input_opd.to(λ.unit).value / λ.value)
        return get_output_fields_jit(ψ=ψ, φ=φ, σ=σ, λ=λ.value, λ0=λ0, output_order=self.output_order)
    
    def process_outputs(self, out: np.ndarray[float]) -> np.ndarray[float]:
        """
        Compute processed kernel outputs from raw output intensities.
        Args:
            out (np.ndarray[float]): Raw output intensities (shape (7,)).
        Returns:
            np.ndarray[float]: Processed kernel outputs (shape (4,)).
        """
        return process_outputs_jit(out)
    
    # Plotting ----------------------------------------------------------------

    def plot_output_phase(self, λ: u.Quantity, ψ: Optional[np.ndarray]=None, plot: bool = True) -> Optional[Any]:
        """Plot output phases and amplitudes of the nuller.

        Computes output responses for each isolated input and plots the phase
        and amplitude of null, dark, and bright outputs on polar diagrams.

        Args:
            λ (u.Quantity): Wavelength for the simulation.
            ψ (Optional[np.ndarray]): Input complex amplitudes (default [0.5,...]).
            plot (bool): If ``True``, display the figure; if ``False``, return the image bytes.
        """
        if ψ is None:
            ψ = np.array([0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j])
        ψ1 = np.array([ψ[0], 0, 0, 0])
        ψ2 = np.array([0, ψ[1], 0, 0])
        ψ3 = np.array([0, 0, ψ[2], 0])
        ψ4 = np.array([0, 0, 0, ψ[3]])
        out1 = self.get_output_fields(ψ1, λ)
        out2 = self.get_output_fields(ψ2, λ)
        out3 = self.get_output_fields(ψ3, λ)
        out4 = self.get_output_fields(ψ4, λ)

        out2 = np.abs(out2[0]) * np.exp(1j * (np.angle(out2[0]) - np.angle(out1[0])))
        out3 = np.abs(out3[0]) * np.exp(1j * (np.angle(out3[0]) - np.angle(out1[0])))
        out4 = np.abs(out4[0]) * np.exp(1j * (np.angle(out4[0]) - np.angle(out1[0])))
        out1 = np.abs(out1) * np.exp(1j * 0)

        n_out = len(out1)
        outs = np.array([out1, out2, out3, out4])

        _, axs = plt.subplots(n_out, 1, figsize=(5, 5*n_out), subplot_kw={'projection': 'polar'})
        for i in range(len(out1)):

            colors = ['yellow', 'green', 'red', 'blue']
            for j, out in enumerate(outs):

                axs[j].scatter(np.angle(out), np.abs(out), color=colors[j], label=f'Input {j + 1}', alpha=0.5)
                axs[j].plot([0, np.angle(out)], [0, np.abs(out)**2], color=colors[j], alpha=0.5)

        m = np.max(np.abs(outs)**2)
        for i, ax in enumerate(axs.flatten()):
            ax.set_ylim(0, m)
            ax.set_title(f'{self._raw_output_labels[i]} output')
        axs[0].legend()
        if not plot:
            plot = BytesIO()
            plt.savefig(plot, format='png')
            plt.close()
            return plot.getvalue()
        plt.show()


#==============================================================================
# Numba-accelerated functions
#==============================================================================

@nb.njit()
def get_output_fields_jit(
        ψ: np.ndarray[complex],
        φ: np.ndarray[float],
        σ: np.ndarray[float],
        λ: float,
        λ0: float,
        output_order: np.ndarray[int]
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], float]:
    """Simulate a 4-telescope Kernel Nuller propagation (numeric approach).

    Note: Does not account for input attenuation and OPD.

    Args:
        ψ (np.ndarray[complex]): Array of 4 input complex amplitudes.
        φ (np.ndarray[float]): Array of 14 injected OPDs (wavelength units).
        σ (np.ndarray[float]): Array of 14 intrinsic OPD errors (wavelength units).
        λ (float): Wavelength of the light.
        λ0 (float): Reference wavelength (wavelength units).
        output_order (np.ndarray[int]): Order of the outputs.

    Returns:
        tuple: (nulls, darks, bright)
            - nulls: Array of 3 null outputs (complex fields)
            - darks: Array of 6 dark outputs (complex fields)
            - bright: Bright output (complex field)
    """
    λ_ratio = λ0 / λ
    
    # 2x2 Nuller
    N = 1 / np.sqrt(2) * np.array([
        [1,  1],
        [1, -1]
    ],dtype=np.complex128)

    # Adjust for wavelength
    Na = np.abs(N)
    Nφ = np.angle(N)
    N = Na * np.exp(1j * Nφ * λ_ratio)

    # 2x2 Cross-Recombiner
    θ: float = np.pi / 2
    R = 1 / np.sqrt(2) * np.array([
        [np.exp(1j * θ / 2), np.exp(-1j * θ / 2)],
        [np.exp(-1j * θ / 2), np.exp(1j * θ / 2)]
    ],dtype=np.complex128)

    # Adjust for wavelength
    Ra = np.abs(R)
    Rφ = np.angle(R)
    R = Ra * np.exp(1j * Rφ * λ_ratio)

    # Add first layer of perturbations & shifts
    Φ = φ + σ # merge perturbations and shifts
    ψ0 = phase.shift_jit(ψ, Φ[:4], λ)

    # First layer of nullers
    ψtmp1 = N @ ψ0[:2]
    ψtmp2 = N @ ψ0[2:]
    ψ1 = np.array([ψtmp1[0], ψtmp1[1], ψtmp2[0], ψtmp2[1]], dtype=np.complex128)

    # Second layer of perturbations & shifts
    ψ1 = phase.shift_jit(ψ1, φ[4:8], λ)

    # Second layer of nullers
    ψtmp1 = N @ np.array([ψ1[0], ψ1[2]])
    ψtmp2 = N @ np.array([ψ1[1], ψ1[3]])
    ψ2 = np.array([ψtmp1[0], ψtmp1[1], ψtmp2[0], ψtmp2[1]], dtype=np.complex128)

    # Splitters and final bright output
    ψb = ψ2[0]
    ψ3 = ψ2[1:].repeat(2) * 1 / np.sqrt(2) 

    # Final perturbations & shifts
    ψ3 = phase.shift_jit(ψ3, φ[8:], λ)

    # Final recombination
    ψtmp1 = R @ np.array([ψ3[0], ψ3[2]])
    ψtmp2 = R @ np.array([ψ3[1], ψ3[4]])
    ψtmp3 = R @ np.array([ψ3[3], ψ3[5]])

    ψout = np.array([ψb, ψtmp1[0], ψtmp1[1], ψtmp2[0], ψtmp2[1], ψtmp3[0], ψtmp3[1]], dtype=np.complex128)
    return ψout[output_order]

@nb.njit()
def process_outputs_jit(out: np.ndarray[complex]) -> np.ndarray[float]:
    """Compute kernel outputs from dark outputs intensities.

    Args:
        darks (np.ndarray[complex]): Array of 6 dark outputs (complex fields).
    """
    k1 = out[1] - out[2]
    k2 = out[3] - out[4]
    k3 = out[5] - out[6]
    return np.array([k1, k2, k3], dtype=np.float64)