import math
import numpy as np
from scipy.special import jv  # , ndtri
import scipy
from types import FunctionType
import warnings


STANDARD_PREFIXES = {"a": 10e-18,
                     "f": 10e-15,
                     "p": 10e-12,
                     "n": 10e-9,
                     "u": 10e-6,
                     "m": 10e-3,
                     "c": 10e-2,
                     "d": 10e-1,
                     "da": 10,
                     "h": 10e2,
                     "k": 10e3,
                     "M": 10e6,
                     "G": 10e9,
                     "T": 10e12,
                     "P": 10e15,
                     "E": 10e18,
                     "atto": 10e-18,
                     "femto": 10e-15,
                     "pico": 10e-12,
                     "nano": 10e-9,
                     "micro": 10e-6,
                     "milli": 10e-3,
                     "centi": 10e-2,
                     "deci": 10e-1,
                     "deca": 10,
                     "hecto": 10e2,
                     "kilo": 10e3,
                     "mega": 10e6,
                     "giga": 10e9,
                     "tera": 10e12,
                     "peta": 10e15,
                     "exa": 10e18}


def polar_to_cartesian(r_phi: list | tuple | np.ndarray) -> np.ndarray:
    """Converts vector in polar coordinates to one in cartesian coordinates.

    Parameters
    ----------
    r_phi : list | tuple | np.ndarray
        A 2-vector with polar coordinate `r` as entry one and `phi` as entry
        two.

    Returns
    -------
    np.ndarray
        A 2-vector with the corresponding cartesian coordinates `x` and `y`.
    """
    r, phi = r_phi

    return np.array([r*np.cos(phi), r*np.sin(phi)])


def cartesian_to_polar(x_y: list | tuple | np.ndarray) -> np.ndarray:
    """Converts vector in cartesian coordinates to one in polar coordinates.

    Parameters
    ----------
    x_y : list | tuple | np.ndarray
        A 2-vector with cartesian coordinate `x` as entry one and `y` as entry
        two.

    Returns
    -------
    np.ndarray
        A 2-vector with the corresponding polar coordinates `r` and `phi`.
    """
    r = np.linalg.norm(x_y, axis=0)
    phi = np.arctan2(x_y[1, :], x_y[0, :])

    return np.array([r, phi])


def convert_units(measurement: np.ndarray, from_unit: str | float,
                  to_unit: str | float) -> np.ndarray:
    """Converts values in measurement from `from_unit` to `to_unit`.

    Parameters
    ----------
    measurement : np.ndarray
        An array of values.
    from_unit : str | float
        Units of values in measurement.
    to_unit : str | float
        Units of values to be returned.

    Returns
    -------
    np.ndarray
        A new array where the values are converted to the new units.
    """
    conversion_factor = parse_unit(from_unit) / parse_unit(to_unit)
    return conversion_factor * measurement


def parse_unit(unit: str) -> float:
    """Parses a unit string into its corresponding magnitude.

    Parameters
    ----------
    unit : str
        The desired unit in string format.

    Returns
    -------
    float
        The magnitude of the unit.
    """
    for key in STANDARD_PREFIXES.keys():
        if unit.startswith(key) and len(key) < len(unit):
            return STANDARD_PREFIXES[key]
    else:
        raise ValueError("Unrecognized unit prefix.")


def random_airy_disc(size: int | tuple, rng=None) -> np.ndarray:
    """Randomly samples values from the airy disc.

    Parameters
    ----------
    size : int | tuple
        Output shape.
    rng : _type_, optional
        Numpy rng object used for random sampling, by default None

    Returns
    -------
    np.ndarray
        Drawn samples from the 2D random airy disc in cartesian coordinates.
    """

    if rng is None:
        rng = np.random.default_rng()

    def cdf(x):
        ans = 1 - jv(0, x)**2 - jv(1, x)**2
        return ans * (x > 0)

    x_values = np.linspace(0, 10000, 1000000)
    inverse_cdf = scipy.interpolate.CubicSpline(cdf(x_values), x_values,
                                                bc_type='clamped',
                                                extrapolate=None)

    # ensures that inverse is at most 10000
    uniform_values = 0.9999363406112249 * rng.uniform(size=size)
    r_phi_airy = np.vstack([inverse_cdf(uniform_values),
                            rng.uniform(0, 2*np.pi, size=size)])

    return polar_to_cartesian(r_phi_airy)


def airy_disc_pdf(x: np.ndarray) -> np.ndarray:
    """Computes the probability density function corresponding to a 1-
    normalized airy disc for the supplied values. Probability density is
    marginalized for r.

    Parameters
    ----------
    x : np.ndarray
        The values to compute the probability density function at.

    Returns
    -------
    np.ndarray
        The values of the function at the supplied values.
    """
    return (3 * np.pi / 4) * (jv(1, x) / x)**2


def airy_disc_cdf(x: np.ndarray) -> np.ndarray:
    """Computes the cumulative density function corresponding to a 1-
    normalized airy disc for the supplied values.

    Parameters
    ----------
    x : np.ndarray
        The values to compute the cumulative density function at.

    Returns
    -------
    np.ndarray
        The values of the function at the supplied values.
    """

    return (np.pi / 4) * (((2 * x**2 - 1) * jv(1, x)**2) / x +
                          2 * x * jv(0, x)**2 - 2 * jv(0, x) * jv(1, x))


def inverse_airy_disc_cdf(x: np.ndarray) -> np.ndarray:
    """Computes values of the inverse airy disc function at given x values.

    Parameters
    ----------
    x : np.ndarray
        Values at which to compute the inverse.

    Returns
    -------
    np.ndarray
        Values of the inverse function.
    """
    def cdf(x):
        ans = 1 - jv(0, x)**2 - jv(1, x)**2
        return ans * (x > 0)

    output = np.zeros(x.shape)
    for idx, val in np.ndenumerate(x):
        output[*idx] = invert_monotonic_function(
            val, cdf, scipy.stats.rayleigh.ppf(val, scale=np.pi/2))
        # ndtri(val))
    return output


def invert_monotonic_function(y: float,
                              fun: FunctionType,
                              guess: float,
                              tol=0.001,
                              attempts=1000) -> float:
    """Inverts a monotonically increasing function using bisection search.

    Parameters
    ----------
    y : float
        The desired value `f(x)` should produce.
    fun : FunctionType
        The function `f(x)`.
    guess : float
        An initial guess for `f^-1(y)`.
    tol : float, optional
        The error tolerated in the solution, by default 0.0001
    attempts : int, optional
        The number of attempts for the bisection search, by default 1000

    Returns
    -------
    float
        The solution to the inverse problem, `f^-1(y)`.

    Raises
    ------
    ValueError
        For very poor initial guesses, the algorithm may fail to find a
        suitable lower or upper bound for the search.
    """

    error = abs(y - fun(guess))

    if error < tol:
        return guess

    # search for upper bound
    if y >= fun(guess):
        lo = guess
        hi = guess + np.finfo(float).eps
        for _ in range(attempts):
            if fun(hi) <= y:
                hi += 10 * hi
            else:
                break
        else:
            raise ValueError("Could not find upper bound for bisection search "
                             "with current guess and attempts.")

    # search for lower bound
    else:
        lo = guess - np.finfo(float).eps
        hi = guess
        for _ in range(attempts):
            if fun(lo) > y:
                lo -= 10 * lo
            else:
                break
        else:
            raise ValueError("Could not find lower bound for bisection search "
                             "with current guess and attempts.")

    # perform bisection search
    for _ in range(attempts):
        if error >= tol:
            x_0 = (lo + hi) / 2
            if fun(x_0) < y:
                lo = x_0
            else:
                hi = x_0
            error = abs(y - fun(x_0))
        else:
            return x_0
    else:
        warnings.warn(
            f"Attempts exceeded for inverse problem. "
            f"Improve initial guess or increase attempts. Error: {error}",
            RuntimeWarning)


class Minflux2D:

    def __init__(self, fwhm=500, unit="nm"):
        self.unit = unit

        # controls the offsets for the excitation pattern. In cartesian
        # coordinates in nanometers.
        self.beam_offsets_norm = np.array([])
        self.beam_pattern = None  # list of beam patterns
        self.Ls = []  # list of L values
        self.fwhm = fwhm
        self.multiplex_cycle = None

    @property
    def beam_offsets_norm(self) -> np.ndarray:
        """Returns the normalized offsets for the beam pattern.

        Returns
        -------
        list
            List of normalized beam offsets in cartesian coordinates.
        """
        return self._beam_offsets_norm

    @beam_offsets_norm.setter
    def beam_offsets_norm(self, beam_offsets: list) -> None:
        """Sets the beam pattern given a list of offsets in cartesian
        coordinates.

        Parameters
        ----------
        beam_offsets : list
            A list of beam offsets in cartesian coordinates.
        """

        self._beam_offsets_norm = np.array([offset / np.linalg.norm(offset)
                                            if np.linalg.norm(offset) > 0
                                            else offset
                                            for offset in beam_offsets])

    @property
    def beam_offsets_polar(self) -> np.ndarray:
        """Returns the offsets for the beam pattern in polar coordinates.

        Returns
        -------
        list
            List of beam offsets in polar coordinates.
        """
        return microscope_utilities.cartesian_to_polar(self.beam_offsets_norm)

    @beam_offsets_polar.setter
    def beam_offsets_polar(self, beam_offsets: list, unit=None) -> None:
        """Sets the beam pattern given a list of offsets in polar coordinates.

        Parameters
        ----------
        beam_offsets : list
            A list of beam offsets in polar coordinates.
        unit : str, optional
            The physical unit for the list. If None, the units are assumed to
            be the same as the unit property, by default None
        """

        def convert_units_and_coords(offset):

            if unit is None:
                return microscope_utilities.polar_to_cartesian(offset)
            else:
                return microscope_utilities.polar_to_cartesian(
                    microscope_utilities.convert_units(offset, from_unit=unit,
                                                       to_unit=self.unit))

        self.beam_offsets_norm = [convert_units_and_coords(offset)
                                  for offset in beam_offsets]

    @property
    def beam_offsets_cartesian(self) -> np.ndarray:
        """Returns the offsets for the beam pattern in cartesian coordinates.
        Convenience function for `beam_offsets_norm`.

        Returns
        -------
        list
            List of beam offsets in cartesian coordinates.
        """
        return self.beam_offsets_norm

    @beam_offsets_cartesian.setter
    def beam_offsets_cartesian(self, beam_offsets: list, unit=None) -> None:
        """Sets the beam pattern given a list of offsets in cartesian
        coordinates.

        Parameters
        ----------
        beam_offsets : list
            A list of beam offsets in cartesian coordinates.
        unit : str, optional
            The physical unit for the list. If None, the units are assumed to
            be the same as the unit property, by default None
        """
        if unit is None:
            self.beam_offsets_norm = beam_offsets

        else:
            self.beam_offsets_norm = [microscope_utilities.convert_units(
                offset, from_unit=unit, to_unit=self.unit)
                for offset in beam_offsets]

    @property
    def beam_pattern(self) -> FunctionType | None:
        """The beam pattern for performing minflux localization

        Returns
        -------
        FunctionType
            The beam pattern function or None. Should take in r coordinates, z
            coordinates, and a center for the beam pattern.
        """
        return self._beam_pattern

    @beam_pattern.setter
    def beam_pattern(self, beam_pattern) -> None:
        """Adds beam pattern for performing minflux localization.

        Parameters
        ----------
        beam_pattern : function
            Beam pattern to use for localization.

        Raises
        ------
        TypeError
            Called if `beam_pattern` is not a callable function.
        """
        if callable(beam_pattern):
            self._beam_pattern = beam_pattern

        elif beam_pattern is None:
            self._beam_pattern = beam_pattern

        else:
            raise TypeError(f"Beam pattern {beam_pattern} is not callable.")

    @property
    def multiplex_cycle_time(self) -> float | None:
        """The total time taken for an imaging cycle

        Returns
        -------
        float | None
            The total time taken for an imaging cycle in us
            or None if the cycle is not set.
        """
        multiplex_cycle_time = sum(
            self.beam_offsets_norm.shape[0] *
            [self.multiplex_cycle["gate_delay"] +
                self.multiplex_cycle["detection_excitation_gate"]]) + \
            self.multiplex_cycle["localization_window"]  # us
        return multiplex_cycle_time

    def add_minflux_L(self, L: float, unit=None) -> None:
        """Adds a parameter L for doing the minflux localization.

        Parameters
        ----------
        L : float
            L, the probing width
        unit : str, optional
            The physical unit for the list. If None, the units are assumed to
            be the same as the unit property, by default None
        """
        if unit is None:
            self.Ls.append(L)

        else:
            self.Ls.append(microscope_utilities.convert_units(
                L, from_unit=unit, to_unit=self.unit))

    def add_minflux_Ls(self, Ls: list[float], unit=None) -> None:
        """Adds all Ls in list for doing the minflux localization.

        Parameters
        ----------
        L : list[float]
            A list of probing widths
        unit : str, optional
            The physical unit for the list. If None, the units are assumed to
            be the same as the unit property, by default None
        """
        for l_val in Ls:
            self.add_minflux_L(l_val, unit)

    # TODO: reimplement
    def calculate_parameter_vectors(self, r, center):
        # Balzarotti S4

        parameter_vectors = []

        def two_dimensional_beam_pattern(r, center):
            return self.beam_pattern(r, 0, center)

        for L_value in self.Ls:
            intensities = np.stack(
                [two_dimensional_beam_pattern(r,
                 center + (L_value / 2) *
                 np.append(offset, 0).reshape(center.shape))
                 for offset in self.beam_offsets_norm], axis=-1)

            # (..., n_beams)
            parameter_vectors.append(
                intensities / np.sum(intensities, axis=-1, keepdims=True))
        return parameter_vectors

    def _calculate_parameter_vector(self,
                                    localization_chunk: np.ndarray,
                                    L_value: float,
                                    beam_position: np.ndarray,
                                    photon_emission_rate: float,
                                    background_emission_rate=0
                                    ) -> tuple[np.ndarray, int, float]:
        """Calculates a parameter vector for localizations in the chunk.

        Parameters
        ----------
        localization_chunk : np.ndarray
            A series of localizations to calculate the parameter vector over.
            It should have shape (number of localizations,
            number of spatial dimensions).
        L_value : float
            The L value used in the tracking experiment. The diameter of the
            probing pattern
        beam_position : np.ndarray
            The position of the excitation beam pattern
        photon_emission_rate : float
            Can be float or np.inf. Gives the rate of photon emissions per unit
            time and excitation strength.
        background_emission_rate : int, optional
            Gives the background rate of photon emissions per unit time,
            by default 0

        Returns
        -------
        np.ndarray
            The observed photon counts.

        Raises
        ------
        ValueError
            Raised if multiplex cycle is not set prior to tracking.
        """

        # add z value to beam offsets
        beam_offsets_norm_z = np.column_stack(
            (self.beam_offsets_norm, np.zeros(self.beam_offsets_norm.shape[0]))
            )

        if self.multiplex_cycle is None:
            raise ValueError("Multiplex cycle is None. Please define a cycle "
                             "before tracking.")

        # compute timing of localization chunk
        dt = self.multiplex_cycle_time / localization_chunk.shape[0]

        cycle_offsets = np.repeat(
            beam_offsets_norm_z,
            self.beam_offsets_norm.shape[0] *
            [max((self.multiplex_cycle["gate_delay"] +
                  self.multiplex_cycle["detection_excitation_gate"]) / dt, 1)],
            axis=0
        )

        cycle_offsets = (L_value / 2) * np.concatenate(
            [cycle_offsets, np.repeat(
                beam_offsets_norm_z[0, :].reshape((1, 3)),
                [max(self.multiplex_cycle["localization_window"] / dt, 0)],
                axis=0)],
            axis=0
        )

        if localization_chunk.shape[0] > 1:
            assert localization_chunk.shape[0] == cycle_offsets.shape[0]

        # check if z coordinates are provided
        if localization_chunk.shape[1] == 3:
            ebp_values = self.beam_pattern(
                localization_chunk[:, :2], localization_chunk[:, -1],
                cycle_offsets + beam_position)

        elif localization_chunk.shape[1] == 2:
            ebp_values = self.beam_pattern(
                localization_chunk, 0, cycle_offsets + beam_position)

        else:
            raise ValueError(f"Unsupported dimension for localizations: "
                             f"{localization_chunk.shape}")

        # TODO: figure out why too-low photon emissions don't work

        # check for finite photon emission rate
        if np.isfinite(photon_emission_rate):
            photon_emissions = np.random.poisson(
                photon_emission_rate * dt * ebp_values +
                background_emission_rate * dt
            )

        # infinite photon emission rate
        else:
            photon_emissions = np.full(ebp_values.shape, np.inf)

        if localization_chunk.shape[0] == 1:
            full_cycle = np.arange(self.beam_offsets_norm.shape[0]) + 1

        else:
            # cycle for one round of excitation
            excitation_cycle = np.array(
                round(self.multiplex_cycle["gate_delay"] / dt) * [0] +
                round(self.multiplex_cycle["detection_excitation_gate"] / dt) *
                [1]
            )

            # cycle for all rounds of excitation
            full_cycle = [(i + 1) * excitation_cycle
                          for i in range(self.beam_offsets_norm.shape[0])]
            full_cycle.append(
                round(self.multiplex_cycle["localization_window"] / dt) * [0]
            )
            full_cycle = np.concatenate(full_cycle)

        if np.isposinf(photon_emission_rate):
            intensities = np.array(
                [[np.sum(ebp_values[full_cycle == i + 1])
                  for i in range(self.beam_offsets_norm.shape[0])]])

        else:
            intensities = np.array(
                [[np.sum(photon_emissions[full_cycle == i + 1])
                  for i in range(self.beam_offsets_norm.shape[0])]])

        return intensities

    def LMS_estimator(self,
                      parameter: np.ndarray,
                      L_value: float,
                      fwhm: float) -> np.ndarray:
        """Least mean squares estimator for particle position.
        Calculates the least mean squares estimator (Eq. S50) from Balzarotti
        et al.

        Parameters
        ----------
        parameter : np.ndarray
            A parameter vector giving the proportion of photons from each
            beam offset.
        L_value : float
            The L value used to generate the parameter vector. The diameter of
            the probing pattern in nm.
        fwhm : float
            The full-width half maximum (fwhm) of the donut excitation beam
            pattern used to generate the parameter vector.

        Returns
        -------
        np.ndarray
            The LMS estimate for the particle position.
        """

        prefactor = -1 / (1 - (np.log(2) * L_value ** 2 / fwhm ** 2))

        p_r = (L_value / 2) * parameter @ self.beam_offsets_norm[:]

        return prefactor * p_r

    def mLMS_estimator(self,
                       parameter: np.ndarray,
                       beta: np.ndarray,
                       L_value: float,
                       fwhm: float) -> np.ndarray:
        """Modified least mean squares estimator for particle position.
        Calculates the least mean squares estimator (Eq. S51) from Balzarotti
        et al.

        Parameters
        ----------
        parameter : np.ndarray
            A parameter vector giving the proportion of photons from each
            beam offset.
        beta : np.ndarray
            The beta values used for the p_0 expansion in the mLMS estimator.
            See Balzarotti et al. for more details.
        L_value : float
            The L value used to generate the parameter vector. The diameter of
            the probing pattern in nm.
        fwhm : float
            The full-width half maximum (fwhm) of the donut excitation beam
            pattern used to generate the parameter vector.

        Returns
        -------
        np.ndarray
            The mLMS estimate for the particle position.
        """

        beta_p_zero_j = np.sum(beta * (
            parameter[:, 0] ** np.arange(
                beta.shape[-1])[:, None]).reshape(parameter.shape[0], -1),
                axis=1)

        return beta_p_zero_j * self.LMS_estimator(parameter, L_value, fwhm)

    def expected_position_estimator(self,
                                    parameter: np.ndarray,
                                    beta_bar: np.ndarray,
                                    L_value: float,
                                    fwhm: float,
                                    n_photons: float) -> np.ndarray:
        """Expectation of the mLMS position estimator.
        Calculates the closed form expression for the expectation of the mLMS
        position estimator according to equation S54 from Balzarotti et al.

        Parameters
        ----------
        parameter : np.ndarray
            A parameter vector giving the proportion of photons from each
            beam offset.
        beta_bar : np.ndarray
            The beta values used for the p_0 expansion in the mLMS estimator.
            See Balzarotti et al. for more details.
        L_value : float
            The L value used to generate the parameter vector. The diameter of
            the probing pattern in nm.
        fwhm : float
            The full-width half maximum (fwhm) of the donut excitation beam
            pattern used to generate the parameter vector.
        n_photons : float
            The total number of photons collected.

        Returns
        -------
        np.ndarray
            The mean localization for the molecule given the provided parameter
            vector.
        """

        assert len(beta_bar) == 3
        N = n_photons
        if np.isposinf(n_photons):
            beta_expansion = beta_bar[0] + beta_bar[1] * parameter[:, 0] + \
                beta_bar[2] * parameter[:, 0] ** 2
        else:
            beta_expansion = beta_bar[0] + \
                ((N - 1) / N) * (beta_bar[1] +
                                 beta_bar[2] / N) * parameter[:, 0] + \
                beta_bar[2] * ((N - 1) * (N - 2) / N ** 2) * \
                parameter[:, 0] ** 2

        # results from an algebraic manipulation of S54
        return np.stack([beta_expansion, beta_expansion]).T * \
            self.LMS_estimator(parameter, L_value, fwhm)

    def bias(self,
             r_bar: np.ndarray,
             beta_bar: np.ndarray,
             L_value: float,
             fwhm: float,
             n_photons: float,
             center=np.zeros((1, 3))) -> np.ndarray:
        """The bias of the mLMS estimator.
        The bias is computed as the difference between the expected value of
        the estimator and the molecule's true position.

        Parameters
        ----------
        r_bar : np.ndarray
            The true position of the molecule in cartesian coordinates.
        beta_bar : np.ndarray
            The beta values used for the p_0 expansion in the mLMS estimator.
            See Balzarotti et al. for more details.
        L_value : float
            The L value used to generate the parameter vector. The diameter of
            the probing pattern in nm.
        fwhm : float
            The full-width half maximum (fwhm) of the donut excitation beam
            pattern used to generate the parameter vector.
        n_photons : float
            The total number of photons collected.
        center : np.ndarray, optional
            The center of the beam pattern, by default np.zeros((1, 3))

        Returns
        -------
        np.ndarray
            The bias for a molecule at the given position.
        """
        # Balzarotti S52
        parameter_vector = self.calculate_parameter_vectors(r_bar, center)[0]
        return self.expected_position_estimator(parameter_vector, beta_bar,
                                                L_value, fwhm,
                                                n_photons) - r_bar

    def define_multiplex_cycle(self,
                               detection_excitation_gate: float,
                               gate_delay: float,
                               localization_window: float) -> None:
        """Defines the multiplex cycle for the MINFLUX experiment.
        The multiplex cycle defines how long each portion of the localization
        routine takes in us.

        Parameters
        ----------
        detection_excitation_gate : float
            The time spent exciting the molecule and detecting photons in
            microseconds.
        gate_delay : float
            The time between molecule excitations in microseconds.
        localization_window : float
            The time required to do the calculations to localize the particle.
        """
        self.multiplex_cycle = {}
        self.multiplex_cycle["detection_excitation_gate"] = \
            detection_excitation_gate
        self.multiplex_cycle["gate_delay"] = gate_delay
        self.multiplex_cycle["localization_window"] = localization_window

    def _track_particle(self,
                        trajectory: np.ndarray,
                        dt: float,
                        L_value: float,
                        photon_emission_rate: float,
                        background_emission_rate: float,
                        minimum_photon_threshold: int,
                        background_threshold: float,
                        maximum_dark_time: float,
                        stickiness: int = 4,
                        beta: None | list = None,
                        initial_beam_position: None | np.ndarray = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tracks a single particle given its trajectory

        Parameters
        ----------
        trajectory : np.ndarray
            The trajectory of a single particle with dimensions (number of
            localizations, number of dimensions)
        dt : float
            The timestep corresponding to the provided trajectory in us
        L_value : float
            The L value used in the tracking experiment. The diameter of the
            probing pattern
        photon_emission_rate : float
            Can be float or np.inf. Gives the rate of photon emissions per unit
            time and excitation strength.
        background_emission_rate : float
            Gives the background rate of photon emissions per unit time,
            by default 0
        minimum_photon_threshold : int
            The minimum number of photons required to continue tracking.
        background_threshold : float
            The threshold used to determine signal above background in units of
            Hz.
        maximum_dark_time : float
            The maximum time signal below background will be tolerated before
            abandoning tracking.
        stickiness : int, optional
            The number of tracking cycles allowed before terminating tracking,
            by default 4.
        beta : np.ndarray, optional
            The beta parameters used in the tracking experiment,
            by default None
        initial_beam_position : np.ndarray, optional
            The initial position of the donut fused in the tracking experiment,
            by default np.zeros((1, 3)).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The observed localizations, the associated errors with the
            actual particle location, and the photon counts measured in the
            MINFLUX experiment

        Raises
        ------
        ValueError
            Raised if dt is greater than the multiplex cycle time and it can't
            be divided into it.
        ValueError
            Raised if dt is less than the multiplex cycle time and it doesn't
            divide it.
        """

        if beta is None:
            def estimator(parameter): return self.LMS_estimator(
                parameter, L_value, self.fwhm)
        else:
            def estimator(parameter): return self.mLMS_estimator(
                parameter, beta, L_value, self.fwhm)

        if initial_beam_position is None:
            initial_beam_position = np.zeros((1, 3))

        # check that dt is valid
        if (dt > self.multiplex_cycle_time / 2 and not
            (math.isclose(
                math.remainder(dt, self.multiplex_cycle_time),
                0, abs_tol=1e-10) or
                math.isclose(
                math.remainder(dt, self.multiplex_cycle_time),
                self.multiplex_cycle_time, abs_tol=1e-10))):
            raise ValueError("dt does not divide evenly into the multiplex "
                             "cycle time.")

        elif (dt < self.multiplex_cycle_time and not math.isclose(
                  math.remainder(self.multiplex_cycle_time, dt),
                  0, abs_tol=1e-10)):

            raise ValueError("The multiplex cycle time does not divide "
                             "evenly into dt.")

        # split the trajectory into chunks
        if (dt > self.multiplex_cycle_time or
                math.isclose(dt, self.multiplex_cycle_time)):
            localization_chunks = np.repeat(
                trajectory,
                trajectory.shape[0] * [round(dt / self.multiplex_cycle_time)],
                axis=0
            )

            # iterate over rows preserving dimension
            # https://stackoverflow.com/questions/56235771/iterate-numpy-array-and-keep-dimensions
            localization_chunks = localization_chunks[:, None]

        else:
            localization_chunks = np.split(
                trajectory,
                trajectory.shape[0] / (self.multiplex_cycle_time / dt),
                axis=0
            )

        # initialize output arrays
        beam_position = initial_beam_position
        position_estimates = np.full((2, len(localization_chunks)), np.nan)
        errors = np.full((trajectory.shape[1], len(localization_chunks)),
                         np.nan)
        photon_emission_rates = \
            {key: np.full((len(localization_chunks)), np.nan)
             for key in ["eco", "ecc", "efo", "efc", "cfr"]}

        # initialize counters
        running_count = 0
        running_intensities = np.zeros((1, self.beam_offsets_norm.shape[0]))
        stickiness_counter = stickiness
        running_dark_time = 0
        center_counts = 0
        offset_counts = 0
        n_itr = 0

        # iterate over chunks
        for i, chunk in enumerate(localization_chunks):
            intensities = self._calculate_parameter_vector(
                chunk, L_value, beam_position, photon_emission_rate,
                background_emission_rate)
            n_itr += 1

            # calculate running photon counts at offset and center
            center_counts += np.sum(intensities[:, 0])
            offset_counts += np.sum(intensities[:, 1:])
            if np.isfinite(photon_emission_rate):
                center_emission_rate = center_counts / \
                    (1e-6 *
                     self.multiplex_cycle["detection_excitation_gate"])
                offset_emission_rate = offset_counts / \
                    ((intensities.shape[-1] - 1) * 1e-6 *
                     self.multiplex_cycle["detection_excitation_gate"])
                center_frequency_ratio = center_emission_rate / \
                    offset_emission_rate
            else:
                center_emission_rate, offset_emission_rate = np.inf, np.inf
                center_frequency_ratio = center_counts / offset_counts

            running_count += np.sum(intensities)
            running_intensities += intensities
            running_dark_time += self.multiplex_cycle_time

            # record photon emissions
            for stat, val in zip(["eco", "ecc", "efo", "efc", "cfr"],
                                 [offset_counts, center_counts,
                                  offset_emission_rate, center_emission_rate,
                                  center_frequency_ratio]):
                photon_emission_rates[stat][i] = val

            # check if validity conditions are all met, observed photons
            # greater than minimum threshold, and observed emission rate
            # greater than snr threshold
            if (running_count >= minimum_photon_threshold and
                    offset_emission_rate >= background_threshold):
                parameter_v = running_intensities / \
                    np.sum(running_intensities, axis=-1, keepdims=True)

                beam_position[:, :2] += estimator(parameter_v)

                # error is taken as the difference between the position and
                # the mean of the chunk
                errors[:, i] = beam_position - np.mean(chunk, axis=0)
                position_estimates[:, i] = beam_position[:, :2]

                # reset stuff
                running_count = 0
                running_intensities = np.zeros(intensities.shape)
                stickiness_counter = stickiness
                running_dark_time = 0
                center_counts = 0
                offset_counts = 0
                n_itr = 0

            # Check termination criteria
            else:
                if stickiness_counter <= 0:
                    return position_estimates, errors, photon_emission_rates

                if running_count < minimum_photon_threshold:
                    stickiness_counter -= 1

                if running_dark_time >= maximum_dark_time:
                    stickiness_counter -= 1
                    running_dark_time = 0

        return position_estimates, errors, photon_emission_rates

    @staticmethod
    def calc_evenly_spaced_points(number_points: int,
                                  include_center=True) -> list:
        """Returns a list of evenly spaced points around the unit circle,
        in cartesian coordinates.

        Parameters
        ----------
        number_points : int
            The number of points around the unit circle.
        include_center : bool, optional
            Whether to include the center point, by default True

        Returns
        -------
        list
            A list of evenly spaced points around the unit circle.
        """
        if include_center:
            offset_points = [np.array([0, 0])]
        else:
            offset_points = []

        offset_points += [np.array([1, k])
                          for k in np.linspace(2 * np.pi, 0, num=number_points,
                                               endpoint=False)[::-1]]

        return [microscope_utilities.polar_to_cartesian(offset)
                for offset in offset_points]

    @staticmethod
    def donut_beam(r: np.ndarray, z: np.ndarray, fwhm_r: float,
                   sigma_z: float, center=np.zeros((1, 3))) -> np.ndarray:
        """Calculates donut beam for a given r, sigma, and center.

        Parameters
        ----------
        r : np.ndarray
            A displacement value at which to calculate beam value.
        z : np.ndarray
            Z values at which to calculate beam value.
        fwhm_r : float
            The spread of the donut in r.
        sigma_z : float
            The spread of the donut in z.
        center : np.ndarray, optional
            Where to center the donut beam, by default (0, 0, 0)

        Returns
        -------
        np.ndarray
            The donut value at the given displacement.
        """

        r_centered = r - center[:, 0:2]  # broadcasting inserts axes in front
        r2 = np.sum(r_centered**2, axis=-1)
        donut_pdf = 4 * np.e * np.log(2) * (r2 / fwhm_r**2) * \
            np.exp(-4 * np.log(2) * (r2 / fwhm_r**2))

        # make normal distribution for z
        z_centered = z - center[:, 2]
        z2 = z_centered**2
        z_norm = (1 / np.sqrt(2 * np.pi)) * np.exp(-z2/(2 * sigma_z**2))

        return donut_pdf * z_norm

    @staticmethod
    def square_beam(r: np.ndarray, center=np.zeros((1, 2))) -> np.ndarray:
        """Calculates a square beam for a given r and center.

        Parameters
        ----------
        r : np.ndarray
            The displacement value at which to calculate beam value.
        center : np.ndarray, optional
            Where to center the beam, by default np.zeros((1, 2))

        Returns
        -------
        np.ndarray
            The beam value at the given displacement.
        """

        r_centered = r - center  # broadcasting inserts axes in front
        return np.sum(r_centered**2, axis=-1)

    @staticmethod
    def mle_two_gaussian(n_0: int, n_photons: int,
                         L_value: float, fwhm: float) -> np.ndarray:
        """The maximum likelihood estimator for two Gaussian beams
        The maximum likelihood estimator for 1D position estimation with two
        Gaussian-shaped excitation profiles as given by equation S40 in
        Balzarotti et al.

        Parameters
        ----------
        n_0 : int
            The number of photons observed at position zero.
        n_photons : int
            The total number of photons observed.
        L_value : float
            The L value used to generate the parameter vector. The diameter of
            the probing pattern in nm.
        fwhm : float
            The full-width half maximum (fwhm) of the donut excitation beam
            pattern used to generate the parameter vector.

        Returns
        -------
        np.ndarray
            The maximum likelihood estimate for the particle position.
        """

        if n_0 == 0 or n_0 == n_photons:
            return 0
        n_1 = n_photons - n_0
        return (fwhm ** 2 / (8 * np.log(2) * L_value)) * np.log(n_0 / n_1)
