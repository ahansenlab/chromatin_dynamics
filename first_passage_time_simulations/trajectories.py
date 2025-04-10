import numpy as np
import warnings
import functools
from abc import ABC, abstractmethod
from types import FunctionType
from tqdm import tqdm


def next_power_of_two(x: int) -> int:
    """Returns the next highest power of two.

    Useful for speeding up fft algorithm.

    Parameters
    ----------
    x : int
        The input value.

    Returns
    -------
    int
        The next highest power of two.
    """
    # https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
    return 1 << (x-1).bit_length()


class DiffusionProcess(ABC):
    """A general class used to simulate a diffusion process.
    """

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    @abstractmethod
    def generate_displacements(self, dt: float, n_dims: int,
                               n_particles: int) -> np.ndarray:
        """Generates displacements for one diffusion time step.

        Returns a random displacment vector for one time step `dt` of the
        simulation for each spatial dimension `n_dims` and particle
        `n_particles.

        Parameters
        ----------
        dt : float
            The time step of the simulation in standard units (s).
        n_dims : int
            The number of spatial dimensions to simulate diffusion in.
        n_particles : int
            The number of particles to return displacements for.

        Returns
        -------
        np.ndarray
            A random array of displacements of size `(n_dims, n_particles)`
        """
        raise NotImplementedError

    @abstractmethod
    def generate_noise_from_mask(self, dt: float, n_dims: int,
                                 mask: np.ndarray) -> np.ndarray:
        """Generates a noise matrix for the supplied mask.

        Returns a matrix contaning noise samples corresponding to the diffusion
        process for contiguous blocks of 1 in the mask.

        Parameters
        ----------
        dt : float
            The time step corresponding to the time dimension of the mask.
        n_dims : int
            The number of spatial dimensions to simulate diffusion in.
        mask : np.ndarray
            A binary mask. Contiguous blocks of 1 along the time dimension will
            yield gaussian noise samples, while 0s will yield 0.

        Returns
        -------
        np.ndarray
            A gaussian noise matrix populated according to the supplied binary
            mask.
        """
        raise NotImplementedError

    def set_seed(self, seed: None | int) -> None:
        """Sets the seed used for random calculations in the class.

        Parameters
        ----------
        seed : int
            Seed to be passed to numpy.random.default_rng().
        """
        # TODO: Extend set seed so it affects calculations done by
        # BindingModel, BindingSpecies, Geometry, as well as RxnModel
        self._rng = np.random.default_rng(seed)


class FractionalBrownianMotion(DiffusionProcess):
    """A fractional Brownian motion class.

    This class implements a fractional Brownian motion diffusion process with
    no drift.

    Parameters
    ----------
    anomalous_diffusion_coefficient : float
        The anomalous diffusion coefficient for the fractional Brownian motion
        in units of um^2 s^(-2H). Note, that for H = 0.5, this value is 2 * D,
        the diffusion coefficient.

    H : float
        The Hurst parameter for the fractional Brownian motion. H takes values
        between 0 and 1. If 0 < H < 0.5, the fBm is subdiffusive. If H = 0.5,
        the fBm reduces to Brownian motion. If 0.5 < H < 1, the fBm is
        superdiffusive.

    Attributes
    ----------
    anomalous_diffusion_coefficient : float
        The anomalous diffusion coefficient for the fractional Brownian motion
        in units of um^2 s^(-2H). Note, that for H = 0.5, this value is 2 * D,
        the diffusion coefficient.

    k_r : float
        An alias for the anomalous diffusion coefficient.

    H : float
        The Hurst parameter for the fractional Brownian motion. H takes values
        between 0 and 1. If 0 < H < 0.5, the fBm is subdiffusive. If H = 0.5,
        the fBm reduces to Brownian motion. If 0.5 < H < 1, the fBm is
        superdiffusive.

    hurst : float
        An alias for the Hurst parameter.

    alpha : float
        The exponent of MSD. Equivalently, 2H where H is the Hurst parameter.

    computation_method : FunctionType
        The method used to compute the fractional Brownian motion. It must be
        one of 'davies-harte' or 'hosking'. See the davies_harte and hosking
        methods for more details.
    """

    def __init__(self, anomalous_diffusion_coefficient: float,
                 H: float) -> None:
        super().__init__()
        self.anomalous_diffusion_coefficient = anomalous_diffusion_coefficient
        self.H = H
        self.computation_method = 'davies-harte'
        self._diffusion_type = "brownian"

    @property
    def diffusion_type(self) -> str:
        return self._diffusion_type

    @property
    def diffusion_params(self) -> dict:
        return {"anomalous_diffusion_coefficient":
                self.anomalous_diffusion_coefficient, "H": self.H}

    @property
    def H(self) -> float:
        return self._H

    @H.setter
    def H(self, new_H: float):
        if new_H >= 1 or new_H <= 0:
            raise ValueError(f"The Hurst parameter must be between 0 and 1 not"
                             f" {new_H}.")
        self._H = new_H

    @property
    def k_r(self) -> float:
        return self.anomalous_diffusion_coefficient

    @k_r.setter
    def k_r(self, anomalous_diffusion_coefficient: float):
        self.anomalous_diffusion_coefficient = anomalous_diffusion_coefficient

    @property
    def hurst(self) -> float:
        return self.H

    @hurst.setter
    def hurst(self, H: float):
        self.H = H

    @property
    def alpha(self) -> float:
        return 2 * self.H

    @alpha.setter
    def alpha(self, new_alpha: float):
        self.H = new_alpha / 2

    @property
    def computation_method(self) -> FunctionType:
        return self._computation_method

    @computation_method.setter
    def computation_method(self, method: str):
        if method == "davies-harte":
            self.method_name = 'davies-harte'
            self._computation_method = self.davies_harte

        elif method == "hosking":
            self.method_name = 'hosking'
            self._computation_method = self.hosking

        else:
            raise ValueError(f"Computation method must be one of "
                             f"'davies-harte' or 'hosking' not {method}")

    def generate_displacements(self, dt: float, n_dims: int,
                               n_particles: int) -> np.ndarray:
        super().generate_displacements(dt, n_dims, n_particles)

    def generate_noise_from_mask(self, dt: float, n_dims: int,
                                 mask: np.ndarray) -> np.ndarray:
        """Generates a fractional gaussian noise matrix for the supplied mask.

        Parameters
        ----------
        dt : float
            The time step corresponding to the time dimension of the mask.
        n_dims : int
            The number of spatial dimensions to simulate diffusion in.
        mask : np.ndarray
            A binary mask. Contiguous blocks of 1 along the time dimension will
            yield fractional gaussian noise samples, while 0s will yield 0.

        Returns
        -------
        np.ndarray
            A fractional gaussian noise matrix populated according to the
            supplied binary mask.
        """
        # add extra dimensions for spatial dimensions
        mask = np.repeat(mask[:, :, np.newaxis], n_dims, axis=2)

        # swap them to get canonical form for matrix
        mask = np.transpose(mask, (0, 2, 1))

        # get version of mask with different values for each contiguous
        # positive block along time dimension. Need to add a row of zeros for
        # dimension
        switch_mask = np.diff(
            np.concatenate([np.zeros((1, n_dims, mask.shape[-1])), mask]),
            axis=0
        )

        # don't count switches back to 0
        switch_mask[switch_mask < 0] = 0

        indexed_mask = mask * np.cumsum(switch_mask, axis=0)
        # max_switches = np.amax(indexed_mask).astype(int)

        output = np.zeros(mask.shape)

        for k in tqdm(range(mask.shape[-1])):
            # get lengths of each subsequence
            unique, counts = np.unique(np.concatenate(
                [[0], indexed_mask[:, 0, k]]), return_counts=True)
            switches = unique.astype(int)[-1]

            if switches == 0:
                continue

            counts = counts.tolist()
            # iterate over switches
            for i in range(1, switches + 1):

                fbm_sample = self.computation_method(self.covariance_sequence,
                                                     counts[i],
                                                     (n_dims,))
                output[indexed_mask[:, 0, k] == i, :, k] = fbm_sample

        # # TODO: check that this is correct
        # # ensure noise is 0 centered
        # output -= output[0, :, :]

        # scale by standard deviation
        return np.sqrt(self.calculate_variance(dt)) * output

    def calculate_variance(self, dt: float) -> float:
        """Calculate the variance for the fBm time step.

        Calculates the variance for one spatial dimension for a given time
        step size.

        Parameters
        ----------
        dt : float
            The time step of the simulation in standard units (s).

        Returns
        -------
        float
            The variance of the fBm with given anomalous diffusion coefficient
            and time step dt.
        """
        # variance for fractional brownian motion
        return self.k_r * (dt)**self.alpha

    def covariance_sequence(self, n_steps: int) -> np.ndarray:
        """The covariance sequence for the specified fractional gaussian noise.

        Parameters
        ----------
        n_steps : int
            The number of steps for which to calculate the covariance.

        Returns
        -------
        np.ndarray
            The covariance sequence for the fractional Gaussian noise with
            length N and Hurst paramter H.
        """
        return self.fractional_gaussian_noise_covariance(n_steps, self.H)

    @staticmethod
    @functools.cache
    def fractional_gaussian_noise_covariance(n_steps: int,
                                             H: float) -> np.ndarray:
        """The covariance function for fractional Gaussian noise.

        Parameters
        ----------
        n_steps : int
            The number of steps for which to calculate the covariance.
        H : float
            The Hurst parameter for the fractional Gaussian noise

        Returns
        -------
        np.ndarray
            The covariance sequence for the fractional Gaussian noise with
            Hurst paramter H.
        """
        time_range = np.arange(0, n_steps)
        covariance_sequence = 0.5 * (np.abs(time_range - 1) ** (2 * H) -
                                     2 * np.abs(time_range) ** (2 * H) +
                                     np.abs(time_range + 1) ** (2 * H))
        return covariance_sequence

    @staticmethod
    def davies_harte(covariance_function: FunctionType,
                     n_steps: int,
                     size: tuple,
                     gaussian_noise: np.ndarray | None = None,
                     rng: np.random.Generator | None = None) -> np.ndarray:
        """Davies-Harte algorithm for sampling from gaussian process.

        Parameters
        ----------
        covariance_function : FunctionType
            A function that can be used to compute a covariance vector for the
            desired Gaussian process.
        n_steps: int
            The number of time samples to compute.
        size : tuple
            The shape of the output excluding the first dimension.
        gaussian_noise : np.ndarray | None, optional
            A vector of standard Gaussian noise samples equal in length to the
            autocorrelation sequence.
        rng : np.random.Generator | None, optional
            The rng object used to perform the sampling. If None, an rng object
            will be generated, by default None

        Returns
        -------
        np.ndarray
            Samples from the Gaussian process defined by the autocorrelation
            vector.

        Raises
        ------
        ValueError
            If the `gaussian_noise` vector does not have a size compatible with
            the supplied `autocorrelation_sequence` and `n_dimensions`.

        Notes
        -----
        The idea of this method is to find the "square root" of the
        autocovariance matrix by embedding it in a larger circulant matrix,
        and making use of the FFT algorithm. As a result it is O(nlog(n)).
        See the following references:

        [1] Dieker, T. Simulation of fractional Brownian motion.
        [2] Davies, R. B. & Harte, D. S. Tests for Hurst effect. Biometrika 74,
        95–101 (1987).
        [3] Wood, A. T. A. & Chan, G. Simulation of Stationary Gaussian
        Processes in [0, 1] d. Journal of Computational and Graphical
        Statistics 3, 409–432 (1994).
        [4] Craigmile, P. F. Simulating a class of stationary Gaussian
        processes using the Davies–Harte algorithm, with application to long
        memory processes. Journal of Time Series Analysis 24, 505–511 (2003).
        [5] Rasmussen, C. E. & Williams, C. K. I. Gaussian Processes for
        Machine Learning. (The MIT Press, 2005).
        doi:10.7551/mitpress/3206.001.0001.
        """

        eigenvalue = FractionalBrownianMotion._davies_harte_eigenvalue(
            covariance_function, n_steps)

        N = next_power_of_two(n_steps)

        if gaussian_noise is None:
            if rng is None:
                rng = np.random.default_rng()
            unit_gaussian = rng.standard_normal(
                (2 * N, *size)
                )
        else:
            # currently only equal sized vectors are accepted
            unit_gaussian = gaussian_noise
            if not unit_gaussian.shape == (N, *size):
                required_shape = (2 * N, *size)
                raise ValueError(f"Incompatible sizes for `gaussian_noise`, "
                                 f"`autocorrelation_sequence`, and "
                                 f"`size`. `gaussian_noise` must have "
                                 f"shape {required_shape}.")

        W_j = np.zeros(unit_gaussian.shape, dtype=complex)
        W_j[0, ...] = np.sqrt(eigenvalue[0] / (2 * N)) * unit_gaussian[0, ...]
        # double transpose for broadcasting
        W_j[1:N, ...] = (np.sqrt(eigenvalue[1:N] / (4 * N)) *
                         (unit_gaussian[1:-1:2, ...] + 1j *
                          unit_gaussian[2::2, ...]).T).T
        W_j[N, ...] = np.sqrt(eigenvalue[N] / (2 * N)) * unit_gaussian[-1, ...]
        W_j[N + 1:, ...] = np.conjugate(W_j[N-1:0:-1, ...])

        return np.fft.fft(W_j, axis=0).real[:n_steps, ...]

    @staticmethod
    @functools.cache
    def _davies_harte_eigenvalue(covariance_function: FunctionType,
                                 N: int,) -> np.ndarray:
        """Helper function to compute fft for Davies-Harte implementation.

        Parameters
        ----------
        covariance_function : FunctionType
            A function that can be used to compute a covariance vector for the
            desired Gaussian process.
        N: int
            The number of time samples to compute.

        Returns
        -------
        np.ndarray
            Eigenvalue vector for use in Davies-Harte algorithm.
        """
        # optimize for fft
        correlation_sequence = covariance_function(next_power_of_two(N))
        # why fft and not ifft here?
        eigenvalue = np.fft.fft(
            np.concatenate((correlation_sequence, [0],
                            correlation_sequence[:0:-1]))).real

        if not np.all(eigenvalue.real >= 0):
            warnings.warn("The embedding of the autocorrelation matrix is not"
                          " positive semidefinite. In this case, this method "
                          "is only approximate. Try increasing N.")

            # See Wood and Chan chapter 4 for information about this.
            eigenvalue_trace = np.sum(eigenvalue)
            eigenvalue[eigenvalue < 0] = 0
            eigenvalue *= (eigenvalue_trace / np.sum(eigenvalue)) ** 2
        return eigenvalue

    @staticmethod
    def hosking(covariance_function: FunctionType,
                n_steps: int,
                size: tuple,
                gaussian_noise: np.ndarray | None = None,
                rng: np.random.Generator | None = None) -> np.ndarray:
        """Hosking algorithm for sampling from Gaussian process.

        Parameters
        ----------
        covariance_function : FunctionType
            A function that can be used to compute a covariance vector for the
            desired Gaussian process.
        n_steps: int
            The number of time samples to compute.
        size : tuple
            The shape of the output excluding the first dimension.
        gaussian_noise : np.ndarray | None, optional
            A vector of standard Gaussian noise samples equal in length to the
            autocorrelation sequence.
        rng : np.random.Generator | None, optional
            The rng object used to perform the sampling. If None, an rng object
            will be generated, by default None

        Returns
        -------
        np.ndarray
            Samples from the Gaussian process defined by the autocorrelation
            vector.

        Raises
        ------
        ValueError
            If the `gaussian_noise` vector does not have a size compatible with
            the supplied `autocorrelation_sequence` and `n_dimensions`.

        Notes
        -----
        Recursively generates samples by inverting the covariance matrix of the
        process. In general, Davies-Harte should be preferred as it is much
        faster (O(nlog(n)) vs O(n^2)), but Hosking can provide exact solutions
        in scenarios where Davies-Harte fails.

        [1] Dieker, T. Simulation of fractional Brownian motion.
        [2] Oksana Banna, Yuliya Mishura, Kostiantyn Ralchenko & Sergiy
        Shklyar. Fractional Brownian Motion.
        [3] Hosking, J. R. M. Modeling Persistence In Hydrological Time Series
        Using Fractional Differencing. Water Resources Research 20, 1898–1908
        (1984).
        """

        correlation_sequence = covariance_function(n_steps)

        if gaussian_noise is None:
            if rng is None:
                rng = np.random.default_rng()
            unit_gaussian = rng.standard_normal(
                (n_steps, *size)
            )
        else:
            unit_gaussian = gaussian_noise
            if not unit_gaussian.shape == (n_steps, *size):
                raise ValueError("Incompatible size for `gaussian_noise`. "
                                 "`gaussian_noise` must have first dimension "
                                 "`n_steps` and remaining dimensions "
                                 "according to `size`.")

        gaussian_process = np.zeros(unit_gaussian.shape)
        gaussian_process[0, ...] = unit_gaussian[0, ...]

        # initialize parameters
        mu = correlation_sequence[1] * gaussian_process[0, ...]
        sigma_sq = 1 - correlation_sequence[1]**2

        d = np.zeros(correlation_sequence.shape)
        d[0] = correlation_sequence[1]

        for i, gamma_val in enumerate(correlation_sequence[2:]):
            # calculate new mean
            mu = d[:i + 1] @ np.moveaxis(gaussian_process[i::-1, ...], 0, -2)

            # compute parameters
            tau = np.dot(correlation_sequence[1:i + 2], d[i::-1])
            phi = (gamma_val - tau) / sigma_sq

            # calculate new variance
            sigma_sq -= (gamma_val - tau) ** 2 / sigma_sq

            # calculate new d vector
            d[:i + 1] -= phi * d[i::-1]
            d[i + 1] = phi

            # x_n is rescaled normal variable
            gaussian_process[i + 1, ...] = (np.sqrt(sigma_sq) *
                                            unit_gaussian[i + 1, ...] + mu)

        return gaussian_process
