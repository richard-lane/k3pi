"""
Implementation for the efficiency reweighter

"""
import numpy as np
from hep_ml.reweight import GBReweighter, BinsReweighter

from lib_efficiency import time_fitter


class TrainingError(Exception):
    """
    Useful because we might be multiprocessing; raise this if anything
    went wrong in a child process

    """

    def __init__(self):
        """
        Error training the BDT or something

        """
        super().__init__(
            "Error training the BDT; see above for child process tracebacks"
        )


class TimeFitReweighter:
    """
    Do the reweighting by performing a fit to decay times

    Does the reweighting AmpGen -> MC (i.e. the weights here apply the
    efficiency, they don't correct for it) - this is because this seems
    to be more stable

    Get the weights by assuming the true decay time distribution is
    e^-t

    """

    def __init__(self):
        """
        List of attributes

        """
        self.fit_vals = None

    def fit(self, _, times: np.ndarray):
        """
        Perform the fit

        The dummy var is for API consistency with BinsReweighter.fit

        """
        # These initial parameters seem to do the right thing, mostly
        # We can't store the fitter as an attribute since it's unserialisable
        fitter = time_fitter.fit(times, (0.21, 1.0, 2.0, 1.0, 2.0, 1.0))

        # We can however store the fit parameters
        self.fit_vals = np.array(fitter.values)

    def _fitted_pdf(self, times: np.ndarray) -> np.ndarray:
        """
        Return (normalised) pdf values at each time passed in

        """
        assert (
            self.fit_vals is not None
        ), "need to call .fit() before we have a fitted pdf"

        return time_fitter.normalised_pdf(times, *self.fit_vals)[1]

    def predict_weights(self, times: np.ndarray) -> np.ndarray:
        """
        Weights to apply the efficiency

        Returns 0 for times below the fitted minimum

        """
        return self._fitted_pdf(times) / np.exp(-times)


class TimeWeighter:
    """
    The time reweighter

    Or via a histogram division

    Basically this class just holds either a BinsReweighter or TimeFitReweighter
    object, and uses this to perform the time weighting

    """

    def __init__(self, min_t: float, fit: bool, n_bins: int, n_neighs: float):
        """
        Tell us whether we're doing a fit to the decay times

        :param min_t: time below which to set all weights to 0 anyway
        :param fit: whether we want to reweight using the decay time fit
                    or by using a BinsReweighter

        """
        self.min_t = min_t
        self.fitter = (
            TimeFitReweighter()
            if fit
            else BinsReweighter(n_bins=n_bins, n_neighs=n_neighs)
        )

    def fit(self, mc_times: np.ndarray, ampgen_times: np.ndarray):
        """
        ampgen times arg is unused if we're doing a fit

        """
        # Reweight AmpGen to MC to avoid getting huge weights
        # If we're doing a histogram division, only use times above the minimum
        if isinstance(self.fitter, BinsReweighter):
            ag_t = ampgen_times[ampgen_times > self.min_t]
            mc_t = mc_times[mc_times > self.min_t]
            print(
                f"{self.fitter.n_bins} bins\n"
                f"{len(ag_t)}, {len(mc_t)} times above minimum.\n"
                f"Avg of {len(ag_t) / self.fitter.n_bins}, "
                f"{len(mc_t) / self.fitter.n_bins} per bin"
            )
            self.fitter.fit(ag_t, mc_t)
        # If we're fitting fit all the times; not just the ones above the min time
        else:
            print(
                f"Performing fit to {len(mc_times):,} times from "
                f"{np.min(mc_times):.4f} to {np.max(mc_times):.4f}"
            )
            self.fitter.fit(None, mc_times)

    def apply_efficiency(self, times):
        """
        Predict weights to apply the efficiency

        """
        above_min = times > self.min_t
        retval = np.zeros_like(times)
        retval[above_min] = self.fitter.predict_weights(times[above_min])

        return retval

    def correct_efficiency(self, times):
        """
        Weights to correct for the efficiency

        """
        above_min = times > self.min_t
        retval = np.zeros_like(times)
        retval[above_min] = 1 / self.fitter.predict_weights(times[above_min])

        return retval


class EfficiencyWeighter:
    """
    Holds a time reweighter that either does a histogram division or a fit to the decay times

    Holds also a BDT reweighter that deals with the phsp efficiency, but uses the time reweighter
    to keep the correlations between time and phase space

    """

    def __init__(
        self,
        target: np.ndarray,
        original: np.ndarray,
        fit: bool,
        min_t: float,
        n_bins: int = 20000,
        n_neighs: float = 10.0,
        **train_kwargs,
    ):
        """
        Perform time weighting and train the BDT

        First finds the decay time efficiency e(t), then trains BDT to reweight mc -> AmpGen * e(t)

        :param target: shape (N, 6) array of phsp points; 6th element of each should be decay time
        :param original: shape (N, 6) array of phsp points; 6th element of each should be decay time
        :param fit: whether to find the decay time efficiency by performing a fit. Does a histogram
                    division if False.
        :param min_t: minimum time, below which weights are set to 0.
                      The histogram division only considers times above this, but the fit fits to
                      all times.
        :param n_bins: if not fitting, the number of bins to use for the hist division
        :param n_neighs: if not fitting, the number of neighbours to account for
                         when doing the histogram division
        :param train_kwargs: kwargs passed to GBReweighter when training

        """
        self._time_weighter = TimeWeighter(min_t, fit, n_bins, n_neighs)
        self._time_weighter.fit(original[:, 5], target[:, 5])

        self._phsp_weighter = GBReweighter(**train_kwargs)

        # Overall we will eight original -> target
        # but here weight the target such that it looks like original to prevent huge weights
        self._phsp_weighter.fit(
            original=original,
            target=target,
            target_weight=self._time_weighter.apply_efficiency(target[:, 5]),
        )

    def time_weights(self, times: np.ndarray) -> np.ndarray:
        """
        Find weights to take mc -> AmpGen

        """
        return self._time_weighter.correct_efficiency(times)

    def phsp_weights(self, phsp_points: np.ndarray) -> np.ndarray:
        """
        Find the weights needed to reweight phsp part of mc -> AmpGen

        """
        return self._phsp_weighter.predict_weights(phsp_points)

    def weights(self, points):
        """
        Normalised to have a mean of 1.0
        """
        retval = self.phsp_weights(points) * self.time_weights(points[:, 5])
        return retval / np.mean(retval)
