import numpy as np
import pylandau
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from typing import Callable
import pandas as pd


def convolution(
    hist_a: np.ndarray, edges_a: np.ndarray, hist_b: np.ndarray, edges_b: np.ndarray
) -> tuple:
    """Compute the convolution.

    Get the convolution of the two histograms and the convolution's bin edges.

    Args:
        hist_a: Counts of histogram a
        edges_a: bin edges of histogram a
        hist_b: Counts of histogram b
        edges_b: bin edges of histogram b

    Returns:
        Convolution counts and convolution edges.
    """
    conv = np.convolve(hist_a, hist_b)

    bin_width = edges_a[1] - edges_a[0]

    new_edges = np.arange(
        edges_a[0] + edges_b[0],
        edges_a[0] + edges_b[0] + (len(conv) * bin_width) + 1e-9,
        bin_width,
    )
    new_edges = new_edges[: len(conv) + 1]

    return conv, new_edges


def ElectronLifetimeFunc(x: np.ndarray, A: float, tau: float) -> np.ndarray:
    """The lifetime function.

    Args:
        x: Time drifted of segment
        A: Initial dQ/dx at the anode
        tau: The lifetime

    Returns:
        dQ/dx at a given time drifted and lifetime.
    """
    return A * np.exp(-x / tau)


def langau_func(
    x: np.ndarray, mpv: float, eta: float, sigma: float, A: float
) -> np.ndarray:
    """Landau distribution function.

    Args:
        x: dQ/dx
        mpv: Most probable value
        eta: Scale of the landauhi_
        sigma: Standard deviation of gaussian
        A: Total area

    Returns:
        Landau gaussian convolution distribution
    """
    return pylandau.langau(x, mpv, eta, sigma, A)


def chi_squared(
    params: list,
    func: Callable[[np.ndarray, float, float], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    uncertainties: np.ndarray,
) -> float:
    """Compute the chi squared.

    Get the chi squared for a given function and set of x and y points.

    Args:
        params: Fitted parameters
        func: Function to get y predictions
        x: x data points.
        y: y data points.
        uncertainties: y value uncertainties

    Returns:
        Chi squared
    """
    y_pred = func(x, *params)

    top = y - y_pred
    nom = top**2
    dom = uncertainties**2
    chi_array = nom / dom
    chi2 = np.sum(chi_array)

    return chi2


def langau_fit(hist: np.ndarray, bin_centers: np.ndarray) -> tuple:
    """Landau gaussian convolution fit.

    Fit the landau gaussian convolution to histogram counts and bin centers.

    Args:
        hist: Histogram counts
        bin_centers: Bin centers of histogram

    Returns:
        The MPV, scale, area, MPV uncertainty, and MPV guess of the fit
    """
    warnings.filterwarnings(
        "ignore", message="sigma > 100 * eta can lead to oszillations. Check result."
    )

    std_bin_centers = np.std(bin_centers)
    eta_guess, sigma_guess = std_bin_centers / 6, std_bin_centers / 4

    mpv_guess, A_guess = bin_centers[np.argmax(hist)], np.max(hist)

    initial_guess = [mpv_guess, eta_guess, sigma_guess, A_guess]

    bounds = ([0, 0.1, 0, 0], [np.inf, np.inf, np.inf, np.inf])

    try:
        params, cov = curve_fit(
            langau_func,
            bin_centers,
            hist,
            p0=initial_guess,
            bounds=bounds,
            max_nfev=1000000,
        )
        mpv_uncertainty = np.sqrt(cov[0, 0])
        mpv, eta, sigma, A = params[0], params[1], params[2], params[3]
        mpv_uncertainty = np.sqrt(cov[0, 0])

    except RuntimeError as e:
        print(f"Fitting failed: {e}")
        params = initial_guess
        mpv, eta, sigma, A = params[0], params[1], params[2], params[3]
        mpv_uncertainty = 0

    return mpv, eta, sigma, A, mpv_uncertainty, mpv_guess


def langau_lifetime(
    dNdx: np.ndarray,
    dqdx: np.ndarray,
    time_drifted: np.ndarray,
    time_bins: np.ndarray,
    dqdx_bins: np.ndarray,
    dNdx_bins: np.ndarray,
    wanted_sim: str,
    output_file: str,
    plotting=True,
) -> int:
    """Extract the lifetime.

    Use the dQ/dx * dN/dx convolution to extract the lifetime.

    Args:
        dNdx: Number of hits per unit length of segment.
        dqdx: Charge per unit length of segment.
        time_drifted: Time drifted of segment.
        time_bins: Time bins.
        dqdx_bins: Bins for dq/dx histogram.
        dNdx_bins: Bins of dN/dx histogram.
        wanted_sim: Simulation of muon selection.
        output_file: File to save plot.

    Returns:
        Nothing, but will save plot of the lifetime.
    """
    method = " [PyLandau (Landau * Gaussian)]"
    fit_points = sum(time_bins <= np.max(time_bins) * 0.1) + 2

    mpvs = np.empty(len(time_bins) - 1)
    times = np.empty(len(time_bins) - 1)

    mpv_uncertainties = np.empty(len(time_bins) - 1)

    masks = []
    convs = []
    edges = []

    for i in range(len(time_bins) - 1):
        mask = (time_drifted >= time_bins[i]) & (time_drifted <= time_bins[i + 1])
        masks.append(mask)
        dq_dx = dqdx[mask]

        hits = dNdx[mask]

        hist_a, edges_a = np.histogram(hits, bins=dNdx_bins, density=True)
        hist_b, edges_b = np.histogram(dq_dx, bins=dqdx_bins, density=True)
        conv, new_edges = convolution(hist_a, edges_a, hist_b, edges_b)
        conv_bin_centers = (new_edges[:-1] + new_edges[1:]) / 2

        convs.append(conv)
        edges.append(new_edges)

        max_value = conv_bin_centers[np.argmax(conv)]

        half_max = max_value / 2
        fwhm_range = [max_value - half_max, max_value + half_max]

        mask_fit_range = (conv_bin_centers >= fwhm_range[0]) & (
            conv_bin_centers <= fwhm_range[1]
        )

        bin_centers_fit = conv_bin_centers[mask_fit_range]
        h_fit = conv[mask_fit_range]

        mpv, eta, sigma, A, mpv_uncertainty, mpv_guess = langau_fit(
            h_fit, bin_centers_fit
        )

        mpvs[i] = mpv
        times[i] = (time_bins[i] + time_bins[i + 1]) / 20
        mpv_uncertainties[i] = mpv_uncertainty

    # Get Electron Lifetime
    params, cov = curve_fit(
        ElectronLifetimeFunc,
        times[fit_points:-1],
        mpvs[fit_points:-1],
        p0=[np.max(mpvs), 2000],
        sigma=mpv_uncertainties[fit_points:-1],
        absolute_sigma=True,
    )
    print(
        f"Extracted lifetime of {params[1]} += {round(np.sqrt(cov[1, 1]), 3)} microseconds"
    )

    if plotting:
        # Lifetime plotting
        chi2 = chi_squared(
            params,
            ElectronLifetimeFunc,
            times[fit_points:],
            mpvs[fit_points:],
            mpv_uncertainties[fit_points:],
        )
        x_fit = np.linspace(times[0], times[-1], 1000)
        fit_curve = ElectronLifetimeFunc(x_fit, *params)

        fig, ax = plt.subplots()

        # Set y lim
        ax.set_ylim(min(mpvs) - 2, max(mpvs) + 2)

        # Plot
        sns.scatterplot(
            x=times[fit_points:],
            y=mpvs[fit_points:],
            marker=r"$\circ$",
            ax=ax,
            color="black",
            label="Fitted",
        )
        sns.scatterplot(
            x=times[:fit_points],
            y=mpvs[:fit_points],
            marker=r"$\circ$",
            ax=ax,
            color="grey",
            label="Not Fitted",
        )
        sns.lineplot(
            x=x_fit,
            y=fit_curve,
            color="orange",
            ax=ax,
            label=f"$e^{{-}}$ lifetime = {params[1]:.4f} $\pm$ {np.sqrt(cov[1, 1]):.4f} [μs] \n $dQ_{{0}}/dx$ = {params[0]:.4} $\pm$ {np.sqrt(cov[0, 0]):.4f} [$ke^{{-}}/cm$] \n $\chi^{2}/ndf = {chi2 / len(mpvs[fit_points:])}$",
        )

        # Axis titles
        ax.set_ylabel(r"MPV of ($\frac{dN}{dx}$ * $\frac{dQ}{dx}$)")
        ax.set_xlabel(r"Time Drifted [$\mu$s]")
        ax.set_title(f"{wanted_sim}" + method)

        # Add error bars using matplotlib
        ax.errorbar(
            x=times[fit_points:],
            y=mpvs[fit_points:],
            yerr=mpv_uncertainties[fit_points:],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
            zorder=2,
        )
        ax.errorbar(
            x=times[:fit_points],
            y=mpvs[:fit_points],
            yerr=mpv_uncertainties[:fit_points],
            fmt="none",
            ecolor="grey",
            elinewidth=1,
            capsize=3,
            zorder=2,
        )

        ax.legend(edgecolor="black", fontsize=8, loc="upper right")

        fig.savefig(output_file)

    return params[1], np.sqrt(cov[1, 1])


if __name__ == "__main__":
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    file = sys.argv[1]
    output_file = sys.argv[2]

    segments = pd.read_csv(file)

    dNdx = segments["dN"] / segments["dx"]
    dqdx = segments["dQ"] / segments["dx"]
    time_drifted = segments["t"]

    time_bins = np.linspace(0, 1960, 20)
    dqdx_bins = np.linspace(0, 250, 70)
    dNdx_bins = np.linspace(0, 30, 42)
    wanted_sim = "MiniRun6.5"

    lifetime, error = langau_lifetime(
        dNdx,
        dqdx,
        time_drifted,
        time_bins,
        dqdx_bins,
        dNdx_bins,
        wanted_sim,
        output_file,
        plotting=True,
    )
