import pytest
import sys
import pylandau
import pandas as pd
import numpy as np

sys.path.append("../src/")
from Lifetime import (
    convolution,
    chi_squared,
    ElectronLifetimeFunc,
    langau_func,
    langau_fit,
    langau_lifetime,
)


@pytest.fixture
def segments_df():
    file = "../results/csvs/MiniRun6.5_segments.csv"
    return pd.read_csv(file)


def test_convolution(segments_df):
    hist_a, edges_a = np.histogram(
        segments_df["dQ"] / segments_df["dx"], np.linspace(0, 250, 70)
    )
    bin_centers = (edges_a[1:] + edges_a[:-1]) / 2

    hist_b, edges_b = np.histogram(
        segments_df["dN"] / segments_df["dx"], np.linspace(0, 30, 42)
    )

    conv, new_edges = convolution(hist_a, edges_a, hist_b, edges_b)
    conv_bin_centers = (new_edges[1:] + new_edges[:-1]) / 2
    assert conv_bin_centers[np.argmax(conv)] <= bin_centers[np.argmax(bin_centers)]


@pytest.mark.parametrize(
    "x, A, tau",
    [(np.linspace(0, 196, 100), 50, 2200), (np.linspace(0, 196, 50), 20, 100)],
)
def test_ElectronLifetimeFunc(x, A, tau):
    assert np.all(ElectronLifetimeFunc(x, A, tau) == A * np.exp(-x / tau))


@pytest.mark.parametrize(
    "x, mpv, eta, sigma, A", [(np.linspace(0, 200, 75), 20, 1, 3, 300)]
)
def test_langau_func(x, mpv, eta, sigma, A):
    assert np.all(
        langau_func(x, mpv, eta, sigma, A) == pylandau.langau(x, mpv, eta, sigma, A)
    )


@pytest.mark.parametrize(
    "params, x, y, uncertainties",
    [([20, 2200], np.linspace(0, 196, 40), np.linspace(20, 50, 40), np.ones(40))],
)
def test_chi_squared(params, x, y, uncertainties):
    assert chi_squared(params, ElectronLifetimeFunc, x, y, uncertainties) >= 0


def test_langau_fit(segments_df):
    hist, bin_edges = np.histogram(
        segments_df["dQ"] / segments_df["dx"], np.linspace(0, 250, 70)
    )
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    mpv, eta, sigma, A, mpv_uncertainty, mpv_guess = langau_fit(hist, bin_centers)

    assert mpv_guess == pytest.approx(bin_centers[np.argmax(hist)], 0.1)


def test_langau_lifetime(segments_df):
    dNdx = segments_df["dN"] / segments_df["dx"]
    dqdx = segments_df["dQ"] / segments_df["dx"]
    time_drifted = segments_df["t"]

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
        output_file="nothing",
        plotting=False,
    )

    assert lifetime > 0
