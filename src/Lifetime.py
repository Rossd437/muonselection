import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import seaborn as sns

import pylandau

import os
import subprocess

def convolution(hist_a, edges_a, hist_b, edges_b):
    # 1. Convolve the heights (pmfs)
    conv = np.convolve(hist_a, hist_b)

    # 2. Compute bin width (must be same for A and B)
    bin_width = edges_a[1] - edges_a[0]

    # 3. Build new edges: len(conv)+1 edges
    new_edges = np.arange(
        edges_a[0] + edges_b[0],
        edges_a[0] + edges_b[0] + (len(conv) * bin_width) + 1e-9,
        bin_width
    )
    new_edges = new_edges[:len(conv)+1]

    return conv, new_edges

ef ElectronLifetimeFunc(x, A, tau):
    return A*np.exp(-x/tau)


def langau_func(x, mpv, eta, sigma, A):
    """Landau distribution function."""
    # Assuming this is a simplified version for the purpose of the example
    return pylandau.langau(x, mpv, eta, sigma, A)

def chi_squared1(params, function, x, y, uncertainties):
    
    y_pred = function(x, *params)

    top = y-y_pred
    nom = top**2
    dom = uncertainties**2
    chi_array = nom/dom
    chi2 = np.sum(chi_array)

    return chi2

def langau_lifetime(nhits, dqdx, time_drifted, time_bins, dqdx_bins, nhits_bins, wanted_sim, plotting=False):
    method = " [PyLandau (Landau * Gaussian)]"
    fit_points = sum(time_bins <= np.max(time_bins)*.1) + 2
    
    mpvs = np.empty(len(time_bins)-1)
    times =  np.empty(len(time_bins)-1)
 
    mpv_uncertainties =  np.empty(len(time_bins)-1)

    masks = []
    convs = []
    edges = []

    for i in range(len(time_bins)-1):

        mask = (time_drifted >= time_bins[i]) & (time_drifted <= time_bins[i+1])
        masks.append(mask)
        dq_dx = dqdx[mask]
        
        hits = nhits[mask]

        hist_a, edges_a = np.histogram(hits, bins=nhits_bins, density=True)
        hist_b, edges_b = np.histogram(dq_dx, bins=dqdx_bins, density=True)      
        conv, new_edges = convolution(hist_a,  edges_a, hist_b, edges_b)
        conv_bin_centers = (new_edges[:-1] + new_edges[1:])/2
        
        convs.append(conv)
        edges.append(new_edges)

        max_value = conv_bin_centers[np.argmax(conv)]
        
        half_max = max_value/2
        fwhm_range = [max_value-half_max, max_value+half_max]
        
        mask_fit_range = (conv_bin_centers >= fwhm_range[0]) & (conv_bin_centers <= fwhm_range[1])
        
        bin_centers_fit = conv_bin_centers[mask_fit_range]
        h_fit = conv[mask_fit_range]

        mpv, eta, sigma, A, mpv_uncertainty, mpv_guess = langau_fit(h_fit, bin_centers_fit)

        mpvs[i] = mpv
        times[i] = (time_bins[i]+time_bins[i+1])/20
        mpv_uncertainties[i] = mpv_uncertainty
 
    #Get Electron Lifetime
    params, cov = curve_fit(ElectronLifetimeFunc, times[fit_points:-1], mpvs[fit_points:-1], p0 = [np.max(mpvs), 2000], sigma=mpv_uncertainties[fit_points:-1], absolute_sigma=True)

    if plotting:
        #Lifetime plotting
        chi2 = chi_squared1(params, ElectronLifetimeFunc, times[fit_points:], mpvs[fit_points:], mpv_uncertainties[fit_points:])
        x_fit = np.linspace(times[0], times[-1],1000)
        fit_curve = ElectronLifetimeFunc(x_fit, *params)
        
        fig, ax = plt.subplots()

        #Set y lim
        ax.set_ylim(min(mpvs)-2,max(mpvs)+2) 

        #Plot
        sns.scatterplot(x=times[fit_points:], y=mpvs[fit_points:],marker=r"$\circ$", ax=ax, color='black', label='Fitted')
        sns.scatterplot(x=times[:fit_points], y=mpvs[:fit_points],marker=r"$\circ$",ax=ax, color='grey', label='Not Fitted')
        sns.lineplot(x=x_fit, y=fit_curve, color='orange', ax=ax, label=f'$e^{{-}}$ lifetime = {params[1]:.4f} $\pm$ {np.sqrt(cov[1,1]):.4f} [μs] \n $dQ_{{0}}/dx$ = {params[0]:.4} $\pm$ {np.sqrt(cov[0,0]):.4f} [$ke^{{-}}/cm$] \n $\chi^{2}/ndf = {chi2/len(mpvs[fit_points:])}$')
        
        #Axis titles
        ax.set_ylabel(r"MPV of ($\frac{dN}{dx}$ * $\frac{dQ}{dx}$)")
        ax.set_xlabel(r"Time Drifted [$\mu$s]")
        ax.set_title(f"{wanted_sim}" + method)

        # Add error bars using matplotlib
        ax.errorbar(x=times[fit_points:], y=mpvs[fit_points:], yerr=mpv_uncertainties[fit_points:], fmt='none', ecolor='black', elinewidth=1, capsize=3, zorder=2)
        ax.errorbar(x=times[:fit_points], y=mpvs[:fit_points], yerr=mpv_uncertainties[:fit_points], fmt='none', ecolor='grey', elinewidth=1, capsize=3, zorder=2)

        ax.legend(edgecolor='black',  fontsize= 8, loc='upper right')
        file_path = f"/global/cfs/cdirs/dune/users/demaross/Electron_Lifetime/Plots/{wanted_sim}"
        if os.path.exists(file_path):
            print("Exist")
        else:
            subprocess.run(["mkdir", f"{file_path}"])
            print(f"Directory made {file_path}") 

        fig.savefig(f"/global/cfs/cdirs/dune/users/demaross/Electron_Lifetime/Plots/{wanted_sim}/{wanted_sim}_pylandau_dNdQ.png")

	return 
