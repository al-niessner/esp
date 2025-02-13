"""
various plotting routines to serve multiple tasks
"""

import numpy as np
import matplotlib.pyplot as plt
import io
from scipy.stats import skew, kurtosis
from numpy.fft import fft, fftfreq


def save_plot(plotfn):
    # extract plot data for states.py
    fig, _ = plotfn()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return buf.getvalue()


def plot_residual_fft(
    selftype,
    fltr,
    p,
    raw_residual,
    rel_residuals,
    subt,
    nf_timeseries,
    nf_timeseries_raw,
    tdur_freq=None,
):

    # create plot for residual statistics
    fig, ax = plt.subplots(3, figsize=(10, 10))
    binspace = np.linspace(-0.02, 0.02, 201)
    raw_label = (
        f"Mean: {np.mean(raw_residual,):.4f} \n"
        f"Stdev: {np.std(raw_residual):.4f} \n"
        f"Skew: {skew(raw_residual):.4f} \n"
        f"Kurtosis: {kurtosis(raw_residual):.4f}\n"
        f"Photon Noise: {nf_timeseries_raw:.2f}"
    )
    ax[0].hist(
        raw_residual,
        bins=binspace,
        label=raw_label,
        color=plt.cm.jet(0.25),
        alpha=0.5,
    )
    detrend_label = (
        f"Mean: {np.mean(rel_residuals):.4f} \n"
        f"Stdev: {np.std(rel_residuals):.4f} \n"
        f"Skew: {skew(rel_residuals):.4f} \n"
        f"Kurtosis: {kurtosis(rel_residuals):.4f}\n"
        f"Photon Noise: {nf_timeseries:.2f}"
    )
    ax[0].hist(
        rel_residuals,
        bins=binspace,
        label=detrend_label,
        color=plt.cm.jet(0.75),
        alpha=0.5,
    )
    ax[0].set_xlabel('Relative Flux Residuals')
    ax[0].legend(loc='best')
    ax[1].scatter(
        subt,
        raw_residual,
        marker='.',
        label=f"Raw ({np.std(raw_residual, 0) * 100:.2f} %)",
        color=plt.cm.jet(0.25),
        alpha=0.25,
    )
    ax[1].scatter(
        subt,
        rel_residuals,
        marker='.',
        label=f"Detrended ({np.std(rel_residuals, 0) * 100:.2f} %)",
        color=plt.cm.jet(0.75),
        alpha=0.25,
    )
    ax[1].legend(loc='best')
    ax[1].set_xlabel('Time [BJD]')
    ax[0].set_title(f'Residual Statistics: {p} {selftype} {fltr}')
    ax[1].set_ylabel("Relative Flux")

    # compute fourier transform of raw_residual
    N = len(raw_residual)
    fft_raw = fft(raw_residual)
    fft_res = fft(rel_residuals)
    xf = fftfreq(len(raw_residual), d=np.diff(subt).mean() * 24 * 60 * 60)[
        : N // 2
    ]
    # fftraw = 2.0/N * np.abs(fft_raw[0:N//2])
    # future: square + integrate under the curve and normalize such that it equals time series variance
    ax[2].loglog(
        xf,
        2.0 / N * np.abs(fft_raw[0 : N // 2]),
        alpha=0.5,
        label='Raw',
        color=plt.cm.jet(0.25),
    )
    ax[2].loglog(
        xf,
        2.0 / N * np.abs(fft_res[0 : N // 2]),
        alpha=0.5,
        label='Detrended',
        color=plt.cm.jet(0.75),
    )
    if tdur_freq:
        ax[2].axvline(tdur_freq, ls='--', color='black', alpha=0.5)
    ax[2].set_ylabel('Power')
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].legend()
    ax[2].grid(True, ls='--')
    plt.tight_layout()

    # save plot to state vector
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return buf.getvalue()
