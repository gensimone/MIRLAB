#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import freqz

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

# extracts one window of frame_length samples in the file, for FFT and LPC
def plot_fft_and_spectral_envelope(filename, frame_length=2048, lpc_order=16):
    # load file: return array size (y), sample rate (sr=None mean original SR)
    y, sr = librosa.load(filename, sr=None)

    # take the center sample of the signal (len(y)//2)
    center = len(y) // 2
    # clip 0 and frame the center from the middle of the signal
    start = max(0, center - frame_length // 2)
    # moves forward from the starting index
    end = min(len(y), start + frame_length)
    # clip the end of the array
    frame = y[start:end]

    # apply a window to the frame to reduce spectral leakage before FFT
    windowed = frame * np.hanning(len(frame))

    # ==========================================================================
    # FFT computation
    # ==========================================================================

    # increase FFT size by zero-padding to interpolate the FFT for smoother plot
    n_fft = frame_length * 4
    # Compute the FFT of the windowed frame, with zero-padding
    spectrum = np.abs(np.fft.fft(windowed, n=n_fft))[:n_fft // 2]
    # conversion to decibel
    spectrum_db = 20 * np.log10(spectrum + 1e-12)
    # normalization (take the max and then normalize)
    spectrum_db -= np.max(spectrum_db)
    # array of frequency values, between 0 and SR/2
    freqs = np.linspace(0, sr/2, n_fft // 2)

    # ==========================================================================
    # LPC computation
    # ==========================================================================

    # compute LPC coeff. on the windowed frame to estimate its spectral envelope
    a = librosa.lpc(windowed, order=lpc_order)
    # compute the frequency response of the LPC filter
    w, h = freqz([1], a, worN=n_fft // 2, fs=sr)
    # conversion to decibel
    envelope_db = 20 * np.log10(np.abs(h) + 1e-12)
    # normalization (take the max and then normalize)
    envelope_db -= np.max(envelope_db)

    # ==========================================================================
    # Plotting
    # ==========================================================================

    # Plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # FFT
    axs[0].plot(freqs, spectrum_db, color="black", linewidth=1)
    axs[0].set_ylabel("FFT dB", color='black')
    axs[0].set_title(" ", color='black')
    axs[0].set_xlim(0, sr/2)
    axs[0].set_ylim(-100, 0)

    # LPC
    axs[1].plot(w, envelope_db, color="black", linewidth=1)
    axs[1].set_xlabel("Hz", color='black')
    axs[1].set_ylabel("LPC dB", color='black')
    axs[1].set_title(" ", color='black')
    axs[1].set_xlim(0, sr/2)
    axs[1].set_ylim(-100, 0)

    # plot style
    for ax in axs:
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.tick_params(colors='black')
        ax.grid(False)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# COMMAND-LINE ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        with open(".selected_audio.txt", "r") as f:
            filename = f.readline().strip()
    except FileNotFoundError:
        print("Error: .selected_audio.txt not found")
        sys.exit(1)

    # optional parameters
    frame_length = 2048
    lpc_order = 16

    # parse optional command line arguments
    if len(sys.argv) > 1:
        try:
            frame_length = int(sys.argv[1])
        except ValueError:
            print("Error: frame_length should be an int")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            lpc_order = int(sys.argv[2])
        except ValueError:
            print("Error: lpc_order should be an int")
            sys.exit(1)

    plot_fft_and_spectral_envelope(filename, frame_length, lpc_order)
