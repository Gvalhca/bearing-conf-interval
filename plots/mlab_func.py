import numpy as np
import scipy.fft


def _nextpow2(x):
    return int(np.ceil(np.log2(abs(x))))


def spect_gen(y=None, Fs=None):
    #   [Y,f]=spect_gen(y,Fs)
    #   this function pruduce signal spectrum
    #   y  : input signal y(t)
    #   Fs : Sampling frequency of y(t);
    #   f  : output frequency vector
    #   Y  : output spectrum vector of the time signal

    #   Note: the fft is used!

    T = 1 / Fs
    L = len(y)
    NFFT = 2 ** _nextpow2(L)
    Y = scipy.fft.fft(y, NFFT) / L
    slice_end = int(NFFT / 2 + 1)
    Y = 2 * np.abs(Y[0:slice_end])
    f = Fs / 2 * np.linspace(0, 1, slice_end)
    f = np.transpose(f)
    Y = np.transpose(Y)
    return Y, f
