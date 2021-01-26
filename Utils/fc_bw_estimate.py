
# -*- coding:utf-8 -*-

from numpy.fft import fft
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import psd
import pandas as pd
import config as cfg


def fc_bw_estimate(iq, fs):
    """
    :param iq: original signal
    :param fs: sample rate
    :return:
    """

    fs = int(fs)
    fs -= fs % 2
    # n_fft = fs // cfg.FS_NFFT
    n_fft = cfg.N_FFT

    X = np.abs(fft(iq, n_fft))
    X_2 = X**2 / n_fft

    power_spectrum = 10 * np.log10(X_2[:n_fft // 2])

    # plt.plot(power_np.abs(fft(iq, n_fft))spectrum)
    # plt.show()

    # 载频估计
    argmax_power_spectrum = np.argmax(power_spectrum)
    fc = argmax_power_spectrum * fs / n_fft

    # 中值滤波
    power_spectrum_filt = signal.medfilt(power_spectrum, kernel_size=3)

    # 带宽边界值估计
    kl = kr = argmax_power_spectrum
    while (power_spectrum_filt[kl] > power_spectrum_filt[argmax_power_spectrum] - 3) and (kl >= 0):
        kl -= 1
    while (power_spectrum_filt[kr] > power_spectrum_filt[argmax_power_spectrum] - 3) and (kr < len(power_spectrum_filt)):
        kr += 1
    deltawidth = kr - kl
    bw = deltawidth * fs / n_fft

    return [fc, bw]


if __name__ == '__main__':

    # fs = 1024
    # num_fft = 1024
    #
    # t = np.arange(0, 1, 1 / fs)
    # f0 = 100
    # f1 = 200
    # # x = np.cos(2*np.pi*f0*t) + 3*np.cos(2*np.pi*f1*t) + np.random.randn(t.size)
    # x = np.cos(2 * np.pi * f0 * t) + np.random.randn(t.size)
    #
    # fc_bw = fc_bw_estimate(x, fs)
    # fc = fc_bw[0]
    # bw = fc_bw[1]
    # print(fc, bw)

    fs = 977419
    DIR = 'D:/36Data/IQ_Data/2020-06-18/0x24022420_2020-06-18-00_01/'
    # DIR = 'D:/Program/PycharmProject/IntermediateFreq/test/'

    fcs = []
    bws = []
    for file in os.listdir(DIR):

        print(file)

        file_path = DIR + file
        iq = np.load(file_path)
        I = iq[0]
        Q = iq[1]

        sig = []
        for i in range(len(I)):
            sig.append(complex(I[i], Q[i]))

        [fc, bw] = fc_bw_estimate(sig, fs)
        fcs.append(fc)
        bws.append(bw)
        print(fc_bw_estimate(sig, fs))

    df = pd.DataFrame({'fc': fcs, 'bw': bws})
    df.to_csv('test_fs_3.csv', index=False)




































