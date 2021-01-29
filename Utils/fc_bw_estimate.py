
# -*- coding:utf-8 -*-

from numpy.fft import fft
import numpy as np
from scipy import signal
import os
import pandas as pd
import config as cfg
import matplotlib.pyplot as plt
import random


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

    # plt.plot(power_spectrum)
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
    # num_fft = 2048
    #
    # t = np.arange(0, 2, 1 / fs)
    # f0 = 100
    # f1 = 200
    # # x = np.cos(2*np.pi*f0*t) + 3*np.cos(2*np.pi*f1*t) + np.random.randn(t.size)
    # x = np.cos(2 * np.pi * f0 * t) + np.random.randn(t.size)

    N = 20
    bit_rate = 1
    fc = 2
    fs = 100

    # 生成比特流
    bit_stream = []
    for i in range(0, N):
        bit = random.randint(0, 1)
        bit = 2 * bit - 1
        bit_stream.append(bit)

    # 生成IQ两路数据
    I = []
    Q = []
    for i in range(0, N):
        if i % 2 == 0:
            I.append(bit_stream[i])
        else:
            Q.append(bit_stream[i])

    # 生成基带信号（加采样）
    bit_data = []
    for i in range(0, N):
        bit_data.append([bit_stream[i]] * (fs // bit_rate))
    I_data = []
    Q_data = []
    for i in range(0, N // 2):
        I_data.append([I[i]] * (2 * fs // bit_rate))
        Q_data.append([Q[i]] * (2 * fs // bit_rate))

    # 生成中频信号（加载波）
    I_carrier = []
    Q_carrier = []
    t_bit = np.linspace(0, 2 // bit_rate, 2 // bit_rate * fs, endpoint=False)
    for i in range(0, N // 2):
        I_carrier.extend((I[i] * np.cos(2 * np.pi * fc * t_bit)).tolist())
        Q_carrier.extend((Q[i] * np.cos(2 * np.pi * fc * t_bit + np.pi / 2)).tolist())

    # 生成复信号
    sig = []
    for i in range(len(I_carrier)):
        sig.append(complex(I_carrier[i], Q_carrier[i]))

    fc_bw = fc_bw_estimate(sig, fs)
    fc = fc_bw[0]
    bw = fc_bw[1]
    print(fc, bw)

    # fs = 977419
    # DIR = 'D:/36Data/IQ_Data/2020-06-18/0x24022420_2020-06-18-00_01/'
    # # DIR = 'D:/Program/PycharmProject/IntermediateFreq/test/'
    #
    # fcs = []
    # bws = []
    # for file in os.listdir(DIR):
    #
    #     print(file)
    #
    #     file_path = DIR + file
    #     iq = np.load(file_path)
    #     I = iq[0]
    #     Q = iq[1]
    #
    #     sig = []
    #     for i in range(len(I)):
    #         sig.append(complex(I[i], Q[i]))
    #
    #     [fc, bw] = fc_bw_estimate(sig, fs)
    #     fcs.append(fc)
    #     bws.append(bw)
    #     print(fc_bw_estimate(sig, fs))
    #
    # df = pd.DataFrame({'fc': fcs, 'bw': bws})
    # df.to_csv('test_fs_3.csv', index=False)




































