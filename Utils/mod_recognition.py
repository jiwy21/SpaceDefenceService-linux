

# -*- coding:utf-8 -*-

from numpy.fft import fft
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import psd
import pandas as pd
from pywt import cwt
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft
import config as cfg
import random
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from scipy.fftpack import hilbert
from scipy.interpolate import interp1d



def r_max_cal(iq, n_fft=cfg.N_FFT):
    """
    :param iq: input signal
    :return:
    """
    amplitude = np.abs(iq[:20000])

    sum = np.sum(amplitude)
    ns = len(amplitude)
    mean = sum / ns

    amplitude_norm = amplitude / mean - 1
    amplitude_norm_fft = np.abs(fft(amplitude_norm, n_fft))
    r_max = np.max(amplitude_norm_fft ** 2) / ns
    r_max_log = np.log10(r_max)

    return r_max


def us42_cal(iq):
    """
    :param iq: input signal
    :return:
    """

    return 0


def uf42_cal(iq):
    """
    :param iq: input signal
    :return:
    """

    return 0


def mod_recognition(iq, n_fft=cfg.N_FFT):
    """
    :param iq: input signal
    :return:
    """

    r_max = r_max_cal(iq, n_fft)
    print(r_max)

    if r_max >= cfg.AMP_THRESHOLD:

        us42 = us42_cal(iq)
        if us42 > cfg.ASK_THRESHOLD:
            return 'ASK'
        else:
            return 'QAM'

    else:

        uf42 = uf42_cal(iq)
        if uf42 < cfg.FSK_THRESHOLD:
            return 'FSK'
        else:
            return 'PSK'
















if __name__ == '__main__':

    N = 40000
    bit_rate = 40
    fc = 40
    fs = 1000

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
    t_bit = np.linspace(0, 1 / bit_rate, int(1 / bit_rate * fs), endpoint=False)
    for i in range(0, N // 2):
        I_carrier.extend((I[i] * np.cos(2 * np.pi * fc * t_bit)).tolist())
        Q_carrier.extend((Q[i] * np.cos(2 * np.pi * fc * t_bit + np.pi / 2)).tolist())

    # 生成复信号并进行频谱搬移
    n = len(I_carrier)
    sigs = []
    t = np.linspace(0, n // fs, n, endpoint=False)
    for i in range(n):
        sig = complex(I_carrier[i], Q_carrier[i])
        # sig *= np.exp(complex(0, 2 * np.pi * fc * t[i]))
        sigs.append(sig)

    # sigs_filtered = butter_lowpass_filter(sigs, fc, fs)

    mod = mod_recognition(sigs)
    print()

    # fs = 900000
    # a = 30
    # DIR = 'D:/36Data/IQ_Data/2020-06-18/0x24022420_2020-06-18-00_01/'
    # # DIR = 'D:/Program/PycharmProject/IntermediateFreq/test/'
    #
    # code_rates = []
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
    #     code_rate = code_rate_estimate(sig, a, fs)
    #     code_rates.append(code_rate)
    #     print(code_rate)
    #
    # df = pd.DataFrame({'code_rates': code_rates})
    # df.to_csv('test_fs_3.csv', index=False)


























































