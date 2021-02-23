

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

    amplitude = np.abs(iq)

    mean = np.mean(amplitude)

    amplitude_norm = amplitude / mean - 1
    amplitude_norm_fft = np.abs(fft(amplitude_norm, n_fft))
    r_max = np.max(amplitude_norm_fft ** 2) / len(amplitude)
    r_max_dB = 10 * np.log10(r_max)

    return r_max_dB


def us42_cal(iq):
    """
    :param iq: input signal
    :return:
    """

    # 幅值
    a = np.abs(iq)

    # 幅值均值
    ma = np.mean(a)

    # 归一化幅度
    scn = a / ma

    # 归一化信号四阶矩紧致性
    us42 = np.mean(scn ** 4) / (np.mean(scn ** 2) ** 2)

    return us42


def uf42_cal(iq, fs):
    """
    :param iq: input signal
    :return:
    """

    # 相位序列
    phi = np.angle(iq)

    # 归一化瞬时频率
    phi_delta = phi[1:] - phi[:-1]
    freq = fs * phi_delta / (2 * np.pi)
    freq -= np.mean(freq)

    # 归一化瞬时频率四阶矩紧致性
    uf42 = np.mean(freq ** 4) / (np.mean(freq ** 2) ** 2)

    return uf42


def mod_recognition(iq, fs, n_fft=cfg.N_FFT):
    """
    :param iq: input signal
    :param fs: sample rate
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

        uf42 = uf42_cal(iq, fs=fs)
        if uf42 < cfg.FSK_THRESHOLD:
            return 'FSK'
        else:
            return 'PSK'


def my_plot(signal):
    """
    :param signal:
    :return:
    """

    signal_i = np.real(signal)
    signal_q = np.imag(signal)
    signal_abs = np.abs(signal)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(signal_i)
    plt.subplot(3, 1, 2)
    plt.plot(signal_q)
    plt.subplot(3, 1, 3)
    plt.plot(signal_abs)
    plt.show()


def awgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise


if __name__ == '__main__':

    N = 2000
    bit_rate = 10000
    fc = 40000
    fs = 1000000
    M = 8
    mul = 1000

    # 生成比特流
    bit_stream = []
    for i in range(0, N):
        bit = random.randint(0, 1)
        bit_stream.append(bit)
    bit_stream = np.array(bit_stream)

    # QPSK
    # 生成I路Q路数据
    bit_stream = 2 * bit_stream - 1
    I = []
    Q = []
    for i in range(0, N):
        if i % 2 == 0:
            I.append(bit_stream[i])
        else:
            Q.append(bit_stream[i])

    # 生成中频信号（加载波）
    I_carrier = []
    Q_carrier = []
    t_bit = np.linspace(0, 1 / bit_rate, int(1 / bit_rate * fs), endpoint=False)
    for i in range(0, N // 2):
        I_carrier.extend((I[i] * np.cos(2 * np.pi * fc * t_bit)).tolist())
        Q_carrier.extend((Q[i] * np.cos(2 * np.pi * fc * t_bit + np.pi / 2)).tolist())

    # 4FSK
    # I = []
    # n = 0
    # for i in range(0, N):
    #     if i % 2 == 0:
    #         n += bit_stream[i]
    #     else:
    #         n = 2 * n + bit_stream[i] + 1
    #         I.append(n)
    #         n = 0
    #
    # I_carrier = []
    # t_bit = np.linspace(0, 1 / bit_rate, int(1 / bit_rate * fs), endpoint=False)
    # for i in range(0, N // 2):
    #     I_carrier.extend((np.cos(2 * np.pi * I[i] * fc * t_bit)).tolist())
    # Q_carrier = hilbert(I_carrier)

    # 4ASK
    # I = []
    # n = 0
    # for i in range(0, N):
    #     if i % 2 == 0:
    #         n += bit_stream[i]
    #     else:
    #         n = 2 * n + bit_stream[i] + 1
    #         I.append(n)
    #         n = 0
    #
    # I = bit_stream + 1
    #
    # I_carrier = []
    # t_bit = np.linspace(0, 1 / bit_rate, int(1 / bit_rate * fs), endpoint=False)
    # for i in range(0, N // 2):
    #     I_carrier.extend((I[i] * np.cos(2 * np.pi * fc * t_bit)).tolist())
    # Q_carrier = hilbert(I_carrier)

    # 16QAM
    # 映射字典
    # bits2iq = {'1110': (-3, 3),  '1010': (-1, 3),  '0010': (1, 3),  '0110': (3, 3),
    #            '1111': (-3, 1),  '1011': (-1, 1),  '0011': (1, 1),  '0111': (3, 1),
    #            '1101': (-3, -1), '1001': (-1, -1), '0001': (1, -1), '0101': (3, -1),
    #            '1100': (-3, -3), '1000': (-1, -3), '0000': (1, -3), '0100': (3, -3)}
    #
    # I = []
    # Q = []
    # for i in range(0, N // 4):
    #     bits = ''
    #     for j in range(0, 4):
    #         bits += str(bit_stream[i * 4 + j])
    #     iq = bits2iq[bits]
    #
    #     I.append(iq[0])
    #     Q.append(iq[1])
    #
    # I_carrier = []
    # Q_carrier = []
    # t_bit = np.linspace(0, 1 / bit_rate, int(1 / bit_rate * fs), endpoint=False)
    # for i in range(0, N // 4):
    #     I_carrier.extend((I[i] * 2000 * np.cos(2 * np.pi * fc * t_bit)).tolist())
    #     Q_carrier.extend((Q[i] * 2000 * np.cos(2 * np.pi * fc * t_bit + np.pi / 2)).tolist())

    # 加入噪声
    snr = 10
    I_carrier = np.array(I_carrier)
    Q_carrier = np.array(Q_carrier)
    I_carrier = awgn(x=I_carrier, snr=snr)
    Q_carrier = awgn(x=Q_carrier, snr=snr)

    # 生成复信号
    n = len(I_carrier)
    sigs = []
    t = np.linspace(0, n // fs, n, endpoint=False)
    for i in range(n):
        sig = complex(I_carrier[i], Q_carrier[i])
        sigs.append(sig)

    # my_plot(sigs)
    mod = mod_recognition(sigs, fs=fs, n_fft=500000)
    print()




























































