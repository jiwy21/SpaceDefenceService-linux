

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


def code_rate_estimate(iq, a, fs):
    """
    :param iq: original signal
    :param fs: sample rate
    :return:
    """
    iq = np.abs(np.array(iq, dtype=complex))
    X = np.abs(cwt(iq, a, 'cgau8', len(iq))[0][0])

    # 求极大极小值点
    max_points = signal.argrelextrema(X, np.greater)[0]
    min_points = signal.argrelextrema(-X, np.greater)[0]
    extreme_points = max_points.tolist() + min_points.tolist()
    extreme_points.sort()

    # 基于极值点求微分
    diff_points = np.zeros(len(X))
    if extreme_points[0] == max_points[0]:
        diff_points[max_points[0]] = X[max_points[0]]
    else:
        diff_points[min_points[0]] = -X[min_points[0]]

    for i in range(1, len(extreme_points)):
        diff_points[extreme_points[i]] = X[extreme_points[i]] - X[extreme_points[i - 1]]

    # 对微分结果进行fft变换
    diff_fft = np.abs(fft(diff_points, cfg.N_FFT)[:cfg.N_FFT // 2])

    # 查找对应位置序号
    K = np.argmax(diff_fft)

    # 计算波特率
    baud_rate = K * fs / cfg.N_FFT

    return baud_rate


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

    fs = 900000
    a = 30
    DIR = 'D:/36Data/IQ_Data/2020-06-18/0x24022420_2020-06-18-00_01/'
    # DIR = 'D:/Program/PycharmProject/IntermediateFreq/test/'

    code_rates = []
    for file in os.listdir(DIR):

        print(file)

        file_path = DIR + file
        iq = np.load(file_path)
        I = iq[0]
        Q = iq[1]

        sig = []
        for i in range(len(I)):
            sig.append(complex(I[i], Q[i]))

        code_rate = code_rate_estimate(sig, a, fs)
        code_rates.append(code_rate)
        print(code_rate)

    df = pd.DataFrame({'code_rates': code_rates})
    df.to_csv('test_fs_3.csv', index=False)



















































































