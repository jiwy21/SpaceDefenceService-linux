
# -*- coding: utf-8 -*-


#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import hilbert

sampling_rate = 256#采样频率
t = np.arange(0,1.0,1.0/sampling_rate)
f1 = 1#频率
f2 = 2
f3 = 3
data = np.piecewise(t,[t<1,t<0.8,t<0.3],
                    [lambda t : 2 * np.sin(2 * np.pi * f2 * t),
                     lambda t : np.sin(2 * np.pi * f2 * t),
                     lambda t : 3 * np.sin(2 * np.pi * f2 * t)])


# Q = np.piecewise(t,[t<1,t<0.8,t<0.3],
#                     [lambda t : hilbert(np.sin(2 * np.pi * f1 * t)),
#                      lambda t : hilbert(np.sin(2 * np.pi * f2 * t)),
#                      lambda t : hilbert(np.sin(2 * np.pi * f3 * t))])

Q = hilbert(data)


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(data)
plt.subplot(2, 1, 2)
plt.plot(Q)
plt.show()


















