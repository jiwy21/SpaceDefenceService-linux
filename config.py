# -*- coding:utf-8 -*-


ZPSX_SOURCE_DIR = '/media/tmxk-111/4C20F5B420F5A55C/36Data/Intermediate/2020-06-13/'
ZPSX_DEST_DIR = '/media/tmxk-111/4C20F5B420F5A55C/36Data/IQ_Data/2020-06-13/'

SXDW_SOURCE_DIR = '/media/tmxk-111/4C20F5B420F5A55C/36Data/sxdw/'

# Days in 0001-01-01 : 2000-01-01
DAYS_DELTA = 730119
TABLE_ZPSX = 'zpsx'
TABLE_SXDW = 'sxdw'

PORT = 29000

USER = 'postgres'
PASSWORD_DB = 'postgres'
PORT_DB = '5432'
SERVER_IP = '127.0.0.1'
DATABASE = 'postgres'

# 定位结果到达时间前的秒数
SECOND_BEFORE_SXDW = 10
SECOND_AFTER_SXDW = 0

# 中频数据到达时间后的秒数
SECOND_BEFORE_INTERMEDIATE = 0
SECOND_AFTER_INTERMEDIATE = 10

# 信噪比估计
EIG_THRESHOLD = 0
N_SNR = 50
K_SNR = 100

# frequency_rate estimate
FS_NFFT = 3
N_FFT = 100000

# 最小包数目
MIN_PACKAGES = 50

# code_rate
# 小波变换尺度
SCALE = 10
# 极大值阶数
MAX_ORDER = 300


# 调制识别阈值
AMP_THRESHOLD = 10
ASK_THRESHOLD = 1.8
FSK_THRESHOLD = 20




