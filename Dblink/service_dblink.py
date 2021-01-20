
import datetime
import psycopg2
import config as cfg
from Model.service_meta import IntermediateMeta, IQMeta
import numpy as np
import time


class ServiceDblink(object):

    def __init__(self):
        pass

    @staticmethod
    def query_intermediate_list(seek_start, seek_end):
        """
        :param seek_start: str
        :param seek_end: str
        :return:
        """
        # 将date、seek_start、seek_end字符串转为date、time格式
        # seek_start_query = datetime.strptime(seek_start, '%Y-%m-%d %H:%M:%S')
        # seek_end_query = datetime.strptime(seek_end, '%Y-%m-%d %H:%M:%S')

        # 编写sql语句
        sql = "select id, count, packages, satellite, arrival_time, duration, freq_down, \
              freq_unit, iqlocation, iqlen, bitrate, coderate, bandwidth from zpsx where \
              arrival_time between '%s' and '%s'" % (seek_start, seek_end)

        # 连接数据库
        conn = psycopg2.connect(database=cfg.DATABASE, user=cfg.USER, password=cfg.PASSWORD_DB,
                                host=cfg.SERVER_IP, port=cfg.PORT_DB)
        cur = conn.cursor()
        cur.execute(sql)
        zpsxs = cur.fetchall()
        conn.close()

        # 格式化中频数据
        zpsxs_uni = []
        for zpsx in zpsxs:
            zpsxs_uni.append(IntermediateMeta(zpsx[0], zpsx[1], zpsx[2], zpsx[3], zpsx[4], zpsx[5], zpsx[6],
                                              zpsx[7], zpsx[8], zpsx[9], zpsx[10], zpsx[11], zpsx[12]))

        return zpsxs_uni

    @staticmethod
    def query_intermediate_iq(id_zpsx):
        """
        :param id_zpsx: 中频时隙数据编号
        :return:
        """

        # 编写sql语句
        sql = "select iqlocation from zpsx where id=%s" % id_zpsx

        # 连接数据库
        conn = psycopg2.connect(database=cfg.DATABASE, user=cfg.USER, password=cfg.PASSWORD_DB,
                                host=cfg.SERVER_IP, port=cfg.PORT_DB)
        cur = conn.cursor()
        cur.execute(sql)
        iqlocation = cur.fetchall()[0][0]
        conn.close()

        # 从文件读取iq两路数据并格式化
        iq = np.load(iqlocation).tolist()
        iq_uni = []
        iq_uni.append(IQMeta(iq[0], iq[1]))

        return iq_uni

    @staticmethod
    def query_intermediate_list_sxdw(seek_time):
        """
        :param seek_time: str
        :return:
        """
        # 将date、seek_start、seek_end字符串转为date、time格式
        # seek_start_query = datetime.strptime(seek_start, '%Y-%m-%d %H:%M:%S')
        # seek_end_query = datetime.strptime(seek_end, '%Y-%m-%d %H:%M:%S')

        seek_time_date = datetime.datetime.strptime(seek_time, '%Y-%m-%d %H:%M:%S')
        seek_time_before = str(seek_time_date - datetime.timedelta(seconds=cfg.SECOND_BEFORE))
        seek_time_after = str(seek_time_date + datetime.timedelta(seconds=cfg.SECOND_AFTER))

        # 编写sql语句
        sql = "select id, count, packages, satellite, arrival_time, duration, freq_down, \
              freq_unit, iqlocation, iqlen, bitrate, coderate, bandwidth from zpsx where \
              arrival_time between '%s' and '%s'" % (seek_time_before, seek_time_after)

        # 连接数据库
        conn = psycopg2.connect(database=cfg.DATABASE, user=cfg.USER, password=cfg.PASSWORD_DB,
                                host=cfg.SERVER_IP, port=cfg.PORT_DB)
        cur = conn.cursor()
        cur.execute(sql)
        zpsxs = cur.fetchall()
        conn.close()

        # 格式化中频数据
        zpsxs_uni = []
        for zpsx in zpsxs:
            zpsxs_uni.append(IntermediateMeta(zpsx[0], zpsx[1], zpsx[2], zpsx[3], zpsx[4], zpsx[5], zpsx[6],
                                              zpsx[7], zpsx[8], zpsx[9], zpsx[10], zpsx[11], zpsx[12]))

        return zpsxs_uni





























