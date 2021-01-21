
import datetime
import psycopg2
import config as cfg
from Model.service_meta import IntermediateMeta, IQMeta, SXDWMeta
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
            zpsxs_uni.append(IntermediateMeta(id=zpsx[0], count=zpsx[1], packages=zpsx[2], satellite=zpsx[3],
                                              arrival_time=zpsx[4], duration=zpsx[5], freq_down=zpsx[6],
                                              freq_unit=zpsx[7], iqlocation=zpsx[8], iqlen=zpsx[9], bitrate=zpsx[10],
                                              coderate=zpsx[11], bandwidth=zpsx[12]))

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
        seek_time_before = str(seek_time_date - datetime.timedelta(seconds=cfg.SECOND_BEFORE_SXDW))
        seek_time_after = str(seek_time_date + datetime.timedelta(seconds=cfg.SECOND_AFTER_SXDW))

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
            zpsxs_uni.append(IntermediateMeta(id=zpsx[0], count=zpsx[1], packages=zpsx[2], satellite=zpsx[3],
                                              arrival_time=zpsx[4], duration=zpsx[5], freq_down=zpsx[6],
                                              freq_unit=zpsx[7], iqlocation=zpsx[8], iqlen=zpsx[9], bitrate=zpsx[10],
                                              coderate=zpsx[11], bandwidth=zpsx[12]))

        return zpsxs_uni


    @staticmethod
    def query_sxdw_list_intermediate(seek_time):
        """
        :param seek_time:
        :return:
        """

        seek_time_date = datetime.datetime.strptime(seek_time, '%Y-%m-%d %H:%M:%S')
        seek_time_before = str(seek_time_date - datetime.timedelta(seconds=cfg.SECOND_BEFORE_INTERMEDIATE))
        seek_time_after = str(seek_time_date + datetime.timedelta(seconds=cfg.SECOND_AFTER_INTERMEDIATE))

        # 编写sql语句
        sql = "select id, id_sxdw, arrival_time, freq_down, freq_down_unit, batchnumber, true_value_lon, "\
                      "true_value_lat, true_value_error, result_confidence, false_value_lon, false_value_lat," \
                      "freq_up, freq_up_unit, multi_access_mode, modulate_pattern, code_mode, bandwidth," \
                      "bandwidth_unit, sps, sps_unit, medial_sat_norad, adjacent_sat_norad1, adjacent_sat_norad2 " \
                      "from sxdw where arrival_time between '%s' and '%s'" % (seek_time_before, seek_time_after)

        # 连接数据库
        conn = psycopg2.connect(database=cfg.DATABASE, user=cfg.USER, password=cfg.PASSWORD_DB,
                                host=cfg.SERVER_IP, port=cfg.PORT_DB)
        cur = conn.cursor()
        cur.execute(sql)
        sxdws = cur.fetchall()
        conn.close()

        # 格式化定位数据
        sxdws_uni = []
        for sxdw in sxdws:
            sxdws_uni.append(SXDWMeta(sxdw[0], sxdw[1], sxdw[2], sxdw[3], sxdw[4], sxdw[5], sxdw[6], sxdw[7],
                                      sxdw[8], sxdw[9], sxdw[10], sxdw[11], sxdw[12], sxdw[13], sxdw[14], sxdw[15],
                                      sxdw[16], sxdw[17], sxdw[18], sxdw[19], sxdw[20], sxdw[21], sxdw[22], sxdw[23]))

        return sxdws_uni

















