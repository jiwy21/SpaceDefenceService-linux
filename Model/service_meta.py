# -*- coding:utf-8 -*-


class IntermediateMeta(object):
    """
    """

    def __init__(self, id=None, count=None, packages=None, satellite=None, arrival_time=None,
                 duration=None, freq_down=None, freq_unit=None, iqlocation=None, iqlen=None,
                 bitrate=None, coderate=None, bandwidth=None):

        """
        :param id:
        :param count:
        :param packages:
        :param satellite:
        :param arrival_time:
        :param duration:
        :param freq_down:
        :param freq_unit:
        :param iqlocation:
        :param iqlen:
        :param bitrate:
        :param coderate:
        :param bandwidth:
        """

        self._id = id
        self._count = count
        self._packages = packages
        self._satellite = satellite
        self._arrival_time = arrival_time
        self._duration = duration
        self._freq_down = freq_down
        self._freq_unit = freq_unit
        self._iqlocation = iqlocation
        self._iqlen = iqlen
        self._bitrate = bitrate
        self._coderate = coderate
        self._bandwidth = bandwidth

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        self._id = val

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, val):
        self._count = val

    @property
    def packages(self):
        return self._packages

    @packages.setter
    def packages(self, val):
        self._packages = val

    @property
    def satellite(self):
        return self._satellite

    @satellite.setter
    def satellite(self, val):
        self._satellite = val

    @property
    def arrival_time(self):
        return self._arrival_time

    @arrival_time.setter
    def arrival_time(self, val):
        self._arrival_time = val

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, val):
        self._duration = val

    @property
    def freq_down(self):
        return self._freq_down

    @freq_down.setter
    def freq_down(self, val):
        self._freq_down = val

    @property
    def freq_unit(self):
        return self._freq_unit

    @freq_unit.setter
    def freq_unit(self, val):
        self._freq_unit = val

    @property
    def iqlocation(self):
        return self._iqlocation

    @iqlocation.setter
    def iqlocation(self, val):
        self._iqlocation = val

    @property
    def iqlen(self):
        return self._iqlen

    @iqlen.setter
    def iqlen(self, val):
        self._iqlen = val

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, val):
        self._bitrate = val

    @property
    def coderate(self):
        return self._coderate

    @coderate.setter
    def coderate(self, val):
        self._coderate = val

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, val):
        self._bandwidth = val


class IQMeta(object):
    """
    """
    def __init__(self, i=None, q=None):
        """
        :param i: i路数据
        :param q: q路数据
        """
        self._i = i
        self._q = q

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, val):
        self._i = val

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = val




