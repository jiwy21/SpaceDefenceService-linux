
from Logic.logic_base import LogicBase
from Dblink.service_dblink import ServiceDblink


class ServiceLogic(LogicBase):
    """
    logic for intelligent service
    """
    def __init__(self):
        LogicBase.__init__(self)

    def list_intermediate(self, seek_start, seek_end):
        """
        :param seek_start:
        :param seek_end:
        :return:
        """
        self.items = ServiceDblink.query_intermediate_list(seek_start, seek_end)
        return self.toJson()

    def extract_iq(self, id):
        """
        :param id:
        :return:
        """
        self.items = ServiceDblink.query_intermediate_iq(id)
        return self.toJson()

    def list_intermediate_sxdw(self, seek_time):
        """
        :param seek_time: 需要检索的时间
        :return:
        """
        self.items = ServiceDblink.query_intermediate_list_sxdw(seek_time)
        return self.toJson()












