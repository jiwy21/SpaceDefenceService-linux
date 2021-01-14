

from View import app
from Logic.service_logic import ServiceLogic
from flask import request


@app.route('/intermediate_list', methods=['GET', 'POST'])
@app.route('/intermediate_list.json', methods=['GET', 'POST'])
def intermediate_list():
    """
    :return: 中频数据json列表
    """
    seek_start = request.args.get('time_begin')
    seek_end = request.args.get('time_end')

    logic = ServiceLogic()
    return logic.list_intermediate(seek_start, seek_end)


@app.route('/<int:id_zpsx>/intermediate_IQ', methods=['GET', 'POST'])
@app.route('/<int:id_zpsx>/intermediate_IQ.json', methods=['GET', 'POST'])
def intermediate_iq(id_zpsx): 
    """
    :param id_zpsx: 中频数据编号
    :return: IQ两路数据
    """
    logic = ServiceLogic()
    return logic.extract_iq(id_zpsx)



















