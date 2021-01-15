

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


@app.route('/intermediate_IQ', methods=['GET', 'POST'])
@app.route('/intermediate_IQ.json', methods=['GET', 'POST'])
def intermediate_iq():
    """
    :return: IQ两路数据
    """
    id_zpsx = int(request.args.get('id_zpsx'))

    logic = ServiceLogic()
    return logic.extract_iq(id_zpsx)



















