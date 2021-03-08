

# -*- coding:utf-8 -*-
import os
import sys
sys.path.append(os.getcwd())




import config as cfg
from View import app




if __name__ == '__main__':
    app.run(host='', port=cfg.PORT, debug=False, threaded=False)






