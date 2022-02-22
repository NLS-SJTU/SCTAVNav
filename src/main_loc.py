# coding=utf-8

import time
from localization.loc_handler import *

if __name__ == '__main__':
    lh = loc_handler()
    lh.start()

    print('[LOCALIZATION]start')
    while(not lh.shutdown):
        if(lh.work):
            lh.updatepos()

        time.sleep(0.01)

    print('[LOCALIZATION]exit')