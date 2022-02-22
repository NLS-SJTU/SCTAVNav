#!/usr/bin/env python
# coding=utf-8

import time
from core.ui import *


if __name__ == '__main__':
    myui = se_ui()
    myui.start()
    print('[UI]system start')

    while(not myui.shutdown):
        myui.showFigure()
        time.sleep(0.01)

    print('[UI]system shutdown')
