# coding=utf-8

import time
from planner.planner_handler import *

if __name__ == '__main__':
    st = time.time()
    nh = planner_handler()
    nh.start()
    print('[PLANNER]planner start')
    while(not nh.shutdown):
        if(nh.replan):
            nh.replanall()

        if(nh.work):
            nh.get_action()

        time.sleep(0.01)
    print('[PLANNER]exit')
