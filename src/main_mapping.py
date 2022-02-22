#!/usr/bin/env python
# coding=utf-8

from mapping.topo_mapping import topo_mapping
import rospy

if __name__ == '__main__':
    rospy.init_node('topomapping', anonymous=False)
    tm = topo_mapping()
    tm.start()
    print('[remind]hfnet is required.')
    print('[MAPPING]mapping starts.')
    rospy.spin()
    tm.running = False