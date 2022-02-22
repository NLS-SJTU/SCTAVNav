#!/usr/bin/python3
# coding=utf-8

import sys
import os
import time
import threading  

import cv2
import numpy as np

class run_thread_c(threading.Thread):
    def __init__(self,name='run_thread_c'):  
        threading.Thread.__init__(self)  
        self.name=name
        
        self.running = True
        self.stopped = threading.Event()
        
        self.print_level=0
        
    def print_info(self,t):
        if self.print_level>=3:
            print(self.name+': &d'+t) 
            sys.stdout.flush()
        
    def print_error_info(self,t):
        if self.print_level>=1:
            print('**** '+self.name+': Error! '+t)
            sys.stdout.flush() 

    def print_warning_info(self,t):
        if self.print_level>=2:
            print('---- '+self.name+': Warning! '+t)
            sys.stdout.flush()
    
    def main_loop(self):
        self.print_error_info('main_loop(), subclasses must override this function')
        return False

    def clean_up(self):
        pass

    def run(self):  
        self.print_info('run(), thread started')

        self.running=True
        self.stopped.clear()

        while self.running:
            if not self.main_loop():
                break

        self.stopped.set()
        self.print_info('run(): thread stopped')

    def stop(self):
        self.print_info('stopping thread...')
        self.running=False

        self.print_info('self.running=False, self.stopped.wait()')
        self.stopped.wait()
        
        self.print_info('cleanup thread...')
        self.clean_up()
        return

