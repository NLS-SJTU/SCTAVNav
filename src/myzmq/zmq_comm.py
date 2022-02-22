#!/usr/bin/python3
# coding=utf-8

import sys
import pickle
import zmq
import numpy as np
from myzmq.jpeg_compress import *
from myzmq.misc import *

version_flag = sys.version[0]


class zmq_comm_cli_c:
    def __init__(self, name='zmq_comm_cli_c', ip='127.0.0.1', port=1111):
        print('[TRK] zmq_comm_cli_c.__init__()')

        self.name=name
        
        print('[TRK] name=%s, making ZMQ REQ socket'%name)
        print('[TRK] server ip: '+ip)
        print('[TRK] server port: %d'%port)
        
        ctx=zmq.Context()
        self.skt=ctx.socket(zmq.REQ)
        self.skt.connect('tcp://%s:%d'%(ip,port))
        return

    def api_call(self,api_name,param):        
        tx = pickle.dumps((self.name,api_name,param), protocol=2) 
        self.skt.send(tx)
        rx=self.skt.recv()
        if(version_flag == '2'):
            return pickle.loads(rx)
        else:
            return pickle.loads(rx,encoding='latin1')
    
    def reset(self,param=None): 
        return self.api_call('reset',param)
    def config(self,param=None): 
        return self.api_call('config',param)
    def query(self,param=None): 
        return self.api_call('query',param)
    def get_result(self,param=None): 
        return self.api_call('get_result',param)
    def execute(self,param=None): 
        return self.api_call('execute',param)
    def stop(self,param=None): 
        return self.api_call('stop',param)
        
    def wait(self,state,delay=0.1,timeout=0.0):
        while True:
            if isinstance(state, list):
                if self.query('state') in state:
                    return state
            else:
                if self.query('state')==state:
                    return state
            time.sleep(delay)
            if timeout>0:
                timeout-=delay
                if timeout<=0: 
                    return 'timeout' 
        return 'error'



class zmq_comm_svr_c(run_thread_c):
    def __init__(self, name='zmq_comm_svr_c', ip='127.0.0.1', port=1001):
        run_thread_c.__init__(self,name)  

        print('[TRK] name=%s, making ZMQ REP socket'%name)
        print('[TRK] server ip: '+ip)
        print('[TRK] server port: %d'%port)
        ctx=zmq.Context()
        self.skt=ctx.socket(zmq.REP)
        self.skt.bind('tcp://*:%d'%port)
        return

    def main_loop(self):
        try:
            rx=self.skt.recv()#
        except:
            time.sleep(0.001)
            return True
        
        if(version_flag == '2'):
            name,api_name,param = pickle.loads(rx)
        else:
            name,api_name,param = pickle.loads(rx,encoding='latin1')
        if name != self.name:
            print('[WRN] name mis-match (%s/%s)'%(name,self.name))
        
        if   api_name=='reset'     : self.skt.send(pickle.dumps(self.reset     (param), protocol=2))
        elif api_name=='config'    : self.skt.send(pickle.dumps(self.config    (param), protocol=2))
        elif api_name=='query'     : self.skt.send(pickle.dumps(self.query     (param), protocol=2))
        elif api_name=='get_result': self.skt.send(pickle.dumps(self.get_result(param), protocol=2))
        elif api_name=='execute'   : self.skt.send(pickle.dumps(self.execute   (param), protocol=2))
        elif api_name=='stop':
            self.skt.send(pickle.dumps(None, protocol=2))
            return False
        else:
            print('unknown api name '+api_name)
            self.skt.send(pickle.dumps(None, protocol=2))
        return True


    def reset(self,param=None): 
        print('[WRN] function reset must be overloaded!')
        return '[WRN] function reset must be overloaded!',param
        
    def config(self,param=None): 
        print('[WRN] function config must be overloaded!')
        return '[WRN] function config must be overloaded!',param
    
    def query(self,param=None): 
        print('[WRN] function query must be overloaded!')
        return '[WRN] function query must be overloaded!',param
    
    def get_result(self,param=None): 
        print('[WRN] function get_result must be overloaded!')
        return '[WRN] function get_result must be overloaded!',param
    
    def execute(self,param=None): 
        print('[WRN] function execute must be overloaded!')
        return '[WRN] function execute must be overloaded!',param

    def wait(self,state,delay=0.1,timeout=0.0):
        while True:
            if isinstance(state, list):
                if self.query('state') in state:
                    return state
            else:
                if self.query('state')==state:
                    return state
            time.sleep(delay)
            if timeout>0:
                timeout-=delay
                if timeout<=0: 
                    return 'timeout'  
        return 'error'

    
def unit_test():
    zmq_comm_svr=zmq_comm_svr_c()
    zmq_comm_svr.start()
    
    zmq_comm_cli=zmq_comm_cli_c()
    print(zmq_comm_cli.reset('test'))
    time.sleep(0.3)
    print(zmq_comm_cli.config('test'))
    time.sleep(0.3)
    print(zmq_comm_cli.query('test'))
    time.sleep(0.3)
    print(zmq_comm_cli.get_result('test'))
    time.sleep(0.3)
    print(zmq_comm_cli.execute('test'))
    time.sleep(0.3)
    
    zmq_comm_svr.stop()

if __name__ == '__main__':
    unit_test()

