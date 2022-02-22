# Global planner
## Intruduction
Global planner submodule handler. It uses Dijkstra to search global path and gives local target during autonomous navigation.

## Usage
* set destination(config);
* send localization result(execute);
* wait until working done(query);
* get result;

## Srvs
* reset(param)
> nothing: param=None, no return

* config(param)
> set destination: param={'set_des_pos':nodeid}, return={}

> start logging: param={'start':None}, return={}

> end logging: param={'end':None}, return={}

* execute(param)
> start computing action: param={'pos_now':[[id0, dir0, prob0], ...], 'crossing_type':-1}, return={}

* query(param)
> ask for whether it is working: param={'navworking':None}, return=True/False

> ask for total path length from one node to destination node: param={'totalpathlength':nodeid}, return=pathlength (m)

* get_result(param)
> get action: param={'action':None}, return='F/L/R'

> get local target: param={'localtarget':{'refnode_id':, 'refimg_id':, 'R_b_in_ref':, 't_b_in_ref':, 'refodom':, 'curodom':, 'realdir':, 'comp_direction':}}, return={'localtarget':[x,y,z], 'odomtarget':[xo,yo,zo]}
