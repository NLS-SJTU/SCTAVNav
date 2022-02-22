# Localization
## Intruduction
Localization submodule handler. It uses VPR to get single localization result.

## Usage
* (reset);
* (config);
* send image(execute);
* wait until working done(query);
* get result(get_result);

## Srvs
* reset(param)
> reset localization information: param=None, no return

* config(param)
> set getting top N result from VPR: param={'top_n_node': N}, return={}

> set path localization with action input or not: param={'update_with_action': True/False}, return={}

> set directions (from self) of images sent to localization at one time, default only one image at 0 degree(forward): param={'directions_of_imgs': [dir0, dir1, ...]}, return={}

* execute(param)
> send images taken in different directions to localization: param={'image':[img0, img1, ...]}, return={}

* query(param)
> ask for best path since last reset: param={'sim_path_loc_result':None}, return={}

> ask for whether it is working: param={'locworking':None}, return=True/False

* get_result(param)
> ask for localization result: param={'est_pos':None}, return=[[node_id0, node_id1, ...], [dir0, dir1, ...], [prob0, prob1, ...]]