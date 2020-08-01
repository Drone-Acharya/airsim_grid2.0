import traj_gen
import numpy as np

from traj_gen import poly_trajectory as pt

import time

def get_dummy_waypoints():
	n_gates = 5
	width = 10
	d_bw_gates = 1*4
	gate_stand = 3*2
	gate_height = 1
	gate_length = 1

	# Build Dummy Environment for Visualisation
	import random
	gate_coords_y = [random.randrange(1, 100*width)/100 for p in range(0, n_gates)]
	gate_coords_x = [d_bw_gates*x for x in range(1, n_gates+1)]
	gate_coords_z = [gate_stand + gate_height/2 for _ in range(n_gates)]

	# Build dummy Env
	waypoints = [[gate_coords_x[i], gate_coords_y[i], gate_coords_z[i]] for i in range(n_gates)]
	gate_stands = [[[gate_coords_x[i], gate_coords_y[i], 0], [gate_coords_x[i], gate_coords_y[i], gate_stand]] for i in range(n_gates)]
	gate_l_up = [[[gate_coords_x[i], gate_coords_y[i]-gate_length/2, gate_stand+gate_height], [gate_coords_x[i], gate_coords_y[i]+gate_length/2, gate_stand+gate_height]] for i in range(n_gates)]
	gate_l_down = [[[gate_coords_x[i], gate_coords_y[i]-gate_length/2, gate_stand], [gate_coords_x[i], gate_coords_y[i]+gate_length/2, gate_stand]] for i in range(n_gates)]
	gate_h_left = [[[gate_coords_x[i], gate_coords_y[i]-gate_length/2, gate_stand], [gate_coords_x[i], gate_coords_y[i]-gate_length/2, gate_stand+gate_height]] for i in range(n_gates)]
	gate_h_right = [[[gate_coords_x[i], gate_coords_y[i]+gate_length/2, gate_stand], [gate_coords_x[i], gate_coords_y[i]+gate_length/2, gate_stand+gate_height]] for i in range(n_gates)]

	return waypoints, gate_length, gate_height

def get_trajectory(waypoints, gate_length, gate_height, init, tdelta = 0.1, time_between_gates = 2):
    dim = 3
    knots = np.linspace(0, (1+len(waypoints))*time_between_gates, num = 2+len(waypoints), dtype=np.float)
    order = 8
    optimTarget = 'end-derivative' #'end-derivative' 'poly-coeff'
    maxConti = 4
    objWeights = np.array([1, 0, 0, 1, 1])
    pTraj = pt.PolyTrajGen(knots, order, optimTarget, dim, maxConti)

    # 2. Pin
    ts = np.array([0.0])
    Xs = np.array([init])
    Xdot = np.array([0, 0, 0])
    Xddot = np.array([0, 0, 0])

    # create pin dictionary
    for i in range(Xs.shape[0]):
        pin_ = {'t':ts[i], 'd':0, 'X':Xs[i]}
        print(pin_)
        pTraj.addPin(pin_)

    pin_ = {'t':ts[0], 'd':1, 'X':Xdot,}
    pTraj.addPin(pin_)
    pin_ = {'t':ts[0], 'd':2, 'X':Xddot,}
    pTraj.addPin(pin_)


    pTraj.addPin({'t':(1+len(waypoints))*time_between_gates, 'd':0, 'X':np.array([waypoints[-1][0]+2, waypoints[-1][1], waypoints[-1][2]])})

    # passCube = np.array([[3.0, 4.], [-3.0, -2.], [ 1., 2] ]) # in shape [dim, xxx]
    # pin_ = {'t':3, 'd':0, 'X':passCube2}
    # pTraj.addPin(pin_)

    for i, wp in enumerate(waypoints):

        pin_ = {'t':time_between_gates+i*time_between_gates, 'd':0, 'X':np.array(wp)}
        # pTraj.addPin(pin_)
        passCube = np.array([wp[0], wp[1], wp[2]]) # in shape [dim, xxx]
        pin_ = {'t':time_between_gates+i*time_between_gates, 'd':0, 'X':passCube}
        pTraj.addPin(pin_)

        print(pin_)
        # break
    print(np.array([waypoints[-1][0]+10, waypoints[-1][1], waypoints[-1][2]]))
    # solve
    pTraj.setDerivativeObj(objWeights)
    print("solving")
    time_start = time.time()
    pTraj.solve()
    time_end = time.time()
    print(time_end - time_start)
    print(int((1+len(waypoints))*time_between_gates//tdelta))
    rng = np.linspace(0, (1+len(waypoints))*time_between_gates, int((1+len(waypoints))*time_between_gates//tdelta))
    velocities = pTraj.eval(rng, 1)
    path = pTraj.eval(rng, 0)

    print(path)
    print(velocities)
    print('path')
    fig_title ='poly order : {0} / max continuity: {1} / minimized derivatives order: {2}'.format(pTraj.N, pTraj.maxContiOrder, np.where(pTraj.weight_mask>0)[0].tolist())
    pTraj.showPath(fig_title)
    return path, velocities