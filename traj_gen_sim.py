import traj_gen
import numpy as np

from traj_gen import poly_trajectory as pt

import time

def get_knots(waypoints, init, end, scale = 10):
    total_time = np.linalg.norm(init - waypoints[0])
    for wpa, wpb in zip(waypoints[1:], waypoints[:-1]):
        total_time += np.linalg.norm(wpa-wpb)
        print(total_time)
    
    total_time += np.linalg.norm(waypoints[-1]-end)
    print(total_time)

    knots = np.zeros((len(waypoints)+2))
    knots[0] = 0
    knots[1] = np.linalg.norm(waypoints[0]-init)/total_time
    for i, (wpa, wpb) in enumerate(zip(waypoints[1:], waypoints[:-1])):
        knots[i+2] = knots[i+1]+np.linalg.norm(wpa-wpb)/total_time
    
    knots[len(waypoints)+1] = 1
    print(scale)
    return (knots*scale).astype(np.float)


def get_trajectory(waypoints, gate_length, gate_height, init, tdelta = 0.1, time_between_gates = 2, silent = False):
    dim = 3
    endPos = np.array([waypoints[-1][0]+1, waypoints[-1][1], waypoints[-1][2]])
    knots = get_knots(waypoints, init, endPos, time_between_gates*(len(waypoints)+2))
    print(knots)
    order = 8
    optimTarget = 'poly-coeff' #'end-derivative' 'poly-coeff'
    maxConti = 6
    objWeights = np.array([0, 0, 0, 1, 1])
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


    pTraj.addPin({'t':knots[-1], 'd':0, 'X':np.array([waypoints[-1][0]+2, waypoints[-1][1], waypoints[-1][2]])})

    # passCube = np.array([[3.0, 4.], [-3.0, -2.], [ 1., 2] ]) # in shape [dim, xxx]
    # pin_ = {'t':3, 'd':0, 'X':passCube2}
    # pTraj.addPin(pin_)

    for i, wp in enumerate(waypoints):

        pin_ = {'t':time_between_gates+i*time_between_gates, 'd':0, 'X':np.array(wp)}
        # pTraj.addPin(pin_)
        passCube = np.array([wp[0], wp[1], wp[2]]) # in shape [dim, xxx]
        pin_ = {'t':knots[i+1], 'd':0, 'X':passCube}
        print(pin_)

        pTraj.addPin(pin_)

        # break
    print(np.array([waypoints[-1][0]+10, waypoints[-1][1], waypoints[-1][2]]))
    # solve
    pTraj.setDerivativeObj(objWeights)
    print("solving")
    time_start = time.time()
    pTraj.solve()
    time_end = time.time()
    print(time_end - time_start)
    rng = np.linspace(0, (2+len(waypoints))*time_between_gates, int((2+len(waypoints))*time_between_gates//tdelta))
    velocities = pTraj.eval(rng, 1)
    path = pTraj.eval(rng, 0)

    print(path)
    print(velocities)
    print('path')
    if not silent: 
        fig_title ='poly order : {0} / max continuity: {1} / minimized derivatives order: {2}'.format(pTraj.N, pTraj.maxContiOrder, np.where(pTraj.weight_mask>0)[0].tolist())
        pTraj.showPath(fig_title)
    return path, velocities
