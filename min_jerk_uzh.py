from __future__ import print_function, division

import sys
sys.path.append("../minimum_jerk_trajectories/Python/")
import quadrocoptertrajectory as quadtraj

# Define the trajectory starting state:
pos0 = [0, 0, 2] #position
vel0 = [0, 0, 0] #velocity
acc0 = [0, 0, 0] #acceleration

# Define the goal state:
posf = [1, 0, 1]  # position
velf = [0, 0, 1]  # velocity
accf = [0, 9.81, 0]  # acceleration

def get_path(pos0, vel0, acc0, posf, velf, accf, Tf = None, debug = False):
	# Define the duration:
	if Tf is None:
		Tf = 10

	# Define the input limits:
	fmin = 5  #[m/s**2]
	fmax = 50 #[m/s**2]
	wmax = 20 #[rad/s]
	minTimeSec = 0.02 #[s]

	# Define how gravity lies:
	gravity = [0,0,-9.81]
	 
	traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, gravity)
	traj.set_goal_position(posf)
	traj.set_goal_velocity(velf)
	traj.set_goal_acceleration(accf)

	# Run the algorithm, and generate the trajectory.
	traj.generate(Tf)

	# Test input feasibility
	inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, minTimeSec)

	# Test whether we fly into the floor
	floorPoint  = [0,0,0]  # a point on the floor
	floorNormal = [0,0,1]  # we want to be in this direction of the point (upwards)
	positionFeasible = traj.check_position_feasibility(floorPoint, floorNormal)
	 
	if debug:
		print("Input feasibility result: ",    quadtraj.InputFeasibilityResult.to_string(inputsFeasible),   "(", inputsFeasible, ")")
		print("Position feasibility result: ", quadtraj.StateFeasibilityResult.to_string(positionFeasible), "(", positionFeasible, ")")

	import numpy as np
	numPlotPoints = 100
	time = np.linspace(0, Tf, numPlotPoints)
	position = np.zeros([numPlotPoints, 3])
	velocity = np.zeros([numPlotPoints, 3])
	acceleration = np.zeros([numPlotPoints, 3])
	thrust = np.zeros([numPlotPoints, 1])
	ratesMagn = np.zeros([numPlotPoints,1])


	for i in range(numPlotPoints):
	    t = time[i]
	    position[i, :] = traj.get_position(t)
	    velocity[i, :] = traj.get_velocity(t)
	    acceleration[i, :] = traj.get_acceleration(t)
	    thrust[i] = traj.get_thrust(t)
	    ratesMagn[i] = np.linalg.norm(traj.get_body_rates(t))

	return position, velocity, acceleration, time, inputsFeasible, positionFeasible