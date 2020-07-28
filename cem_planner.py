import math
import numpy as np
import time
from collections import deque
import airsim

from min_jerk_uzh import get_path
from tqdm import tqdm
import json
import random

class Trajectory():
	def __init__(self, traj_len = 5):
		self.traj_len = traj_len
		self.dim_var = 2
		self.waypoint_coords = None
		self.waypoint_velocities = None
		self.set_env()

	def get_weights_dim(self):
		return self.traj_len*7 # ({2-pos, 2-vel, 2-acc}(x,y), 1-time)

	def set_weights(self, weights):
		self.waypoint_coords = [[weights[i], weights[i+1]] for i in range(0,len(weights),3*self.dim_var + 1)]
		self.waypoint_velocities = [[weights[i], weights[i+1]] for i in range(2,len(weights),3*self.dim_var + 1)]
		self.waypoint_acceleration = [[weights[i], weights[i+1]] for i in range(4,len(weights),3*self.dim_var + 1)]
		self.waypoint_dur = [weights[i] for i in range(6,len(weights),3*self.dim_var + 1)]

	def print_weights(self):
		if not self.waypoint_velocities or not self.waypoint_coords:
			print("Weights Not Initialised")
		else:
			for i in range(self.traj_len):
				print("Coordinates for waypoint:{} --> ({}, {}, Height)".format(i, self.waypoint_coords[i][0], self.waypoint_coords[i][1]))
				print("Velocities for waypoint:{} --> ({}, {}, 0)".format(i, self.waypoint_velocities[i][0], self.waypoint_velocities[i][1]))
				print("Acceleration for waypoint:{} --> ({}, {}, 0)".format(i, self.waypoint_acceleration[i][0], self.waypoint_acceleration[i][1]))
				print("Time to reach waypoint:{} --> {}".format(i, self.waypoint_dur[i]))

	def get_trajectory(self, start_x, start_v, start_a, debug = False):
		full_path = []
		full_vel = []
		full_time = []
		inputFeasibility = []
		posFeasibility = []

		# print("=====================")
		for i in range(self.traj_len):
			pos0 = self.waypoint_coords[i-1] + [start_x[2]] if i>0 else start_x
			vel0 = self.waypoint_velocities[i-1] + [0] if i>0 else start_v
			acc0 = self.waypoint_acceleration[i-1] + [0] if i>0 else start_a
			posf = self.waypoint_coords[i] + [start_x[2]]
			velf = self.waypoint_velocities[i] + [0]
			accf = self.waypoint_acceleration[i] + [0]
			Tf = self.waypoint_dur[i]

			# print("------------")
			# print("Pos {}: {}".format(i,posf))
			# print("Vel {}: {}".format(i,velf))
			# print("Acc {}: {}".format(i,accf))

			piecewise_path, piecewise_vel, piecewise_acc, piecewise_time, inF, pF = get_path(pos0, vel0, acc0, posf, velf, accf, Tf, debug)
			full_path.extend(piecewise_path)
			full_vel.extend(piecewise_vel)
			full_time.extend(piecewise_time)
			inputFeasibility.append(inF)
			posFeasibility.append(pF)

		return full_path, full_vel, inputFeasibility, posFeasibility, full_time

	def set_env(self):
		waypoints, obspaths, y = get_dummy_waypoints()
		self.baseline_waypoints = waypoints
		self.baseline_obspaths = obspaths
		self.yn = y
	
	def evaluate(self, weights, height):

		# NEED FOR A better score function
		self.set_weights(weights)
		start_x = [0, 0, height]
		start_v = [0, 0, 0]
		start_a = [0, 0, 0]

		full_path, full_vel, inF, pF, full_time = self.get_trajectory(start_x, start_v, start_a)

		avg_vel = (1/len(full_vel))*np.sum(np.array(full_vel), axis = 0)
		min_vel = np.min(np.array(full_vel), axis = 0)[0]
		ep_return = np.int32((np.array(inF) + np.array(pF)) == 0)

		min_yvel = np.min(np.array(full_vel), axis = 0)[1]
		max_yvel = np.max(np.array(full_vel), axis = 0)[1]
		yvellim = max(max_yvel, -min_yvel)

		baseline_waypoints = self.baseline_waypoints
		obspaths = self.baseline_obspaths
		y = self.yn

		maxbound = np.max(np.array(baseline_waypoints)[:, 1])
		minbound = np.min(np.array(baseline_waypoints)[:, 1])
		yp = np.array(full_path)[:, 1]
		dev = np.maximum(np.abs(yp-maxbound), np.abs(yp-minbound))
		dev = dev*(np.where((yp < maxbound) & (yp > minbound), False, True))
		deviation = np.mean(dev)

		within_threshold = np.int32(np.abs(np.array(self.waypoint_coords)[:, 1] - np.array(baseline_waypoints)[:, 1])<y)
		# print("Constraint 1(Waypoints): {}".format(within_threshold))
		# print("Constraint 2(Total Time): {}".format(np.mean(np.array(self.waypoint_dur))))
		# print("Constraint 3(Feasibility): {}".format(ep_return))
		# print("Score 1: {}".format(np.min(ep_return * within_threshold)))
		# print(min_vel)

		ep_return = 100*(np.min(ep_return * within_threshold)) - deviation + 10*(min_vel)
		# print("Total Score: {}".format(ep_return))
		return ep_return


def cem(n_iterations=100, print_every=5, pop_size=10, elite_frac=0.2, sigma=0.5, traj_len = 5):

	best_params = None

	"""    
	Params
	======
		n_iterations (int): maximum number of training iterations
		max_t (int): maximum number of timesteps per episode
		gamma (float): discount rate
		print_every (int): how often to print average score (over last 100 episodes)
		pop_size (int): size of population at each iteration
		elite_frac (float): percentage of top performers to use in update
		sigma (float): standard deviation of additive noise
	"""
	agent = Trajectory(traj_len = traj_len)

	n_elite=int(pop_size*elite_frac)

	scores_deque = deque(maxlen=100)
	scores = []

	estimates, noises, height = get_initialisations(agent)
	data = {"wp": agent.baseline_waypoints, 
			"ob": agent.baseline_obspaths,
			"yn": agent.yn}
	with open('CEM_data/arrseed.json', 'w') as fp:
		json.dump(data, fp)

	best_weight = estimates

	for i_iteration in tqdm(range(1, n_iterations+1)):
		weights_pop = [best_weight + (noises*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
		rewards = np.array([agent.evaluate(weights, height) for weights in weights_pop])

		elite_idxs = rewards.argsort()[-n_elite:]
		elite_weights = [weights_pop[i] for i in elite_idxs]
		best_weight = np.array(elite_weights).mean(axis=0)

		reward = agent.evaluate(best_weight, height=5.0)
		scores_deque.append(reward)
		scores.append(reward)
		
		best_params = best_weight
		
		if i_iteration % print_every == 0:
			print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

		# if np.mean(scores_deque)>=90.0:
		# 	print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
		# 	break
	
		# Save ALL
		np.save("CEM_data/best_params.npy", np.array(best_params))

	return scores, best_params

def get_dummy_waypoints():
	n_gates = 5
	width = 5
	d_bw_gates = 1*4
	gate_stand = 3*2
	gate_height = 0.55*2
	gate_length = 1*4
	ynoise = gate_length/4

	# Build Dummy Environment for Visualisation
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

	all_obspaths = gate_stands + gate_l_up + gate_l_down + gate_h_left + gate_h_right
	return waypoints, all_obspaths, ynoise

def get_initialisations(agent):
	# Get Initial guesses for Best params and initial conditions
	traj_len = agent.traj_len
	waypoints = agent.baseline_waypoints
	obstacles = agent.baseline_obspaths
	ynoise = agent.yn
	
	# Since Height remains constant
	height = waypoints[0][2]

	# Initial estimates and noises
	estimates = []
	noises = []

	for i in range(traj_len):
		estimates.extend([waypoints[i][0], waypoints[i][1], 0.5, 0, 0, 0, 10]) 
		noises.extend([0, 0.01*ynoise, 0.5, 0.5, 0, 0, 0.1])
		# noises.extend([0]*7)

	return np.array(estimates), np.array(noises), height

def format_path(wp):
	path = []
	for pt in wp:
		x = pt[0]
		y = pt[1]
		z = -1*pt[2]
		path.append(airsim.Vector3r(x,y,z))
	return path

def move_on_best_path():
	
	client = airsim.MultirotorClient()
	client.confirmConnection()
	client.enableApiControl(True)
	client.simFlushPersistentMarkers()

	print("arming the drone...")
	client.armDisarm(True)

	state = client.getMultirotorState()
	if state.landed_state == airsim.LandedState.Landed:
		print("taking off...")
		client.takeoffAsync().join()
	else:
		client.hoverAsync().join()

	time.sleep(1)

	state = client.getMultirotorState()
	if state.landed_state == airsim.LandedState.Landed:
		print("take off failed...")
		sys.exit(1)

	traj_len = 5
	traj = Trajectory(traj_len)

	try: 
		weights = list(np.load("CEM_data/best_params.npy"))
		print("Weights: ")
		for w in weights:
			print(w)
	except:
		print("Params not Found")
		weights = list(np.random.randn(traj.get_weights_dim()))
	try:
		with open('CEM_data/arrseed.json', 'r') as fp:
			data = json.load(fp)
		waypoints, obspaths = data["wp"], data["ob"]
		print("Baseline Waypoints")
		for wp in waypoints:
			print(wp)
	except:
		print("Env Data Not Found")
		waypoints, obspaths, _ = get_dummy_waypoints()

	traj.set_weights(weights)
	
	# Moving Drone to initial position
	z = -waypoints[0][2]
	print("make sure we are hovering at {} meters...".format(-z))
	client.moveToPositionAsync(0,0,z,1).join()

	for obs in obspaths:
		path = format_path(obs)
		client.simPlotLineStrip(path, color_rgba=[0.0, 0.0, 1.0, 1.0], is_persistent = True)

	# Formality measures, all initial values
	k = client.simGetGroundTruthKinematics()
	pos0 = [k.position.x_val, k.position.y_val, -1*k.position.z_val]
	vel0 = [k.linear_velocity.x_val, k.linear_velocity.y_val, -1*k.linear_velocity.z_val]
	acc0 = [k.linear_acceleration.x_val, k.linear_acceleration.y_val, -1*k.linear_acceleration.z_val]

	full_path, full_vel, inputFeasibility, posFeasibility, full_time = traj.get_trajectory(pos0, vel0, acc0, debug = True)
	path = format_path(full_path)

	print("flying on path...")
	client.simPlotLineStrip(path, is_persistent = True)
	duration = full_time[1] - full_time[0]

	# Velocity based traversal
	for i in range(len(full_vel)):
		vx, vy, vz = full_vel[i][0], full_vel[i][1], full_vel[i][2]
		result = client.moveByVelocityAsync(vx, vy, -1*vz, duration).join()

	print("landing...")
	client.landAsync().join()
	print("disarming...")
	client.armDisarm(False)
	client.enableApiControl(False)
	print("done.")

import sys

mode = sys.argv[1]
print(mode)
if int(mode) == 0:
	scores, best_params = cem()
else:
	move_on_best_path()

