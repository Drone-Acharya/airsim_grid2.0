import sys
import os
import contextlib
import math
import numpy as np
import time
from collections import deque
import airsim

from traj_gen_sim import get_trajectory
from trajectory_follower import fly_with_piecewise_control, format_path, format_path2, get_dummy_waypoints
import json
import random

class Trajectory():
	def __init__(self, traj_len = 5):
		self.traj_len = traj_len
		self.dim = 1       # dim = 1 for each wp(Y(East) coordinate for each waypoint)
		self.extra_dim = 1 # extra dim for ts
		self.set_env()

	def set_env(self):
		waypoints, gate_l, gate_h, obspaths = get_dummy_waypoints()
		self.baseline_waypoints = waypoints
		self.baseline_obspaths = obspaths
		self.baseline_dur = 0.1
		self.gate_l = gate_l
		self.gate_h = gate_h

	def get_weights_dim(self):
		return self.traj_len*self.dim + self.extra_dim

	def set_weights(self, weights):
		self.waypoint_coords_y = weights[:-1]
		self.dur = weights[-1]

	def get_current_waypoints(self):
		current_waypoints = []
		for i in range(self.traj_len):
			current_waypoints.append([self.baseline_waypoints[i][0], self.waypoint_coords_y[i], self.baseline_waypoints[i][2]])
		return current_waypoints

	def print_weights(self):
		if self.dur == None:
			print("Weights Not Initialised")
		else:
			for i in range(self.traj_len):
				print("Coordinates for waypoint:{} --> ({}, {}, {})".format(i, self.baseline_waypoints[i][0], self.waypoint_coords_y[i], self.baseline_waypoints[i][2]))
			print("Time b/w waypoints --> {}".format(self.dur))

	def get_trajectory(self, pos0, debug = False):
		waypoints = self.get_current_waypoints()
		ts = self.dur
		with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
			path, vel = get_trajectory(waypoints, self.gate_l, self.gate_h, pos0, ts, silent = True)
		return path, vel
	
	def evaluate(self, weights, debug = False):
		# NEED FOR A better SCORE function
		self.set_weights(weights)
		if debug:
			self.print_weights()

		# Initial Position
		pos0 = [0, 0, self.baseline_waypoints[0][2]]

		# Run Trajectory Planner
		full_path, full_vel = self.get_trajectory(pos0)

		# Score Calc
		# Baseline Score - Total average forward velocity - Stays Same for some reason
		avg_fwd_vel = (1/full_vel.shape[1])*np.sum(full_vel[0, :])

		# Deviation from mid-
		dev = 0
		maxbound_y = max(self.baseline_waypoints[:][1])
		minbound_y = min(self.baseline_waypoints[:][1])
		for i in range(full_path.shape[1]):
			wp_y = full_path[:, i][1]
			if wp_y > maxbound_y:
				dev += (wp_y - maxbound_y)**2
			elif wp_y < minbound_y:
				dev += (wp_y - minbound_y)**2

		# Window entry point deviation less than threshold from origin
		wdev = 0
		for i in range(len(self.baseline_waypoints)):
			base_y = self.baseline_waypoints[i][1]
			curr_y = self.get_current_waypoints()[i][1]
			if abs(curr_y - base_y) > self.gate_l/8:
				wdev += (curr_y - base_y)**4

		ep_return = -dev-wdev

		return ep_return


def cem(n_iterations= 100, print_every=5, pop_size=10, elite_frac=0.2, sigma=0.5, traj_len = 5):

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

	estimates, noises = get_initialisations(agent)

	# SAVING FOR RECREATING ENV FOR VISUALISATION
	data = {"wp": agent.baseline_waypoints, 
			"ob": agent.baseline_obspaths}
	with open('CEM_data/arrseed.json', 'w') as fp:
		json.dump(data, fp)

	# Initialise
	best_weight = estimates

	for i_iteration in range(1, n_iterations+1):
		weights_pop = [best_weight + (noises*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
		rewards = np.array([agent.evaluate(weights, debug = False) for weights in weights_pop])

		elite_idxs = rewards.argsort()[-n_elite:]
		elite_weights = [weights_pop[i] for i in elite_idxs]
		best_weight = np.array(elite_weights).mean(axis=0)

		reward = agent.evaluate(best_weight, debug = False)
		scores_deque.append(reward)
		scores.append(reward)
		
		best_params = best_weight
		
		if i_iteration % print_every == 0:
			print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
	
		# Save ALL params
		np.save("CEM_data/best_params.npy", np.array(best_params))

	return scores, best_params

def get_initialisations(agent):
	# Get Initial guesses for Best params and initial conditions
	traj_len = agent.traj_len
	waypoints = agent.baseline_waypoints
	dur = agent.baseline_dur
	
	# Initial estimates and noises
	estimates = []
	noises = []
	# Pos
	for i in range(traj_len):
		estimates.append(waypoints[i][1]) 
		noises.append(0.1)
	# Time
	estimates.append(dur)
	noises.append(0.0)

	return np.array(estimates), np.array(noises)


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
		waypoints, l, h, obspaths = get_dummy_waypoints()

	# GET DATA
	traj.set_weights(weights)
	traj.print_weights()
	
	# Moving Drone to initial position
	z = -1 * waypoints[0][2]
	print("make sure we are hovering at {} meters...".format(-z))
	client.moveToPositionAsync(0,0,z,1).join()

	for obs in obspaths:
		path = format_path2(obs)
		client.simPlotLineStrip(path, color_rgba=[0.0, 0.0, 1.0, 1.0], is_persistent = True)

	# Formality measures, all initial values
	k = client.simGetGroundTruthKinematics()
	pos0 = [k.position.x_val, k.position.y_val, -1*k.position.z_val]
	vel0 = [k.linear_velocity.x_val, k.linear_velocity.y_val, -1*k.linear_velocity.z_val]
	acc0 = [k.linear_acceleration.x_val, k.linear_acceleration.y_val, -1*k.linear_acceleration.z_val]

	print("Computing Path ...")
	path, vel = traj.get_trajectory(pos0, debug = True)

	path = format_path(path)

	print("Flying on path ...")
	client.simPlotLineStrip(path, is_persistent = True)

	# Velocity based traversal
	for i in range(vel.shape[1]):
		vx, vy, vz = vel[0][i], vel[1][i], vel[2][i]
		result = client.moveByVelocityAsync(vx, vy, vz, traj.dur).join()

	print("landing...")
	client.landAsync().join()
	print("disarming...")
	client.armDisarm(False)
	client.enableApiControl(False)
	print("done.")

def move_on_base_path():
	
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

	try:
		with open('CEM_data/arrseed.json', 'r') as fp:
			data = json.load(fp)
		waypoints, obspaths = data["wp"], data["ob"]
		print("Baseline Waypoints")
		for wp in waypoints:
			print(wp)
	except:
		print("Env Data Not Found")
		waypoints, l, h, obspaths = get_dummy_waypoints()

	# Moving Drone to initial position
	z = -1 * waypoints[0][2]
	print("make sure we are hovering at {} meters...".format(-z))
	client.moveToPositionAsync(0,0,z,1).join()

	for obs in obspaths:
		path = format_path2(obs)
		client.simPlotLineStrip(path, color_rgba=[0.0, 0.0, 1.0, 1.0], is_persistent = True)

	# Formality measures, all initial values
	k = client.simGetGroundTruthKinematics()
	pos0 = [k.position.x_val, k.position.y_val, -1*k.position.z_val]
	vel0 = [k.linear_velocity.x_val, k.linear_velocity.y_val, -1*k.linear_velocity.z_val]
	acc0 = [k.linear_acceleration.x_val, k.linear_acceleration.y_val, -1*k.linear_acceleration.z_val]

	print("Computing Path ...")
	_, gate_l, gate_h, _ = get_dummy_waypoints()
	gate_length = gate_l
	gate_height = gate_h
	ts = 0.1
	path, vel = get_trajectory(waypoints, gate_length, gate_height, pos0, ts)

	path = format_path(path)

	print("Flying on path ...")
	client.simPlotLineStrip(path, is_persistent = True)

	# Velocity based traversal
	for i in range(vel.shape[1]):
		vx, vy, vz = vel[0][i], vel[1][i], vel[2][i]
		result = client.moveByVelocityAsync(vx, vy, vz, ts).join()

	print("landing...")
	client.landAsync().join()
	print("disarming...")
	client.armDisarm(False)
	client.enableApiControl(False)
	print("done.")

if __name__ == "__main__":
	mode = sys.argv[1]
	if int(mode) == 0:
		scores, best_params = cem()
	elif int(mode) == 1:
		move_on_best_path()
	else:
		move_on_base_path()

