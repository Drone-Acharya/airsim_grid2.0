import airsim

import sys
import time
import numpy as np

def fly_with_piecewise_control():

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
	
	# AirSim uses NED coordinates so negative axis is up.
	# z of -15 is 15 meters above the original launch point.
	z = -6
	print("make sure we are hovering at 3 meters...")
	client.moveToZAsync(z, 1).join()

	print("Computing Waypoints...")
	# PATH REQUIRED IN THIS CASE
	# waypoints = get_path_over_all_obstacles(client, fast = True)
	waypoints, gate_length, gate_height = get_dummy_waypoints(client)

		# Initial
	k = client.simGetGroundTruthKinematics()
	pos0 = [k.position.x_val, k.position.y_val, -k.position.z_val]
	vel0 = [k.linear_velocity.x_val, k.linear_velocity.y_val, k.linear_velocity.z_val]
	acc0 = [k.linear_acceleration.x_val, k.linear_acceleration.y_val, k.linear_acceleration.z_val]
	ts = 0.1
	print("Computing Path ...")
	path, vel = get_trajectory(waypoints, gate_length, gate_height, pos0, ts)

	path = format_path(path)

	print("Flying on path ...")
	client.simPlotLineStrip(path, is_persistent = True)

	print(vel.shape)
	# Velocity based traversal
	for i in range(vel.shape[1]):
		vx, vy, vz = vel[0][i], vel[1][i], vel[2][i]
		result = client.moveByVelocityAsync(vx, vy, vz, ts).join()
		print(i)

	print("landing...")
	client.landAsync().join()
	print("disarming...")
	client.armDisarm(False)
	client.enableApiControl(False)
	print("done.")

from traj_gen_sim import get_trajectory

def format_path(wp):
	path = []
	
	for i in range(wp.shape[1]):
		pt = wp[:, i]
		x = pt[0]
		y = pt[1]
		z = -pt[2]
		path.append(airsim.Vector3r(x,y,z))
	return path


def format_path2(wp):
	path = []
	for pt in wp:
		x = pt[0]
		y = pt[1]
		z = -1*pt[2]
		path.append(airsim.Vector3r(x,y,z))
	return path

def get_dummy_waypoints(client):
	n_gates = 5
	width = 5
	d_bw_gates = 1*4
	gate_stand = 3*2
	gate_height = 0.55*2
	gate_length = 1*4

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

	all_obspaths = gate_stands + gate_l_up + gate_l_down + gate_h_left + gate_h_right
	for obspath in all_obspaths:
		path = format_path2(obspath)
		client.simPlotLineStrip(path, color_rgba=[0.0, 0.0, 1.0, 1.0], is_persistent = True)

	print(waypoints)

	return waypoints, gate_length, gate_height

def get_path_over_all_obstacles(client, fast = False):
	
	# Return Square for debugging
	if fast:
		waypoint_arr = []
		waypoint_arr.append([0, 0, 20])
		waypoint_arr.append([10, 0, 20])
		waypoint_arr.append([10, 10, 20])
		waypoint_arr.append([0, 10, 20])
		waypoint_arr.append([0, 0, 20])
		return waypoint_arr

	# Get All objects
	objects = client.simListSceneObjects()

	quad_pos = client.simGetVehiclePose()
	quad_x, quad_y, quad_z = quad_pos.position.x_val, quad_pos.position.y_val, quad_pos.position.z_val
	
	objects = ['Cone_5', 'Cylinder2', 'Cylinder3', 'Cylinder4', 'Cylinder5', 'Cylinder6', 'Cylinder7', 'Cylinder8', 'Cylinder_2', 'OrangeBall']
	
	waypoints = []

	for obj in objects:
		obj_pos = client.simGetObjectPose(obj)
		obj_x, obj_y, obj_z = obj_pos.position.x_val, obj_pos.position.y_val, obj_pos.position.z_val
		waypoints.append([obj_x, obj_y, quad_z])
	
	return waypoints

fly_with_piecewise_control()
