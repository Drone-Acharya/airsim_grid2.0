import airsim

import sys
import time
import numpy as np

def fly():
	client = airsim.MultirotorClient()
	client.confirmConnection()
	client.enableApiControl(True)

	# client.simFlushPersistentMarkers()

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
	z = -20
	print("make sure we are hovering at 20 meters...")
	client.moveToZAsync(z, 1).join()

	print("Computing Waypoints...")
	waypoints = None #get_path_over_all_obstacles(client)

	print("Computing Path...")
	arr_path, arr_vel, duration = get_trajectory(waypoints)
	
	path = format_path(arr_path)

	print("flying on path...")
	client.simPlotLineStrip(path, is_persistent = True)

	# print(path)
	# result = client.moveOnPathAsync(path,
	#                         5, 120,
	#                         airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 20, 1).join()
	result = client.moveOnPathAsync(path, 5).join()

	# CHANGEEEEE PLEASE - Velocity based traversal
	# for i in range(len(arr_vel)):
	# 	vx, vy, vz = arr_vel[i][0], arr_vel[i][1], arr_vel[i][2]
	# 	result = client.moveByVelocityAsync(vx, vy, vz, duration).join()

	# drone will over-shoot so we bring it back to the start point before landing.
	client.moveToPositionAsync(0,0,z,1).join()
	print("landing...")
	client.landAsync().join()
	print("disarming...")
	client.armDisarm(False)
	client.enableApiControl(False)
	print("done.")


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
	waypoints = get_dummy_waypoints(client)

	for i in range(len(waypoints)):
		# Initial
		k = client.simGetGroundTruthKinematics()
		pos0 = [k.position.x_val, k.position.y_val, -1*k.position.z_val]
		vel0 = [k.linear_velocity.x_val, k.linear_velocity.y_val, -1*k.linear_velocity.z_val]
		acc0 = [k.linear_acceleration.x_val, k.linear_acceleration.y_val, -1*k.linear_acceleration.z_val]
		
		print("Computing Path {} ...".format(i))
		posf = waypoints[i] # May need to change
		velf = [0.5, 0, 0]
		accf = [None, None, None]

		piecewise_path, piecewise_vel, piecewise_acc, piecewise_time = get_path(pos0, vel0, acc0, posf, velf, accf)
		duration = piecewise_time[1] - piecewise_time[0]
	
		path = format_path(piecewise_path)

		print("Flying on path {} ...".format(i))
		client.simPlotLineStrip(path, is_persistent = True)

		# Velocity based traversal
		for i in range(len(piecewise_vel)):
			vx, vy, vz = piecewise_vel[i][0], piecewise_vel[i][1], piecewise_vel[i][2]
			result = client.moveByVelocityAsync(vx, vy, -1*vz, duration).join()

	# drone will over-shoot so we bring it back to the start point before landing.
	client.moveToPositionAsync(0,0,z,1).join()
	print("landing...")
	client.landAsync().join()
	print("disarming...")
	client.armDisarm(False)
	client.enableApiControl(False)
	print("done.")


from min_jerk_uzh import get_path

def format_path(wp):
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
		path = format_path(obspath)
		client.simPlotLineStrip(path, color_rgba=[0.0, 0.0, 1.0, 1.0], is_persistent = True)

	return waypoints


def get_trajectory(waypoint_arr = None):

	# Define Waypoint Array
	if waypoint_arr is None:
		waypoint_arr = []
		waypoint_arr.append([0, 0, 20])
		waypoint_arr.append([10, 0, 20])
		waypoint_arr.append([10, 10, 20])
		waypoint_arr.append([0, 10, 20])
		waypoint_arr.append([0, 0, 20])


	# Define Velocity Array, all subsequent points have fixed velocity towards x-axis(?)
	velocity_arr = []
	velocity_arr.append([0, 0, 0])
	for _ in range(len(waypoint_arr)-1):
		velocity_arr.append([20, 0, 0])
	
	# Acceleration Array, zero at start, zero at end
	acceleration_arr = []
	acceleration_arr.append([0, 0, 0])
	for _ in range(len(waypoint_arr)-2):
		acceleration_arr.append([0, 0, 0])
	acceleration_arr.append([0, 0, 0])

	full_path = []
	full_vel = []

	for i in range(len(waypoint_arr)-1):
		pos0 = waypoint_arr[i]
		vel0 = velocity_arr[i]
		acc0 = acceleration_arr[i]
		posf = waypoint_arr[i+1]
		velf = velocity_arr[i+1]
		accf = acceleration_arr[i+1]

		piecewise_path, piecewise_vel, piecewise_acc, piecewise_time = get_path(pos0, vel0, acc0, posf, velf, accf)
		full_path.extend(piecewise_path)
		full_vel.extend(piecewise_vel)

	delta_T = piecewise_time[1] - piecewise_time[0]	

	neg_path = []
	for pathpt in full_path:
		neg_path.append([pathpt[0], pathpt[1], -1*pathpt[2]])

	return neg_path, full_vel, delta_T	


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
