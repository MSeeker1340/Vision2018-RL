'''
    module preprocessing

The preprocessing functions act on a `bodyinfo` object returned from
`io.read_skeleton_file`. `transform_bodies` computes the body rigid coordinate
system and add to the joints dataframe the transformed coordinates ('v1', 'v2'
and 'v3'). `add_displacements_and_actions` computes the displacements ('d1',
'd2', 'd3') and actions ('a1', 'a2', 'a3') between frames of the transformed
coordinates. `preprocess` binds toghether these operations.
'''
import numpy as np

def preprocess(bodyinfo):
    transform_bodies(bodyinfo)
    add_displacements_and_actions(bodyinfo)

# Coordinate transformations
def coordinate_base(neck, left_hip, right_hip):
    u1 = right_hip - neck
    u1 /= np.linalg.norm(u1)
    u2 = left_hip - neck
    u3 = np.cross(u1, u2)
    u3 /= np.linalg.norm(u3)
    u2 = np.cross(u1, u3)
    return np.vstack((u1, u2, u3))

def scaling_factor(neck, right_hip):
    return np.linalg.norm(right_hip - neck)

def body_to_coordinates(body):
    return np.vstack((body.joints.x, body.joints.y, body.joints.z)).T

def transform_coordinate(coordinates, neck, left_hip, right_hip):
    sf = scaling_factor(neck, right_hip)
    base = coordinate_base(neck, left_hip, right_hip)/sf
    return np.matmul(coordinates-neck, base.T)

def transform_body(body):
    coordinates = body_to_coordinates(body)
    neck = coordinates[2]
    left_hip = coordinates[12]
    right_hip = coordinates[16]
    transformed = transform_coordinate(coordinates, neck, left_hip, right_hip)
    T = transformed.T
    body.joints["v1"] = T[0]
    body.joints["v2"] = T[1]
    body.joints["v3"] = T[2]

def transform_bodies(bodyinfo):
    for i in range(len(bodyinfo)):
        for j in range(len(bodyinfo[i])):
            transform_body(bodyinfo[i][j])

# Displacements and actions
def add_displacements_and_actions(body_info):
	add_displacements(body_info)
	add_actions(body_info)

def add_actions(body_info):
	'''
	Add a_i^t = d_i^{t+1} - d_i^{t} to body_info
	which are three new columns
	'''
	for frame in range(len(body_info)):
		for person in range(len(body_info[frame])):
			if frame == len(body_info) - 1:
				joints_last = body_info[frame][person].joints
				joints_last['a1'] = np.zeros(len(joints_last['d1']))
				joints_last['a2'] = np.zeros(len(joints_last['d1']))
				joints_last['a3'] = np.zeros(len(joints_last['d1']))
			else:
				add_action(body_info[frame+1][person].joints,body_info[frame][person].joints)

def add_action(joints_t_p_1,joints_t):
	'''
	Given d_i^{t+1} and d_i^t
	Add a_i^t = d_i^{t+1} - d_i^{t} to joints_t
	'''
	joints_t['a1'] = joints_t_p_1['d1'] - joints_t['d1']
	joints_t['a2'] = joints_t_p_1['d2'] - joints_t['d2']
	joints_t['a3'] = joints_t_p_1['d3'] - joints_t['d3']

def add_displacements(body_info):
	'''
	Add d_i^t = v_i^{t} - v_i^{t-1} to body_info
	which are three new columns
	'''
	for frame in range(len(body_info)):
		for person in range(len(body_info[frame])):
			if frame == 0:
				joints_0 = body_info[frame][person].joints
				joints_0['d1'] = np.zeros(len(joints_0['v1']))
				joints_0['d2'] = np.zeros(len(joints_0['v1']))
				joints_0['d3'] = np.zeros(len(joints_0['v1']))
			else:
				add_displacement(body_info[frame-1][person].joints,body_info[frame][person].joints)

def add_displacement(joints_t_m_1,joints_t):
	'''
	Given v_i^{t-1} and v_i^t
	Add d_i^t = v_i^{t} - v_i^{t-1} to joints_t
	'''
	joints_t['d1'] = joints_t['v1'] - joints_t_m_1['v1']
	joints_t['d2'] = joints_t['v2'] - joints_t_m_1['v2']
	joints_t['d3'] = joints_t['v3'] - joints_t_m_1['v3']
