import numpy as np
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
	
