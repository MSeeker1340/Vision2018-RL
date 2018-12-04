def compute_action_by_states(state_t, state_t_plus_1):
	'''
	Input:

	S^t ={s_i^t}, i = 1 to 25
  		s_i^t = (v_i, d_i)^t
      	where v_i^t=(X_i, Y_i, Z_i)^t
              d_i^t = v_i^{t} - v_i^{t-1}

    S^{t+1}

    Output:
	A^t ={a_i^t}, i = 1 to 25
  		a_i^t = (a_i^x, a_i^y, a_i^z)^t
      	where a_i^t = d_i^{t+1} - d_i^{t}
    '''
    a_t = []
    for i in range(len(state_t)):
		current_d = state_t[i][1]
		next_d = state_t_plus_1[i][1]
		a = [next_d[j] - current_d[j] for j in range(len(next_d))]
		a_t.append(a)

	return a_t
