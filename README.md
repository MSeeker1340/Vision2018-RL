# Vision2018-RL
Final course project for CSCI-GA 3033: Vision meets machine learning
Instructions:
https://d1b10bmlvqabco.cloudfront.net/attach/jm0yvg6k8c072i/j7ay8cyggyt4yk/jp7lj9s8p2ew/Reinforcement_Learning.11232018.pdf

## Homework by 12/6
### Task 1: Skeleton coordinate transformation
Input:

	coordinate(v_i) 

Output:

	(X, Y, Z)_i

### Task 2: Compute action

Input:

	S^t ={s_i^t}, i = 1 to 25
      s_i^t = (v_i, d_i)^t
	      where v_i^t=(X_i, Y_i, Z_i)^t
	            d_i^t = v_i^{t} - v_i^{t-1}
  


Output:

	A^t ={a_i^t}, i = 1 to 25
      a_i^t = (a_i^x, a_i^y, a_i^z)^t
	      where a_i^t = d_i^{t+1}- d_i^{t}


