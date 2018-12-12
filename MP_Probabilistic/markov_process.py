import numpy as np
from numpy.linalg import inv


def convert_in_index(elem):
	if elem == "rain" or elem == "umbrella":
		return 0
	return 1

''' This function is needed to normalize the probabilities distribution computed in different functions '''

def normalize(prob):    #ax+ay = 1    a = 1/(x+y)
	alpha = float(1)/(prob[0]+prob[1])
	p = list(prob)
	p[0] = p[0]*alpha
	p[1] = p[1]*alpha
	return p

'''This function execute the filtering operation at time t. T is the matrix containing the probabilities of the transition model. O is the matrix containing the
   probabilities of the observation model. f_old is the prior probability of the transition model at t == 0
   the algorithm works in the following way
   - if t == 0 then I am at the start of the sequence and i have to return P(X0) so I return prior
   - else i make the recursive call decrementing the t index and when the recursion ends i have the forward message from the past and i can compute the new one
   and return it'''

def filtering(T,O,obs,prior,t):
	if t == 0:
		return prior
	else:
		f = filtering(T,O,obs,prior,t-1)
		somma = 0
		for i in range(0,len(T)):
			somma+= T[i]*f[i]
		res = O[:,convert_in_index(obs[t][1])]*somma
		return normalize(res)

''' This is the iterative implementation of the filtering function '''

def filtering_iterative(T,O,obs,prior,t):
	f = prior
	for j in range(1,t+1):
		somma = 0
		for i in range(0,len(T)):
			somma+= T[i]*f[i]
		res = O[:,convert_in_index(obs[j][1])]*somma
		f = normalize(res)
	return f 

'''This algorithms is neede to compute the backward messages. back_m the message from the future and initialzed to [1,1]. n is the point from
   where we start the process of backwarding(generally from the last observation). t is the time at what point we are computing the smoothing.
   O is the observation model and T is the transition model. obs is the sequence of observation.
   - if n == t then i am arrived to the maximum future and return the back_m
   - Otherwise i compute the backward message at time n and I return the recursive call passing at the past the backward message just computed'''

def backwards(n,t,O,T,obs,back_m):
	if n == t:
		return back_m
	back = 0
	for i in range(0,len(T)):
		back+=O[i][convert_in_index(obs[n][1])]*back_m[i]*T[i]
	return backwards(n-1,t,O,T,obs,back)

''' This algorithm is the one needed for computing the smoothing probabilities at a given time k. O is the observation model and T is the transition
	model. obs is the sequence of observation and k is the time at when we compute the smoothing and n is the time when we had the last observation in the
	future. I compute the forward message untill time k and compute the backward message at time k+1 and retunr the normalized pointwise product between
	the 2 '''

def smoothing(O,T,obs,k,n):
	f_1_k = filtering(T,O,obs,np.array([0.5,0.5]),k)
	b_k1_t = backwards(n,k,O,T,obs,np.array([1.0,1.0]))
	return normalize(f_1_k*b_k1_t)

''' This function compute the smoothing of a complete sequence obs. REMEMBER THAT INDEXES OF TIME GO FROM 0 TO LEN(OBS)-1.
	For every t from 0 to len(obs) - 1 i simply use the smoothing function and every time i change the parameter k of the function because i have to
	do smoothing for a differente time index.
	Quite inefficient because at every iteration it computes the forward message from the start and also the backward message '''

def smoothing_sequence(obs,O,T):
	smoothings_ = list()
	for i in range(0,len(obs)):
		smoothings_.append(smoothing(O,T,obs,i,len(obs)-1))
	return smoothings_

''' This function is used in smoothing_sequence_different. It's very similar to the computation of backward message function.
	filterings is the list of all the filterings at every time step n and back_m is the backward message from n+1.
	When the function is called for the first time n is equal to the last time at which we have an observation, len(obs)-1, and back_m is an array of 1s.
	If n == 0 then I am at the start of the sequence and i return a list containing the smoothing at time 0 using the the right filtering message and
	the back from n+1
	Else I compute the back message at time n for n-1 and i make the recursive call passing this information. when the recursion ends, the results is the list
	containing all the smoothing values fro t < n and i can compute the smoothing value for time n, using filtering[n] and append this value to the list.
	Finally I can return this modified list'''

def smoothing_sequence_rec(O,T,obs,n,filterings,back_m):
	if n == 0:
		return [normalize(filterings[n]*back_m)]
	else:
		back = 0
		for i in range(0,len(T)):
			back+=O[i][convert_in_index(obs[n][1])]*back_m[i]*T[i]				# I compute the new backward message using the information from the future
		res = smoothing_sequence_rec(O,T,obs,n-1,filterings,back)
		sm = normalize(filterings[n]*back_m)
		res.append(sm)
		return res

'''This function is used in smoothing a sequence to compute all the filtering messages needed in the complete sequence. It's very similar to the filtering
   algorithm previously defined but it returns a list containing all the filtering messages at every k-th time. So at every k-th i compute the current forward
   message and I do the recursive call. When the call ends it returns the list containing all the forward message from the future and i add at the head of this 
   list the current forward message and return this new list.
   The parameters of the function are the same as the filtering algorithm'''

def filtering_sequence_rec(T,O,obs,prior,t):
	if t == 0:
		return [prior]
	else:
		f = filtering_sequence_rec(T,O,obs,prior,t-1)
		somma = 0
		for i in range(0,len(T)):
			somma+= T[i]*f[t-1][i]
		res = normalize(O[:,convert_in_index(obs[t][1])]*somma)
		f.append(res)
		return f

''' I am still computing the smoothing of a complete sequence obs but i try a more efficient technique. I first compute all the filtering messages and put them
	in a list. Then i call a recursive function that computes the backward message from len(obs) - 1 untill 1 and at every step it also computes the smoothing
	of a temporal index.
	I am using the filtering_sequence_rec function in order to compute the list of the filtering messages needed in am ore efficient way. If i simply do a for
	loop where I compute the messages then i restart every time from t = 0 and recompute all the previous messages'''

def smoothing_sequence_different(obs,O,T):
	filterings = filtering_sequence_rec(T,O,obs,np.array([0.5,0.5]),len(obs)-1)
	return smoothing_sequence_rec(O,T,obs,len(obs)-1,filterings,np.array([1.0,1.0]))

''' This algorithm implements the matrix algorithm for HMM. We use the matrix T which is the Transition Model, O_true and O_false are the sensor model diagonal
	matrixes build based on the value of the perception received at time k-th. Then the current forward message is computed multiplying in the right order the
	transition model matrix the sensor model matrix and the old forward message and the rest is equal to the filtering algorithm previously developed'''

def filtering_matrix_algorithm(k,T,O,obs,f_old,t):
	if k == t+1:
		return f_old
	else:
		if k == 0:
			return filtering_matrix_algorithm(k+1,T,O,obs,f_old,t)
		if obs[k][1] == "umbrella":
			O_true = np.array([[O[0][0],0.0],
				   			   [0.0,O[1][0]]])
			res = O_true.dot((T.T).dot(f_old))
			return filtering_matrix_algorithm(k+1,T,O,obs,normalize(res),t)
		O_false = np.array([[O[0][1],0.0],
				   			[0.0,O[1][1]]])
		res = O_false.dot((T.T).dot(f_old))
		return filtering_matrix_algorithm(k+1,T,O,obs,normalize(res),t)

''' This function computes the smoothing sequences from observation using the matrix algorithm for HMM. When the function is called for the first time filtering_m
	is equal to the filtering at the last time index possible which is len(obs)-1 and back_m is a vectore of 1s. At every recursive call, instead, filtering_m
	is the forward message for time index t and back_m is the backward message from t+1
	If t == 0 then I am at the start of the time sequence and return a list containing the smoothing of index t = 0 
	Else i divide the case of evidence being true or false because i need a different diagonal matrix for the observation model and i do:
	- I compute the backward message for time t using the matrix operations
	- I compute the forward message at time t-1 from the filtering_m of the future using the inverse of the matrix equation
	- I call recursevely passing these informations to the past
	- The result of the recursion call is a list containing all the smoothing up to t-1 and I compute the smoothing value for t using the filtering_m and
	back_m which are the correct parameters to use at time t and then i add the smoothing value to the list and then i return this list modified'''

def smoothing_matrix_rec(T,O,obs,filtering_m,back_m,t):
	if t == 0:
		return [normalize(filtering_m*back_m)]
	else:
		if obs[t][1] == "umbrella":
			O_true = np.array([[O[0][0],0.0],
				   			   [0.0,O[1][0]]])
			back = T.dot(O_true.dot(back_m))
			f_old = normalize(inv((T.T)).dot(inv(O_true).dot(filtering_m)))
			res = smoothing_matrix_rec(T,O,obs,f_old,back,t-1)
			smoothing = normalize(filtering_m*back_m)
			res.append(smoothing)
			return res
		O_false = np.array([[O[0][1],0.0],
				   		   [0.0,O[1][1]]])
		back = T.dot(O_false.dot(back_m))
		f_old = normalize(inv((T.T)).dot(inv(O_false).dot(filtering_m)))
		res = smoothing_matrix_rec(T,O,obs,f_old,back,t-1)
		smoothing = normalize(filtering_m*back_m)
		res.append(smoothing)
		return res

''' This function compute the smoothings of a observation sequence using the matrix algorithms for the HMM. First i compute the filtering of the last timw index
	of the observation which is len(obs)-1 and then i call the auxiliar function smoothing_matrix_rec which computes the list of all smoothing values for
	time indexes from 0 to len(obs)-1 '''

def smoothing_matrix_algorithm(obs,O,T):
	last_filtering = filtering_matrix_algorithm(0,T,O,obs,np.array([0.5,0.5]),len(obs)-1)
	return smoothing_matrix_rec(T,O,obs,last_filtering,np.array([1.0,1.0]),len(obs)-1)