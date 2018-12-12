import numpy as np
from numpy.linalg import inv


class Markov_Process:

	def __init__(self,T,O,state_values,obs_values,prior):
		self.T = T 								# This is the transition matrix
		self.O = O 								# This is the observation matrix
		self.state_values = list(state_values)  # all the possible values that the state can have
		self.obs_values = list(obs_values) 		# all the possible values that the observation can have
		self.prior = prior						# prior probability of a state in t = 0 


	''' This Function is needed in order to generate n samples of length l'''

	def generate_samples(self,n,l):
		samples = list()
		for i in range(0,n):
			samples.append(list()) 
		for i in range(0,n):
			for j in range(0,l):
				if j == 0:																		# at time 0 i only have to sample the state and not the obs
					samples[i].append(np.random.choice(self.state_values,1, p = self.prior))
				else:
					obs = list()
					if samples[i][j-1][0] == self.state_values[0]:							# I am sampling the new state with prior based on the value of the parent which is only the previous state
						obs.append(np.random.choice(self.state_values,1, p = self.T[0])[0])
					else:
						obs.append(np.random.choice(self.state_values,1, p = self.T[1])[0])
					if obs[0] == self.state_values[0]:										# I just sampled the current state so now i can sample the current observation using the prior wrt to the values of its parents whic is just the current state_values				
						obs.append(np.random.choice(self.obs_values,1, p = self.O[0])[0])
					else:
						obs.append(np.random.choice(self.obs_values,1, p = self.O[1])[0])
					samples[i].append(obs)
		self.samples = list(samples)

	def get_samples(self):
		return self.samples

	def convert_in_index(self,elem):
		if elem == self.state_values[0] or elem == self.obs_values[0]:
			return 0
		return 1

	''' This function is needed to normalize the probabilities distribution computed in different functions '''

	def normalize(self,prob):    #ax+ay = 1    a = 1/(x+y)
		alpha = float(1)/(prob[0]+prob[1])
		p = list(prob)
		p[0] = p[0]*alpha
		p[1] = p[1]*alpha
		return p

	'''This function execute the filtering operation at time t. T is the matrix containing the probabilities of the transition model. O is the matrix containing the
   	   probabilities of the observation model. prior is the prior probability of the transition model at t == 0
	   This is the iterative implementation of the filtering function '''

	def filtering(self,obs,t):
		f = self.prior
		for j in range(1,t+1):                                          #j is the time index that goes from 1 to t index where i want to compute the real filtering
			somma = 0
			for i in range(0,len(self.T)):								#here i am computing P(Xt+1|z1:t) = sum(P(Xt+1|xt)P(xt|z1:t))
				somma+= self.T[i]*f[i]														 				#xt
			res = self.O[:,self.convert_in_index(obs[j][1])]*somma 	    #here i am finally computing P(Xt+1|z1:t+1) = alpha*P(zt+1|Xt+1)*P(Xt+1|z1:t) and normalizing so the new message is
			f = self.normalize(res)										#ready for the next iteration
		return f 


	'''This algorithms is neede to compute the backward messages. back_m the message from the future and initialzed to [1,1]. n is the point from
   	   where we start the process of backwarding(generally from the last observation). t is the time at what point we are computing the smoothing.
       O is the observation model and T is the transition model. obs is the sequence of observation.'''

	def backwards(self,n,t,obs):
		back_m = np.array([1,1])				#backward message at time n+1
		for j in range(n,t,-1):				#then i go back from n up to t and compute the backward message at every iteration to pass to the past
			if j == 0:
				return back_m
			back = 0
			for i in range(0,len(self.T)):														#bk+1:t = P(Zk+1:t|Xk) = sum(P(zk+1|xk+1)P(zk+2:t|xk+1)P(xk+1|Xk))
				back+=self.O[i][self.convert_in_index(obs[j][1])]*back_m[i]*self.T[i]									#xk+1
			back_m = back
		return back_m

	''' This algorithm is the one needed for computing the smoothing probabilities at a given time k. O is the observation model and T is the transition
		model. obs is the sequence of observation and k is the time at when we compute the smoothing and n is the time when we had the last observation in the
		future. I compute the forward message untill time k and compute the backward message at time k+1 and retunr the normalized pointwise product between
		the 2 
		Time complexity O(t)'''

	def smoothing(self,obs,k,n):
		f_1_k = self.filtering(obs,k)
		b_k1_t = self.backwards(n,k,obs)
		return self.normalize(f_1_k*b_k1_t)

	''' This function is used in smoothing_sequence. It's very similar to the computation of backward message function.
		filterings is the list of all the filterings at every time step n and back_m is the backward message from n+1.
		When the function is called for the first time n is equal to the last time at which we have an observation, len(obs)-1, and back_m is an array of 1s.
		Initially back_m contains the backwards message at n+1 which is an array of ones and result will be the list of all the smoothing values
		At every iteration from n to t-1 going backwards I compute the smoothing values using the correct filtering message at time j and then compute
		the new backwards message for the past which is the next iteration j-1'''

	def smoothing_seq(self,obs,t,n,filterings):
		back_m = np.array([1,1])
		result = list()
		for j in range(n,t-1,-1):
			if j == 0:												#needed to handle special case at j == 0 where I don't have the observation
				s = self.normalize(filterings[j]*back_m)
				result.append(s)
				r = result[::-1]
				return r 
			sm = self.normalize(filterings[j]*back_m)				# I am computing the smoothing value at time j
			result.append(sm)
			back = 0
			for i in range(0,len(self.T)):
				back+=self.O[i][self.convert_in_index(obs[j][1])]*back_m[i]*self.T[i]     # I am computing the new backwards message
			back_m = back
		r = result[::-1]    # i have to invert the order of the list to have the smoothing values in increasing time index
		return r

	'''This function is used in smoothing a sequence to compute all the filtering messages needed in the complete sequence. It's very similar to the filtering
	   algorithm previously defined but it returns a list containing all the filtering messages at every k-th time. When the call ends it returns the list containing all the forward message from the future and i add at the head of this 
	   list the current forward message and return this new list.
	   The parameters of the function are the same as the filtering algorithm'''

	def filtering_seq(self,obs,t):
		f = [self.prior]
		for j in range(1,t+1):
			if j == len(obs):
				return f
			somma = 0
			for i in range(0,len(self.T)):
				somma+= self.T[i]*f[j-1][i]
			res = self.normalize(self.O[:,self.convert_in_index(obs[j][1])]*somma)
			f.append(res)
		return f

	''' I am still computing the smoothing of a complete sequence obs but i try a more efficient technique. I first compute all the filtering messages and put them
		in a list. Then i call a recursive function that computes the backward message from len(obs) - 1 untill 1 and at every step it also computes the smoothing
		of a temporal index.
		I am using the filtering_sequence_rec function in order to compute the list of the filtering messages needed in am ore efficient way. If i simply do a for
		loop where I compute the messages then i restart every time from t = 0 and recompute all the previous messages
		Time complexity O(t)
		Space complexity O(|f|t) with |f| dimension of forward messages'''

	def smoothing_sequence(self,obs,t,fr):
		filterings = self.filtering_seq(obs,fr)
		return self.smoothing_seq(obs,t,fr,filterings)

	''' This algorithm implements the matrix algorithm for HMM. We use the matrix T which is the Transition Model, O_true and O_false are the sensor model diagonal
		matrixes build based on the value of the perception received at time k-th. Then the current forward message is computed multiplying in the right order the
		transition model matrix the sensor model matrix and the old forward message and the rest is equal to the filtering algorithm previously developed'''

	def filtering_matrix_algorithm(self,obs,t):
		f = self.prior
		for j in range(1,t+1):
			if j == len(obs):
				return f
			res = 0
			if obs[j][1] == self.obs_values[0]:
				O_true = np.array([[self.O[0][0],0.0],
								   [0.0,self.O[1][0]]])
				res = O_true.dot((self.T.T).dot(f))								# I am computing f1:t+1 = alpha*Ot+1*transpose(T)*f1:t
			else:
				O_false = np.array([[self.O[0][1],0.0],
									[0.0,self.O[1][1]]])
				res = O_false.dot((self.T.T).dot(f))
			f = self.normalize(res)
		return f

		''' This function computes the smoothing sequences from observation using the matrix algorithm for HMM. When the function is called for the first time f
		is equal to the filtering at the last time index possible which is n and back_m is a vectore of 1s. At every iteration, instead, f
		is the forward message for time index t and back_m is the backward message from t+1 and i iterate backwards.
		If t == 0 then I am at the start of the time sequence and return a list containing the smoothing of index t = 0 
		Else i divide the case of evidence being true or false because i need a different diagonal matrix for the observation model and i do:
		- I compute the backward message for time t using the matrix operations
		- I compute the forward message at time t-1 from the filtering_m of the future using the inverse of the matrix equation
		- I compute the smoothing value at time j and append it to the final list to return'''

	def smoothing_matrix_rec(self,obs,filtering_m,t,n):
		f = filtering_m    								#filtering message of last time index
		back_m = np.array([1,1])
		result = list()
		for j in range(n,t-1,-1):
			if j == 0:									#needed to handle the special case at j == 0 where I don't have any observations
				s = self.normalize(f*back_m)
				result.append(s)
				r = result[::-1]
				return r
			if obs[j][1] == self.obs_values[0]:
				O_true = np.array([[self.O[0][0],0.0],
								   [0.0,self.O[1][0]]])
				smoothing = self.normalize(f*back_m)				# I compute the smoothing value at index j
				result.append(smoothing)
				back_m = self.T.dot(O_true.dot(back_m))							# I compute the new backwards message for next iteration bk+1:t = TOk+1bk+2:t
				f = self.normalize(inv((self.T.T)).dot(inv(O_true).dot(f)))     # I compute the new forward message with the inversion formula f1:t = alpha'*(transpose(T))^-1*(Ot+1)^-1*f1:t+1
			else:
				O_false = np.array([[self.O[0][1],0.0],
									[0.0,self.O[1][1]]])
				smoothing = self.normalize(f*back_m)
				result.append(smoothing)
				back_m = self.T.dot(O_false.dot(back_m))
				f = self.normalize(inv((self.T.T)).dot(inv(O_false).dot(f)))
		r = result[::-1]
		return r

		''' This function compute the smoothings of a observation sequence using the matrix algorithms for the HMM. First i compute the filtering of the last timw index
		of the observation which is len(obs)-1 and then i call the auxiliar function smoothing_matrix_rec which computes the list of all smoothing values for
		time indexes from 0 to len(obs)-1 '''

	def smoothing_matrix_algorithm(self,obs,to,fr):
		last_filtering = self.filtering_matrix_algorithm(obs,fr)
		return self.smoothing_matrix_rec(obs,last_filtering,to,fr)