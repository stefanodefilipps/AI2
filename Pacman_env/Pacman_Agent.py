
class Pacman_Agent:

	''' This class implements the agent that operates in the environment. In this particular case it has a MDP model of the problem so it can do planning
		through value iteration and compute the exact optimal policy'''

	def __init__(self,MDP):
		self.MDP = MDP  	# the self has to know the MDP of the environment
		self.U = dict()		# this is the utilities table for all the state of the MDP initially all entries are 0
		self.pi_opt = dict()
		for s in MDP.states:
			self.U[str(s)] = 0			#initialize the utilities of all states to 0 when instantiated a new instance of the class
		for s in self.MDP.states:
			self.pi_opt[str(s)] = self.MDP.actions[0]		#create a new random policy

	''' This function implements the value function algorithm.
		delta is the maximum variation of state value during a single iteration and if less than a certain threshold then i stop the iteration because i am very
		close tp convergence of the values.
		I also have a max_iter parameter that specifies that maximum number of iterations in case the convergence is very slow'''

	def value_iteration_learning(self,epsilon,max_iter):
		delta = 100
		for s in self.MDP.states:
			self.U[str(s)] = 0
		while delta > (float(epsilon*(1-self.MDP.discount))/self.MDP.discount) and max_iter > 0:
			delta = 0
			print("ITER: "+str(max_iter))
			print(delta)
			print(float(epsilon*(1-self.MDP.discount))/self.MDP.discount)
			max_iter = max_iter-1
			for s in self.MDP.states:
				u = self.U[str(s)]				#store old U(s)
				self.U[str(s)] = self.max_over_actions_value(s)    #update U(s) computing max(a) R(a,s) + gamma*sum(s')P(s'|a,a)*U(s')
				if abs(u - self.U[str(s)]) > delta:					#compute the variance wrt previous iteration and store only the maximu value in absolute value
					delta = abs(u - self.U[str(s)])
		return self.U			#indicates that it has finished iterating to the external user


	''' This is an auxialiary function used in value_iteration_learning to choose the maximum value of the neighbour of a state when we take an action
		from state i try all the action and sum the reward collected with the state value of the new state. 
		I find the maximum of these values and this will be the new estimate of the state value
		computing max(a) R(a,s) + gamma*sum(s')P(s'|a,a)*U(s')'''

	def max_over_actions_value(self,state):
		max_value = self.MDP.rewards(state,self.MDP.actions[0])+self.MDP.discount*self.U[str(self.MDP.successors(state,self.MDP.actions[0]))]
		for a in self.MDP.actions:
			s = self.MDP.successors(state,a)
			v = self.MDP.rewards(state,a)+self.MDP.discount*self.U[str(s)]
			if v > max_value:
				max_value = v
		return max_value


	''' This function compute the optimal value given the optimal state values U
		For each state the algorithm execute all the action and compute the sum between the reward collected and the state value of the next state and 
		found the action that maximizes these values'''


	def optimal_policy(self):
		self.pi_opt = dict()
		for s in self.MDP.states:
			self.pi_opt[str(s)] = self.MDP.actions[0]
		for state in self.MDP.states:
			max_action = self.MDP.actions[0]
			max_value = self.MDP.rewards(state,self.MDP.actions[0])+self.MDP.discount*self.U[str(self.MDP.successors(state,self.MDP.actions[0]))]
			for a in self.MDP.actions:
				s = self.MDP.successors(state,a)
				v = self.MDP.rewards(state,a)+self.MDP.discount*self.U[str(s)]
				if v > max_value:
					max_value = v
					max_action = a
			self.pi_opt[str(state)] = max_action
		return self.pi_opt
