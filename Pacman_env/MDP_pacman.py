import itertools as it

class MDP_pacman:

	'''The state is rapresented in the following way
		<x,y,xg1,yg1,...,xgi,ygi,xw1,yw1,...xwk,ywk,xc1,yc1,...,xcn,ycn,c1,...,cj,e>

		--- x and y are the coordinates of the agent
		--- xgi and ygi are the coordinates of the static ghost
		--- xwk and ywk are the coordinates of the walls
		--- xcn and ycn are the coordinates of the cherries
		--- cj are boolean flag that inidicates if cherry in xcj and ycj has been eaten or not by pacman
		--- e is a boolean flag that indicates if all the cherries have been eaten

		For semplicity of implementation, though, we use as state <x,y,c1,...,cj,e> during computation and other informations are stored in the obs structure
		because they are static and never change'''

	def __init__(self,env,discount):
		self.actions = env.actions			# action set A of the MDP
		self.size = env.size
		self.obs = env.observation()
		self.discount = discount
		self.states = self.create_all_possible_states(env)

	''' This function compute the rewards when the agent is in state and do action a
		Reward function of the MDP SxAxS ---> R'''

	def rewards(self,state,a):
		s = self.successors(state,a)
		if [s[0],s[1]] in self.obs["ghosts"]:      #i am in the same position as a ghost and then i get a punishment or i bumped against a wall and hence i remained in the same position as before the action
			return -100
		elif (s[0] == state[0] and s[1] == state[1]):
			return -30
		elif state[len(state)-1] == 1:   #i have eaten all the cherries
			return 1000
		#i have to check that in the next state i have eaten a cherry and in that case i have to give a reward
		elif self.eaten_cherry(state,s):
			return 10
		else:
			return 0

	''' This function return the state resulting from the application of action a in state.
		Important to notice that our environment is fully observable and in particular is deterministic so when i execute a particular action there is only
		one state my agent can end up in
		I first compute the new position of the agent and if the position is a wall or it's outside of the grid, then the position doesn't change and neither
		the rest of the state
		If in the new position reached there is a cherry, then I have to put to 0 the flag corresponding to that particular cherry, in order to keep track
		of the eaten cherries and check that if all the flag are 0s then i reached the goal and also set the corresponding boolean flag to 1
		Otherwise simply return the new state which is equal to the previous one but with te coordinates of the agent modified
		Transition function of the MDP  SxA ---> S and in this case is deterministic'''

	def successors(self,state,a):
		s = [state[0],state[1]]
		step = self.translate_action(a)
		new_s = [s[0]+step[0],s[1]+step[1]]
		if new_s in self.obs["walls"] or new_s[0]<0 or new_s[0] == self.size or new_s[1]<0 or new_s[1] == self.size:        #if in the next position i have a wall then i don't do anything and return the old state
			return state
		new_state = list(state)
		new_state[0] = new_s[0]
		new_state[1] = new_s[1]               #otherwise i change position of pacman
		if new_s in self.obs["cherries"] and (state[self.obs["cherries"].index(new_s)+2] == 1): #in this case i reached a position where i have a not eaten cherry and so i ate and remove it from that position
			new_state[self.obs["cherries"].index(new_s)+2] = 0
			finish = True
			for i in range(2,len(state)-1):   
				if state[i] == 1:
					finish = False
			if finish:
				new_state[len(state)-1] = 1                 #if a have eaten all the cherries then put the flag variable to one
		return new_state

	''' I take the action in textual form and translate it in the 2 values that are neede in actual computation'''

	def translate_action(self,action):
		if action not in self.actions:
			return [0,0]
		if action == "UP":
			return [-1,0]
		if action == "DOWN":
			return [1,0]
		if action == "LEFT":
			return [0,-1]
		if action == "RIGHT":
			return [0,1]

	''' This function is used in order to know if the state is one where all cherries have been eaten'''

	def eaten_cherry(self,previous_s,next_s):
		eaten = False
		c_1 = 0
		c_2 = 0
		for i in range(2,len(previous_s)-1):        # I scan only the element of the state where i have stored the infos about the presence or not of the cherries
			c_1+=previous_s[i]
			c_2+=next_s[i]                          # I am counting how many cherries are uneaten in previous and next state 
		eaten = (c_2 == (c_1 - 1))                  # If in the next state i have a number of uneaten cherries equal to the previous one minus 1 then i have eaten the cherry that was in the next state
		return eaten

	def zeros(self,s):
		for i in range (2,len(s)-1):
			if s[i] == 1:
				return False
		return True


	''' This is an auxialiary function in order to create all the possible useful states of the MDP
		It is important to notice that I have to eliminate all the state where the position of the agent is in the same position of a wall
		It is important to notice that I eliminate all the states where we say that the goal has been reached but not all the cherries have been eaten
		These 2 improvements allow us to prune a lot of useless states because unreachable and make the computation faster'''

	def create_all_possible_states(self,env):
		if env == None:
			return
		sets = [list(range(env.size)),list(range(env.size))]
		for c in range(len(self.obs["cherries"])):
			sets.append([0,1])
		sets.append([0,1])
		states = list(it.product(*sets))    #at this point i have all the states needed of all the possible combinations
		for i in range(len(states)):
			states[i] = list(states[i])
		#not_wanted = list(filter(lambda s: (not self.zeros(s))(self.zeros(s) and (s[len(s)-1] == 1)) or ((s[len(s)-1] == 0) and not self.zeros(s)),self.states))
		not_wanted = list(filter(lambda s: (not self.zeros(s)) and s[len(s)-1] == 1 ,states))
		states = [item for item in states if item not in not_wanted]
		states = list(filter(lambda s: [s[0],s[1]] not in self.obs["walls"],states))
		return states