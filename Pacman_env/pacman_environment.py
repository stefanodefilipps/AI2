import random

class Environment_PacMan:

	def __init__(self,size,density,ghosts,walls):
		"""Environments have fixed size and cherry counts."""
		self.size = size				#size of the grid size*size
		self.density = density			#number of cherries
		self.n_ghosts = ghosts
		self.n_walls = walls
		self.actions = ["UP","DOWN","LEFT","RIGHT"]
		self.next_reward = 0


	'''This second init is used only when i have to try value iteration in order to have a fixed environment that i can use multiple times'''

	def init_2(self):
		self.pacman = [3,3]
		self.cherries = [[0,0],[0,1],[0,4],[1,2],[2,3],[4,0],[4,1],[3,4],[4,4]]
		self.cherries_place = list()
		for i in range(len(self.cherries)):
			self.cherries_place.append(1)
		self.walls = [[1,0],[2,0],[3,0],[4,2],[4,3]]
		self.ghosts = [[2,1],[1,4]]
		self.next_reward = 0
		self.density = len(self.cherries)
		self.n_ghosts = len(self.ghosts)
		self.n_walls = len(self.walls)
	
	"""Place pacman, cherries, and ghost at random locations."""

	def initialize(self):
		
		locations = list()
		for r in range(0,self.size):
			for c in range(0,self.size):
				locations.append([r, c])
		
		random.shuffle(locations)
		self.pacman = locations.pop()
		
		self.cherries = list()				 #contains position of the cherries
		self.cherries_place = list()         #it's a boolean list that says if in position cherries[i] the cherry has been eaten(cherries_place[i] = 0) or not (cherries_place[i] = 1)
		for count in range(self.density):
			self.cherries.append(locations.pop())
			self.cherries_place.append(1)
			
		self.ghosts = list()				 #contains position of the ghosts
		for count in range(self.n_ghosts):
			self.ghosts.append(locations.pop())

		self.walls = list()					 #contains position of the walls
		for count in range(self.n_walls):
			self.walls.append(locations.pop())

		self.next_reward = 0     #this variable contains the last reward received by the agent

	"""Print the environment."""

	def display(self):
		for r in range(self.size):
			for c in range(self.size):
				if [r,c] in self.ghosts:
					print 'G',
				elif [r,c] == self.pacman:
					print 'O',
				elif [r,c] in self.cherries and self.cherries_place[self.cherries.index([r,c])] == 1:
					print '.',
				elif [r,c] in self.walls:
					print 'X',
				else:
					print ' ',
			print
		print

	"""Return the actions the agent may try to take."""

	def actions(self):
		if self.terminal():
			return None
		else:
			return self.actions

	def terminal(self):
		"""Return whether the episode is over."""
		if self.next_reward == -100:			#in this case i encountered a ghost or i got against a wall and so game over
			return True
		elif self.eaten_all_cherries(): # i check if all the cherries have been eaten and if true then the game is won
			return True
		else:
			return False

	def reward(self):
		"""Return the reward earned at during the last update."""
		return self.next_reward

	''' This is the ction that the agent can execute as in the gym interface and returns the observation the reward and wheter the episode has endend or not'''

	def step(self, action):

		# if the episode has ended then the environment can't change and we need to initialize it again in order to start a new episode

		print(action)

		pacman = self.pacman

		#if action is not allowed than don't do anything
		# Pacman moves as chosen
		print(self.translate_action(action))
		[r, c] = self.pacman
		[dr, dc] = self.translate_action(action)
		self.pacman = [r+dr, c+dc]

		# Negative reward for hitting a ghost

		if self.pacman in self.ghosts:
			self.next_reward = -100

		# Negative reward for hitting a wall

		elif self.pacman in self.walls or self.pacman[0]<0 or self.pacman[0] == self.size or self.pacman[1]<0 or self.pacman[1] == self.size:
			self.next_reward = -30
			self.pacman = pacman

		# Positive reward for consuming a cherry

		elif self.pacman in self.cherries and self.cherries_place[self.cherries.index(self.pacman)] == 1:
			self.next_reward = 10
			self.cherries_place[self.cherries.index(self.pacman)] = 0

		elif self.eaten_all_cherries():
			self.next_reward = 1000

		else:
			self.next_reward = 0

		d = dict()

		d["observation"] = self.observation()
		d["reward"] = self.reward()
		d["done"] = self.terminal()

		return d


	def observation(self):
		"""Return a description of the state of the environment."""
		s = dict()
		
		# Baseline feature noting how many cherries are left
		s['cherries left'] = len(self.cherries)
		#Position of PacMan
		s['pacman'] = self.pacman
		#Positions of ghosts
		s['ghosts'] = self.ghosts
		#Positions of walls
		s['walls'] = self.walls
		#Positions of cherries
		s['cherries'] = self.cherries
		#A flag value to know if the cherry in a given position has been eaten or not
		s['cherries_place'] = self.cherries_place

		return s

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

	def eaten_all_cherries(self):
		if len(list(filter(lambda x: x == 1,self.cherries_place))) == 0:
			return True
		return False