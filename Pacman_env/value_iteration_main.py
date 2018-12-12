from Pacman_Agent import *
from pacman_environment import *
from MDP_pacman import *

pacman_env = Environment_PacMan(5,8,3,6)
pacman_env.init_2()
pacman_env.display()
MDP = MDP_pacman(pacman_env,0.8)	#create the MDP of the problem
pacman_agent = Pacman_Agent(MDP)	#create the agent
pacman_agent.value_iteration_learning(0.1,300)	#compute the optimal value for each state
pacman_agent.optimal_policy()	#compute the optimal policy

obs = pacman_env.observation()
pacman_env.display()
print(pacman_env.terminal())

while not pacman_env.terminal():		#until we haven't reached the end of the episode the agent execute an action following the optimal policy previously computed
	state = [obs["pacman"][0],obs["pacman"][1]]
	for i in range(len(obs["cherries_place"])):
		state.append(obs["cherries_place"][i])
	if pacman_env.terminal():
		state.append(1)
	else:
		state.append(0)
	#now i have constructed the state of the agent and i can consult the policy
	action = pacman_agent.pi_opt[str(state)]
	print(state)
	print("ACTION: "+action)
	pacman_env.step(action)
	obs = pacman_env.observation()
	pacman_env.display()


