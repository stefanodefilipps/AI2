from pacman_environment import *
import random 



actions = ["UP","DOWN","LEFT","RIGHT"]
pacman_env = Environment_PacMan(10,40,4,20)
pacman_env.initialize()
pacman_env.display()

for i in range(5):
	action_index = random.randint(0,3)
	action = actions[action_index]
	print("action chosen: "+action)
	obs = pacman_env.step(action)
	pacman_env.display()
	print(obs["reward"])
	print(obs["observation"])