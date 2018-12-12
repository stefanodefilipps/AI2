from pacman_environment import *
import sys

pacman_env = Environment_PacMan(10,40,4,20)
pacman_env.initialize()
pacman_env.display()
action = raw_input("Insert an action: ")
while action != "QUIT":
	obs = pacman_env.step(action)
	pacman_env.display()
	print(obs["reward"])
	print(obs["observation"])
	if pacman_env.terminal():
		restart = raw_input("END OF EPISODE DO YOU WANT TO RESTART? [y/n]: ")
		if restart == "n" or restart != "y":
			break
		else:
			pacman_env.initialize()
			pacman_env.display()
			action = raw_input("Insert an action: ")
	action = raw_input("Insert an action: ")
