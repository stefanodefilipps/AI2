In this directory there is the implementaion of the third exercise of the Probabilistic Reasoning homework.

--- pacman_environment.py

In this file is implemented the class that represents the environment of the chosen game, in this case Pacman

--- Pacman_main.py

In this file is implemented a simple script to interact with the environment through bash commands until the QUIT command is inserted.

--- pacman_random.py

In this file is implemented a randomize interaction of a possibile agent with the environment. 5 moves are generated randomly and executed in the environment.

--- MDP_pacman.py

In this file is implemented the class represnting the model of the Markov Decision Problem that our aget can use to plan the optimal policy.

--- Pacman_Agent.py

In this file is implemented the class represnting the agent that interacts with the environment. In particular contains the implementation of the value iteration algorithm to do planning given the model of the MDP and the function to compute the optimal policy.

--- value_iteration_main.py

In this file is implemented qa simple script to create an environment, create an agent and compute the optimal policy through the value iteration algorithm and thereafter executing the optimal policy just computed