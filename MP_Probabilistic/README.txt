In this directory are implemented all the function for the second point of the Probabilistic reasoning Homework.

--- MP_class.py

in this file is implemented the Markov Process and the function to do probabilist inference. In particular in this file there are the data structure to represent the Markov Process(the transfer model and observation model as matrix) and the functions to execute the filtering and smoothing prediction.
In this class is also implemented the improved version of the smoothing algorithm using the matrix formulation and formulas

--- markov_process.py

In this file are implemented different version of the filtering and smoothing algorithms, considering both the recursive and iterative formuation of the algorithms.
It was a preliminary file and all the function were later moved in the MP_class.py file

--- main.py

In this file are presented the trial that i computed in order to verify that the alhorithms were functioning correctly. Therefore, i both used the function in MP_class.py and markov_process.py to verify that the results were consistent.

--- final_main.py

In this file there is the actual main to try the class implementing the markov process and so i only use the function implemented in the MP_class.py file.
I first generate the samples and then i use only the first sequence to do a filtering prediction and a smoothing prediction using both the inefficient implementation and the matrix implementation