import numpy as np
from markov_process import *
from MP_class import *

#this is the transition model
T = np.array([[0.7,0.3],
			  [0.3,0.7]])

Obs = np.array([[0.9,0.1],
				[0.2,0.8]])

O_true = np.array([[0.9,0.0],
				   [0.0,0,2]])

O_false = np.array([[0.1,0.0],
				    [0.0,0,8]])

rain = ["rain","not rain"]
umbrella = ["umbrella","not umbrella"]

mp = Markov_Process(T,Obs,rain,umbrella,np.array([0.5,0.5]))
mp.generate_samples(15,20)

print("FIRST SEQUENCE OF 20 ELEMENTS")
for i in mp.samples[0]:
	print(i)
print("FILTERING AT TIME T = 7: "+str(mp.filtering(mp.samples[0],7)))
print("SMOOTHING WHOLE FIRST SEQUENCE: ")
smoothings_ = mp.smoothing_sequence(mp.samples[0],0,len(mp.samples[0])-1)
for i in range(0,len(smoothings_)):
	print(smoothings_[i])
smoothing_matrix = mp.smoothing_matrix_algorithm(mp.samples[0],0,len(mp.samples[0])-1)
print("SMOOTHING MATRIX ALGORITHM MP")
for i in range(0,len(smoothing_matrix)):
	print(smoothing_matrix[i])