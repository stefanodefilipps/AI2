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

#samples is a list containing the 15 samples
samples = list()

#every sample is a list containing a tupe of observation and state

for i in range(0,15):
	samples.append(list()) 


for i in range(0,15):
	for j in range(0,20):
		if j == 0:
			samples[i].append(np.random.choice(rain,1, p = [0.5,0.5]))
			#print(samples[i])
		else:
			obs = list()
			if samples[i][j-1][0] == "rain":
				obs.append(np.random.choice(rain,1, p = T[0])[0])
			else:
				obs.append(np.random.choice(rain,1, p = T[1])[0])
			if obs[0] == "rain":
				obs.append(np.random.choice(umbrella,1, p = Obs[0])[0])
			else:
				obs.append(np.random.choice(umbrella,1, p = Obs[1])[0])
			samples[i].append(obs)

for e in samples[0]:
	print(e)

obs = ["",["rain","umbrella"],["rain","umbrella"]]

print(filtering(T,Obs,samples[0],np.array([0.5,0.5]),7))
print(filtering(T,Obs,obs,np.array([0.5,0.5]),2))

mp = Markov_Process(T,Obs,rain,umbrella,np.array([0.5,0.5]))
mp.generate_samples(15,20)

print(mp.filtering(samples[0],7))

print("BACKWARD NORMAL")
print(backwards(2,1,Obs,T,obs,np.array([1,1])))
print(mp.backwards(2,1,obs))
print("SMOOTHING: "+str(smoothing(Obs,T,obs,1,2)))
print("SMOOTHING MP: "+str(mp.smoothing(obs,1,2)))
smoothings_ = smoothing_sequence(obs,Obs,T)
print("SMOOTHING SEQUENCE")
for i in range(0,len(smoothings_)):
	print(smoothings_[i])
smoothings_ = mp.smoothing_sequence(obs,0,2)
print("SMOOTHING SEQUENCE MP")
for i in range(0,len(smoothings_)):
	print(smoothings_[i])

smoothing_matrix = smoothing_matrix_algorithm(obs,Obs,T)
print("SMOOTHING MATRIX ALGORITHM")
for i in range(0,len(smoothing_matrix)):
	print(smoothing_matrix[i])

smoothing_matrix = mp.smoothing_matrix_algorithm(obs,0,2)
print("SMOOTHING MATRIX ALGORITHM MP")
for i in range(0,len(smoothing_matrix)):
	print(smoothing_matrix[i])

print("SMOOTHING2: "+str(smoothing(Obs,T,samples[0],6,len(samples[0])-1)))
print("SMOOTHING2 MP: "+str(mp.smoothing(samples[0],6,len(samples[0])-1)))

f = open("test.txt","a")
for e in samples[0]:
	f.write(str(e)+"\n")
f.write("FILTERINGS\n")
for i in range(0,len(samples[0])):
	fl = mp.filtering(samples[0],i)
	f.write(str(fl)+"\n")
f.write("SMOOTHINGS\n")
smoothing_matrix = mp.smoothing_sequence(samples[0],0,len(samples[0])-1)
for s in smoothing_matrix:
	f.write(str(s)+"\n")
for e in samples[1]:
	f.write(str(e)+"\n")
f.write("FILTERINGS\n")
for i in range(0,len(samples[1])):
	fl = mp.filtering(samples[1],i)
	f.write(str(fl)+"\n")
f.write("SMOOTHINGS\n")
smoothing_matrix = mp.smoothing_sequence(samples[1],0,len(samples[1])-1)
for s in smoothing_matrix:
	f.write(str(s)+"\n")
f.close()


