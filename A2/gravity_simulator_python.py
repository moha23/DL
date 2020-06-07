import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)

masses = np.load('DL_Ass_1_q2_input/masses.npy')
positions = np.load('DL_Ass_1_q2_input/positions.npy')
velocities = np.load('DL_Ass_1_q2_input/velocities.npy')

def threshold_cond(positions, threshold_dist):
	dist_matrix = np.zeros((100, 100))
	for i in range(len(positions)):
		dist_matrix[i] = np.sqrt(np.sum((positions[i] - positions) ** 2, axis=1))
	np.fill_diagonal(dist_matrix, 1)
	min_value = np.min(dist_matrix)
	return (min_value < threshold_dist)

def compute_accel(positions, mass):
	G = 6.67*(10**5)
	accel = np.zeros((100, 2))
	for i in range(len(accel)):
		total = np.zeros((2,))
		for j in list(range(0, i)) + list(range(i+1, len(accel))):
			displacement_vector = positions[i] - positions[j]
			total += ((masses[j] / (np.linalg.norm(displacement_vector)) ** 3) * displacement_vector)
		total = (-1 * G) * total
		accel[i] = total

	return accel

delta_t = 10**-4
count = 0
plt.scatter(positions[:, 0], positions[:, 1])
plt.savefig('Images/' + str(count) + "_fig.png")
plt.xlim((-100, 100))
plt.ylim((-100, 100))
plt.clf()
while (threshold_cond(positions, 0.1) == 0):
	accel = compute_accel(positions, masses)
	new_positions = positions + (velocities  * delta_t) + (0.5 * accel * (delta_t ** 2))
	new_velocities = velocities + accel * delta_t
	plt.scatter(new_positions[:, 0], new_positions[:, 1])
	positions = new_positions
	velocities = new_velocities
	plt.xlim(-100, 100)
	plt.ylim(-100, 100)
	plt.savefig('Images/' + str(count) + "_fig.png")
	plt.clf()
	if count == 2:
		print(count)
		print(new_velocities)
		exit()
	count += 1
	# print(count)

