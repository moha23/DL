import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# np.set_printoptions(threshold=np.nan)

masses = np.load('DL_Ass_1_q2_input/masses.npy')
positions = np.load('DL_Ass_1_q2_input/positions.npy')
velocities = np.load('DL_Ass_1_q2_input/velocities.npy')

sess = tf.Session()

# constructing the graph
masses_tensor_const = tf.constant(masses, name = 'masses')
masses_tensor = tf.transpose(masses_tensor_const)
positions_placeholder = tf.placeholder(tf.float64)
velocities_placeholder = tf.placeholder(tf.float64)

# Compute the accel vector
G = tf.constant((-6.67*(10**5)), dtype=tf.float64)
a2 = tf.broadcast_to(positions_placeholder, [100, 100, 2])
a1 = tf.transpose(a2, perm=[1, 0, 2])
displacement_mat = a1 - a2
denominator = (tf.norm(displacement_mat, axis=2)) ** 3
a3 = masses_tensor / denominator
a4 = tf.stack([a3, a3], axis=2)
a5 = a4 * displacement_mat
a6 = G * a5
a6_0 = a6[:,:,0]
a7 = tf.linalg.set_diag(a6_0, np.zeros(100))
a6_1 = a6[:,:,1]
a8 = tf.linalg.set_diag(a6_1, np.zeros(100))
a9 = tf.stack([a7, a8], axis=2)
accel = tf.reduce_sum(a9, axis=1)

#compute new_positions, new_velocities
delta_t = tf.constant(10**-4, dtype=tf.float64)
b1 = velocities_placeholder * delta_t
half = tf.constant(0.5, dtype=tf.float64)
delta_square = delta_t ** 2
b2 = half * accel * delta_square
new_positions_tensor = positions_placeholder + b1 + b2
new_velocities_tensor = velocities_placeholder + (accel * delta_t)

#compute minimum distance between any two points
displacement_mat_norm = tf.norm(displacement_mat, axis=2)
displacement_mat_norm = tf.linalg.set_diag(displacement_mat_norm, np.ones(100))
min_value_tensor = tf.reduce_min(displacement_mat_norm)

count = 0
with tf.Session() as sess:
	threshold_dist = 0.1
	while True:
		plt.scatter(positions[:, 0], positions[:, 1])
		plt.xlim(-100, 100)
		plt.ylim(-100, 100)
		plt.savefig('Images/' + str(count) + "_fig.png")
		plt.clf()
		min_value = sess.run(min_value_tensor, {positions_placeholder: positions, velocities_placeholder: velocities})
		if min_value < threshold_dist:
			break
		positions, velocities = sess.run([new_positions_tensor, new_velocities_tensor], {positions_placeholder: positions, velocities_placeholder: velocities})
		np.save('output_positions', positions)
		np.save('output_velocities', velocities)
		count += 1


