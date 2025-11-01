import numpy as np

import matplotlib.pyplot as plt


# Define 2D vectors (only x and y will be plotted)

v1 = np.array([3, 4])

v2 = np.array([2, 1])


# Vector operations

v_add = v1 + v2

v_sub = v1 - v2


# Set up the plot

plt.figure(figsize=(8, 8))


# Plot original vectors from origin

plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1 = [3, 4]')

plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2 = [2, 1]')


# Plot vector addition from origin

plt.quiver(0, 0, v_add[0], v_add[1], angles='xy', scale_units='xy', scale=1, color='g', alpha=0.7, label='v1 + v2 = [{} , {}]'.format(v_add[0], v_add[1]))



# Plot vector subtraction from origin

plt.quiver(0, 0, v_sub[0], v_sub[1], angles='xy', scale_units='xy', scale=1, color='m', alpha=0.7, label='v1 - v2 = [{} , {}]'.format(v_sub[0], v_sub[1]))


# Show subtraction by drawing v2 from tip of v1 to tip of v2

plt.quiver(v2[0], v2[1], v1[0] - v2[0], v1[1] - v2[1], angles='xy', scale_units='xy', scale=1, color='m', alpha=0.3, linestyle='dashed')


# Annotate vectors

plt.text(v1[0], v1[1], 'v1', fontsize=12, color='r', ha='right')

plt.text(v2[0], v2[1], 'v2', fontsize=12, color='b', ha='left')

plt.text(v_add[0], v_add[1], 'v1+v2', fontsize=12, color='g', ha='left')

plt.text(v_sub[0], v_sub[1], 'v1-v2', fontsize=12, color='m', ha='left')


# Axes settings

plt.xlim(-2, max(v1[0], v2[0], v_add[0], v_sub[0]) + 3)

plt.ylim(-2, max(v1[1], v2[1], v_add[1], v_sub[1]) + 3)

plt.grid(True)

plt.axhline(0, color='black', linewidth=0.5)

plt.axvline(0, color='black', linewidth=0.5)

plt.gca().set_aspect('equal', adjustable='box')

plt.legend()

plt.title('Vector Addition and Subtraction (2D Projection)')

plt.xlabel('X-axis')

plt.ylabel('Y-axis')


plt.show()
