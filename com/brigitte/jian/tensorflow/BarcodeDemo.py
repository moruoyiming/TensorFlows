import matplotlib.pyplot as plt
import numpy as np
np.random.seed(19680801)
# Fixing random state for reproducibility
x = np.random.rand(500) > 0.7
# the bar
barprops = dict(aspect='auto', cmap='binary', interpolation='nearest')
fig = plt.figure()
# a vertical barcode
ax1 = fig.add_axes([0.1, 0.1, 0.1, 0.8])

ax1.set_axis_off()

ax1.imshow(x.reshape((-1, 1)), **barprops)
# a horizontal barcode
ax2 = fig.add_axes([0.3, 0.4, 0.6, 0.2])

ax2.set_axis_off()

ax2.imshow(x.reshape((1, -1)), **barprops)

plt.show()
