import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15])
y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])

tck = interpolate.splrep(x, y, k=1, s=0)
xnew = np.linspace(0, 15)

fig, axes = plt.subplots(3)

axes[0].plot(x, y, 'x', label = 'data')
axes[0].plot(xnew, interpolate.splev(xnew, tck, der=0), label = 'Fit')
axes[1].plot(x, interpolate.splev(x, tck, der=1), label = '1st dev')
#dev_2 = interpolate.splev(x, tck, der=2)
#axes[2].plot(x, dev_2, label = '2st dev')

#turning_point_mask = dev_2 == np.amax(dev_2)
#axes[2].plot(x[turning_point_mask], dev_2[turning_point_mask],'rx',
#                label = 'Turning point')
for ax in axes:
       ax.legend(loc = 'best')

plt.show()