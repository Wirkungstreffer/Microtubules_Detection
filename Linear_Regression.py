import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15])
y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])

tck = interpolate.splrep(x, y, k=3, s=0)
xnew = np.linspace(0, 15)

fig, axes = plt.subplots(2)

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

import numpy as np
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# parameters for setup
n_data = 20

# segmented linear regression parameters
n_seg = 3

np.random.seed(0)
fig, (ax0, ax1) = plt.subplots(1, 2)

# example 1
#xs = np.sort(np.random.rand(n_data))
#ys = np.random.rand(n_data) * .3 + np.tanh(5* (xs -.5))

# example 2
xs = np.linspace(-1, 1, 20)
ys = np.random.rand(n_data) * .3 + np.tanh(3*xs)

dys = np.gradient(ys, xs)

rgr = DecisionTreeRegressor(max_leaf_nodes=n_seg)
rgr.fit(xs.reshape(-1, 1), dys.reshape(-1, 1))
dys_dt = rgr.predict(xs.reshape(-1, 1)).flatten()

ys_sl = np.ones(len(xs)) * np.nan
for y in np.unique(dys_dt):
    msk = dys_dt == y
    lin_reg = LinearRegression()
    lin_reg.fit(xs[msk].reshape(-1, 1), ys[msk].reshape(-1, 1))
    ys_sl[msk] = lin_reg.predict(xs[msk].reshape(-1, 1)).flatten()
    ax0.plot([xs[msk][0], xs[msk][-1]],
             [ys_sl[msk][0], ys_sl[msk][-1]],
             color='r', zorder=1)

ax0.set_title('values')
ax0.scatter(xs, ys, label='data')
ax0.scatter(xs, ys_sl, s=3**2, label='seg lin reg', color='g', zorder=5)
ax0.legend()

ax1.set_title('slope')
ax1.scatter(xs, dys, label='data')
ax1.scatter(xs, dys_dt, label='DecisionTree', s=2**2)
ax1.legend()

plt.show()