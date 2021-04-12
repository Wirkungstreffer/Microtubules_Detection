from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


data = pd.read_csv('xclara.csv')
print(data.shape)
data.head()

f1 = data['V1'].values
f2 = data['V2'].values

X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=6)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

k = 3

C_x = np.random.randint(0, np.max(X)-20, size=k)

C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)


plt.scatter(f1, f2, c='black', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='red')

C_old = np.zeros(C.shape)
print(C)

clusters = np.zeros(len(X))

iteration_flag = dist(C, C_old, 1)

tmp = 1

while iteration_flag.any() != 0 and tmp < 20:
    for i in range(len(X)):
        distances = dist(X[i], C, 1)
        # print(distances)
        cluster = np.argmin(distances) 

        clusters[i] = cluster
        
    # print("the distinct of clusters: ", set(clusters))
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        # print(points)
        # print(np.mean(points, axis=0))
        C[i] = np.mean(points, axis=0)
        # print(C[i])
    # print(C)
    
    print ('loop time %d' % tmp)
    tmp = tmp + 1
    iteration_flag = dist(C, C_old, 1)
    print("distance between new point and old point：", iteration_flag)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()

for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')

plt.show()