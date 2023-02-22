from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall
import pandas as pd
import matplotlib.pyplot as plt



"""!
The pyclustering library clarans implementation requires
list of lists as its input dataset.
Thus we convert the data from numpy array to list.
"""

df = pd.read_csv(r'resources/2d-data-no-classes.csv')
print(df.shape[0])
data = df.values.tolist()

#get a glimpse of dataset
print("A peek into the dataset : ")
print(data[:4])

num_of_clusters = 3

"""!
@brief Constructor of clustering algorithm CLARANS.
@details The higher the value of maxneighbor, the closer is CLARANS to K-Medoids, and the longer is each search of a local minima.

@param[in] data: Input data that is presented as list of points (objects), each point should be represented by list or tuple.
@param[in] number_clusters: amount of clusters that should be allocated.
@param[in] numlocal: the number of local minima obtained (amount of iterations for solving the problem).
@param[in] maxneighbor: the maximum number of neighbors examined.        
"""
clarans_instance = clarans(data, num_of_clusters, 20, 5)

#calls the clarans method 'process' to implement the algortihm
(ticks, result) = timedcall(clarans_instance.process)
print("Execution time : ", ticks, "\n")

#returns the clusters
clusters = clarans_instance.get_clusters()

#returns the mediods
medoids = clarans_instance.get_medoids()


print("Index of the points that are in a cluster : ")
print(clusters)
print("The index of medoids that algorithm found to be best : ")
print(medoids)

x = [[], [], []]
y = [[], [], []]

for i in range(num_of_clusters):
    for c in clusters[i]:
        x[i].append(data[c][0])
        y[i].append(data[c][1])

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x[0], y[0], s=10, c='r', marker="s", label='0')
ax1.scatter(x[1], y[1], s=10, c='g', marker="s", label='1')
ax1.scatter(x[2], y[2], s=10, c='b', marker="s", label='2')
ax1.scatter(data[medoids[0]][0], data[medoids[0]][1], s=30, c='r', marker="^", label='0')
ax1.scatter(data[medoids[1]][0], data[medoids[1]][1], s=30, c='g', marker="^", label='1')
ax1.scatter(data[medoids[2]][0], data[medoids[2]][1], s=30, c='b', marker="^", label='2')
plt.legend(loc='upper left')
plt.title("Reference 2d data with 3 clusters")
plt.show()