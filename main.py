import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Node import Node
from CLARANS import CLARANS


def prepare_numerical_data_for_nodes(nodes):
    min_attributes = {}
    max_attributes = {}
    for col_name, value in nodes[0].numerical_attributes.items():
        min_attributes.update({col_name: value})
        max_attributes.update({col_name: value})

    for n in nodes:
        for col_name, value in n.numerical_attributes.items():
            if col_name in min_attributes and value < min_attributes[col_name]:
                min_attributes.update({col_name: value})
            if col_name in max_attributes and value > max_attributes[col_name]:
                max_attributes.update({col_name: value})

    range = {}
    for col_name, value in min_attributes.items():
        if col_name in max_attributes:
            r = abs(max_attributes[col_name] - min_attributes[col_name])
            range.update({col_name: r})

    for n in nodes:
        n.numerical_attributes_range = range

if __name__ == '__main__':
    df = pd.read_csv(r'resources\2d-data.csv')
    print(df.shape[0])

    nodes = np.empty(df.shape[0], dtype=object)

    #2-d dataset (x and y)
    for index, row in df.iterrows():
        nodes[index] = Node(index, None, None, x=row['x'], y=row['y'])

    #mixed dataset, nominal and numerical
    # for index, row in df.iterrows():
    #     numerical_cols = {'Longitude': row['Longitude'], 'Latitude': row['Latitude']}
    #     nominal_cols = {'continent': row['continent']}
    #     nodes[index] = Node(index, numerical_cols, nominal_cols)

    #nominal dataset
    # for index, row in df.iterrows():
    #     nominal_cols = {'education': row['education'], 'marital_status': row['marital_status'],
    #                     'occupation': row['occupation'], 'relationship': row['relationship'], 'race': row['race'],
    #                     'sex': row['sex'], 'income': row['income']}
    #     nodes[index] = Node(index, None, nominal_cols)

    print(nodes[0])

    if nodes[0].numerical_attributes != None:
        prepare_numerical_data_for_nodes(nodes)


    algorithm = CLARANS(nodes,num_of_iterations=1000, max_neighbours_examined=80, num_of_clusters=3)
    best_medoids, clustered_nodes = algorithm.run()


    x = [[], [], []]
    y = [[], [], []]

    for n in clustered_nodes:
        x[n.cluster].append(n.x)
        y[n.cluster].append(n.y)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x[0], y[0], s=10, c='r', marker="s", label='0')
    ax1.scatter(x[1], y[1], s=10, c='g', marker="s", label='1')
    ax1.scatter(x[2], y[2], s=10, c='b', marker="s", label='2')
    ax1.scatter(best_medoids[0].x, best_medoids[0].y, s=30, c='r', marker="^", label='0')
    ax1.scatter(best_medoids[1].x, best_medoids[1].y, s=30, c='g', marker="^", label='1')
    ax1.scatter(best_medoids[2].x, best_medoids[2].y, s=30, c='b', marker="^", label='2')
    plt.title("2d data with 3 clusters")
    plt.legend(loc='upper left')
    plt.show()


    """
    x_values = []
    y_values = []

    for m in best_medoids:
        x_values.append([])
        y_values.append([])

    for n in clustered_nodes:
        for col_name, value in n.numerical_attributes.items():
            if col_name == 'Longitude':
                x_values[n.cluster].append(value)
            if col_name == 'Latitude':
                y_values[n.cluster].append(value)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(len(best_medoids)):
        ax1.scatter(x_values[i], y_values[i], s=10, c=colors[i], marker="s", label=i)
        ax1.scatter(best_medoids[i].numerical_attributes['Longitude'], best_medoids[i].numerical_attributes['Latitude'], s=30, c=colors[i], marker="^", label=i)

    # plt.legend(loc='upper left')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title("Geographical locations - data with 6 clusters")
    plt.xlim(-180,180)
    plt.ylim(-90, 90)
    plt.show()
    """

    """
    x_values = []
    y_values = []

    for m in best_medoids:
        x_values.append([])
        y_values.append([])

    for n in clustered_nodes:
        for col_name, value in n.nominal_attributes.items():
            if col_name == 'education':
                x_values[n.cluster].append(value)
            if col_name == 'income':
                y_values[n.cluster].append(value)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(len(best_medoids)):
        ax1.scatter(x_values[i], y_values[i], s=10, c=colors[i], marker="s", label=i)
        ax1.scatter(best_medoids[i].nominal_attributes['education'], best_medoids[i].nominal_attributes['income'], s=30, c='k', marker="^", label=i)

    plt.xlabel('Education')
    plt.ylabel('Income')
    plt.title("Nominal data with 6 clusters")
    plt.show()
    """