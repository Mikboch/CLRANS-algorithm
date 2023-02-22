import math
import random
import numpy as np
import time

class CLARANS:
    def __init__(self, data, num_of_iterations=1000, max_neighbours_examined=80, num_of_clusters=3):
        self.num_of_iterations = num_of_iterations
        self.max_neighbours_examined = max_neighbours_examined
        self.num_of_clusters = num_of_clusters
        self.nodes = data

        self.min_cost = math.inf
        self.best_medoids = []
        self.best_medoids_for_iteration = []

        self.medoids_distances_matrix = np.zeros((self.num_of_clusters, len(self.nodes)))

    def run(self):
        start_time = time.time()
        i = 1
        while i <= self.num_of_iterations:
            self.medoids_distances_matrix.fill(0)
            self.best_medoids_for_iteration = []

            current_medoids_indices = np.random.choice(len(self.nodes), self.num_of_clusters, replace=False)
            current_medoids = np.take(self.nodes, current_medoids_indices)

            self.medoids_distances_matrix = self.set_distances_for_all_medoids(current_medoids)
            self.assign_objects_to_clusters(self.medoids_distances_matrix)
            best_cost_for_iteration = self.get_total_cost(self.medoids_distances_matrix)

            j = 1
            while j <= self.max_neighbours_examined:
                new_medoid_index = self.get_random_medoid(current_medoids)
                new_medoids = self.random_medoid_replacement(current_medoids, new_medoid_index)
                self.medoids_distances_matrix[new_medoid_index] = self.set_distances_for_medoid(new_medoid_index)
                self.assign_objects_to_clusters(self.medoids_distances_matrix)
                new_cost = self.get_total_cost(self.medoids_distances_matrix)

                if new_cost < best_cost_for_iteration:
                    current_medoids = new_medoids.copy()
                    self.best_medoids_for_iteration = new_medoids.copy()
                    best_cost_for_iteration = new_cost
                    continue
                else:
                    j += 1

                    if j <= self.max_neighbours_examined:
                        continue
                    elif best_cost_for_iteration < self.min_cost:

                        self.min_cost = best_cost_for_iteration
                        if len(self.best_medoids_for_iteration) != self.num_of_clusters:
                            self.best_medoids = current_medoids.copy()
                        else:
                            self.best_medoids = self.best_medoids_for_iteration.copy()


                    i+=1
                    if i > self.num_of_iterations:
                        self.medoids_distances_matrix = self.set_distances_for_all_medoids(self.best_medoids)
                        self.assign_objects_to_clusters(self.medoids_distances_matrix)
                        execution_time = time.time() - start_time
                        print("Execution time: " + str(execution_time))
                        return self.best_medoids, self.nodes


    def get_random_medoid(self, medoids):
        new_medoid_index = random.randint(0, len(medoids)-1)
        return new_medoid_index

    def random_medoid_replacement(self, medoids, replaced_medoid_index):
        random_new_medoid_index = random.randint(0, len(self.nodes)-1)
        medoids[replaced_medoid_index] = self.nodes[random_new_medoid_index]
        return medoids


    def get_euclidean_distance_for_points(self, node_A, node_B):
        return math.sqrt(math.pow(node_A.x - node_B.x, 2) + math.pow(node_A.y - node_B.y, 2))

    def get_gower_distance(self, node_A, node_B):
        total_distance = 0

        #distances for numerical attributes
        if node_A.numerical_attributes != None:
            for col_name, value in node_A.numerical_attributes.items():
                if col_name in node_B.numerical_attributes and col_name in node_A.numerical_attributes_range:
                    total_distance += abs(node_A.numerical_attributes[col_name] - node_B.numerical_attributes[col_name])/node_A.numerical_attributes_range[col_name]

        # distances for nominal attributes
        if node_A.nominal_attributes != None:
            for col_name, value in node_A.nominal_attributes.items():
                if col_name in node_B.nominal_attributes and node_A.nominal_attributes[col_name] != node_B.nominal_attributes[col_name]:
                    total_distance += 1
                else:
                    pass

        return total_distance

    def set_distances_for_medoid(self, medoid_index):
        distances_for_medoid = np.zeros(len(self.nodes))
        for i in range(len(self.nodes)):
            if(self.nodes[i].x != None and self.nodes[i].y != None):
                distances_for_medoid[i] = self.get_euclidean_distance_for_points(self.nodes[medoid_index], self.nodes[i])
            else:
                distances_for_medoid[i] = self.get_gower_distance(self.nodes[medoid_index], self.nodes[i])
        return distances_for_medoid

    def set_distances_for_all_medoids(self, medoids):
        medoids_distances_matrix = np.zeros((len(medoids), len(self.nodes)))
        for i in range(len(medoids)):
            medoids_distances_matrix[i] = self.set_distances_for_medoid(medoids[i].id_in_dataset)

        return medoids_distances_matrix

    def assign_objects_to_clusters(self, medoids_distances_matrix):
        cluster_id_for_objects = np.argmin(medoids_distances_matrix, axis=0)

        for i in range(len(self.nodes)):
            self.nodes[i].cluster = cluster_id_for_objects[i]

    def get_total_cost(self, medoids_distances_matrix):
        minimal_distances_for_objects = np.amin(medoids_distances_matrix, axis=0)
        return np.sum(minimal_distances_for_objects)