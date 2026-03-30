"""
K-Means Clustering
Medium
Machine Learning


Your task is to write a Python function that implements the k-Means clustering algorithm. This function should take specific inputs and produce a list of final centroids. k-Means clustering is a method used to partition n points into k clusters. The goal is to group similar points together and represent each group by its center (called the centroid).
Function Inputs:
points: A list of points, where each point is a tuple of coordinates (e.g., (x, y) for 2D points or (x, y, z) for 3D points). All points must have the same dimensionality.
k: An integer representing the number of clusters to form
initial_centroids: A list of initial centroid points, each a tuple of coordinates with the same dimensionality as the input points
max_iterations: An integer representing the maximum number of iterations to perform
Function Output:
A list of the final centroids of the clusters, where each centroid is rounded to the nearest fourth decimal.
Example:
Input:
points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
Output:
[(1, 2), (10, 2)]
Reasoning:
Given the initial centroids and a maximum of 10 iterations, the points are clustered around these points, and the centroids are updated to the mean of the assigned points, resulting in the final centroids which approximate the means of the two clusters. The exact number of iterations needed may vary, but the process will stop after 10 iterations at most.
"""

def k_means_clustering(points, k, initial_centroids, max_iterations):
    def euclidean_distance(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def compute_centroid(cluster):
        dims = len(cluster[0])
        return tuple(
            round(sum(p[d] for p in cluster) / len(cluster), 4)
            for d in range(dims)
        )

    centroids = list(initial_centroids)

    for _ in range(max_iterations):
        # Assignment step: assign each point to nearest centroid
        clusters = [[] for _ in range(k)]
        for point in points:
            nearest = min(range(k), key=lambda i: euclidean_distance(point, centroids[i]))
            clusters[nearest].append(point)

        # Update step: recompute centroids
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                new_centroids.append(compute_centroid(cluster))
            else:
                new_centroids.append(centroids[i])  # keep old if cluster is empty

        # Check for convergence
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return centroids