from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# Define two vectors
embedding1 = np.array([1.5, 2.0, 3.5])
embedding2 = np.array([4.0, 1.0, 2.5])

# Calculate Euclidean distance
euclidean_dist = euclidean_distances([embedding1], [embedding2])[0][0]
print("Euclidean distance:", euclidean_dist)
