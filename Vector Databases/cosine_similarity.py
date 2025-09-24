from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define two vectors
embedding1 = np.array([1.5, 2.0, 3.5])
embedding2 = np.array([4.0, 1.0, 2.5])

# Calculate cosine similarity
cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]
print("Cosine similarity:", cosine_sim)
