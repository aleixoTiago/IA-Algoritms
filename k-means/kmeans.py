import numpy as np

def kmeans(X, k, max_iters=100):
    n, d = X.shape
    centroids = X[np.random.choice(n, k, replace=False)]

    for _ in range(max_iters):
        # atribuição
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # atualização
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

# exemplo
X = np.array([[1,1], [2,1], [4,3], [5,4]])
centroids, labels = kmeans(X, k=2)
print(centroids, labels)
