import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, x_new, k=3):
    distances = np.linalg.norm(X_train - x_new, axis=1)
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    return Counter(k_labels).most_common(1)[0][0]

# Dados simples
X_train = np.array([[1,2], [2,3], [3,3], [6,5], [7,7]])
y_train = np.array(['A', 'A', 'A', 'B', 'B'])

x_new = np.array([2,2])
print(knn_predict(X_train, y_train, x_new, k=3))
