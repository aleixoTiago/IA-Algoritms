from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Criar dados artificiais
X, y = make_classification(n_samples=1000, n_features=2,
                           n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression(lr=0.1, n_iters=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print("Acurácia:", accuracy)
