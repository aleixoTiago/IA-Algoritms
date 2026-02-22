class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X, y):
        self.classes = set(y)
        n_samples = len(y)

        # Probabilidades das classes P(C)
        for c in self.classes:
            self.class_probs[c] = y.count(c) / n_samples
            self.feature_probs[c] = {}

        n_features = len(X[0])

        # Probabilidades condicionais P(x_i | C)
        for c in self.classes:
            X_c = [X[i] for i in range(n_samples) if y[i] == c]

            for j in range(n_features):
                values = [x[j] for x in X_c]
                self.feature_probs[c][j] = {}

                for v in set(values):
                    self.feature_probs[c][j][v] = values.count(v) / len(values)

    def predict(self, X):
        predictions = []
        for x in X:
            scores = {}

            for c in self.classes:
                prob = self.class_probs[c]

                for j, value in enumerate(x):
                    prob *= self.feature_probs[c][j].get(value, 0)

                scores[c] = prob

            predictions.append(max(scores, key=scores.get))

        return predictions


# ---------------- TESTE ---------------- #

X = [
    ["Sim", "Sim"],
    ["Sim", "Sim"],
    ["Sim", "Não"],
    ["Não", "Sim"],
    ["Não", "Não"],
    ["Não", "Não"],
    ["Sim", "Não"],
    ["Não", "Sim"]
]

y = ["Passa", "Passa", "Passa", "Passa",
     "Reprova", "Reprova", "Reprova", "Reprova"]

model = NaiveBayes()
model.fit(X, y)

novo_aluno = [["Sim", "Não"]]
print(model.predict(novo_aluno))  # esperado: ['Passa']
