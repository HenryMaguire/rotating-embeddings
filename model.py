import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch


class SVDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.U_ = None
        self.Vh_ = None
        self.classes_ = None
        self.class_embeddings_ = None

    def fit(self, X, y):
        """
        X are expected to be embeddings, and y are expected to be class labels.
        """
        # Convert inputs to torch tensors
        X = torch.FloatTensor(X)
        self.classes_ = np.unique(y)

        self.class_embeddings_ = torch.FloatTensor(class_embeddings)

        # Get target embeddings for each class
        target_indices = torch.LongTensor(
            [self.classes_.tolist().index(label) for label in y]
        )
        target_embeddings = self.class_embeddings_[target_indices]

        # Compute SVD
        C = target_embeddings.T @ X
        U, _, Vh = torch.linalg.svd(C)

        self.U_ = U
        self.Vh_ = Vh
        return self

    def predict(self, X):
        if not isinstance(X, torch.FloatTensor):
            X = torch.FloatTensor(X)
        if X.ndim == 1:
            X = X.unsqueeze(0)

        transformed = self.U_ @ self.Vh_ @ X.T
        similarities = torch.nn.functional.cosine_similarity(
            transformed.T.unsqueeze(1), self.class_embeddings_.unsqueeze(0), dim=2
        )
        predictions = similarities.argmax(dim=1).numpy()
        return self.classes_[predictions]

    def predict_proba(self, X):
        if not isinstance(X, torch.FloatTensor):
            X = torch.FloatTensor(X)
        if X.ndim == 1:
            X = X.unsqueeze(0)

        transformed = self.U_ @ self.Vh_ @ X.T
        similarities = torch.nn.functional.cosine_similarity(
            transformed.T.unsqueeze(1), self.class_embeddings_.unsqueeze(0), dim=2
        )
        return torch.softmax(similarities, dim=1).numpy()


# Usage example:
clf = SVDClassifier()
clf.fit(train_input_embeddings, train_labels)

# Evaluate
y_pred = clf.predict(test_input_embeddings)
print("Accuracy:", (y_pred == test_labels).mean())

# Get probabilities
probs = clf.predict_proba(test_input_embeddings)
