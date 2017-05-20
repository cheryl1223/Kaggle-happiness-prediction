from sklearn.preprocessing import LabelBinarizer,binarize
import numpy as np

class NaiveBayes:
    def __init__(self, binarize=.0, fit_prior=True,
                 class_prior=None):
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        self.feature_count_ += np.dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self):
        self.feature_log_prob_ = (np.log(self.feature_count_ ) -
                                  np.log(self.class_count_ .reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""

        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)

        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features, n_features_X))

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = np.dot(X, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll
    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of"
                                 " classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = (np.log(self.class_count_) -
                                     np.log(self.class_count_.sum()))
        else:
            self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

    def fit(self, X, y, sample_weight=None):

        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                       dtype=np.float64)
        self._count(X, Y)
        self._update_feature_log_prob()
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

