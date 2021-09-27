import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Loading the dataset
digits = load_digits()

# Scaling the dataset according
data = scale(digits.data)

# Used to get labels
y = digits.target

# Number of classifications
k = len(np.unique(y))

samples, features = data.shape


# Used to benchmark kmeans algorithm from sklearn docs
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


# Classifier
clf = KMeans(n_clusters=k, init="random", n_init=k)
bench_k_means(clf, "1", data)
