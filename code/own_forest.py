import time
import math
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

from own_tree import UniversalTreeRegressor


def _without_replacement(rng, n_max, size):
    bag = set()
    while size > 0:
        vals = np.unique(rng.integers(0, n_max, size))
        cur = size if vals.shape[0] > size else vals.shape[0]
        size -= cur
        for val in vals[:cur]:
            bag.add(val)
    return np.array(list(bag))


def _generate_indices(rng, n, size, bootstrap):
    if bootstrap:
        return rng.integers(0, n, size)
    return _without_replacement(rng, n, size)


def _generate_bagging_indices(rng, n, fraction, bootstrap):
    size = math.ceil(fraction * n)
    return _generate_indices(rng, n, size, bootstrap)


def _fit_k_estimators(k_estimators, seeds, ensemble, X, y, G, coefs, trace):
    estimators = []
    estimators_features = []
    estimators_fit_time = []

    for i in range(k_estimators):
        rng = np.random.default_rng(seeds[i])
        samples = _generate_bagging_indices(
            rng, X.shape[0], ensemble.sub_sample_size, ensemble.bootstrap)
        features = _generate_bagging_indices(
            rng, X.shape[1], ensemble.sub_feature_size, ensemble.bootstrap)

        estimator = ensemble.make_estimator(random_state=seeds[i])

        if trace:
            start = time.time()
            estimator.fit(X[samples][:, features], G[samples], coefs)

            estimators_fit_time.append(time.time() - start)
        else:
            estimator.fit(X[samples][:, features], G[samples], coefs)

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features, estimators_fit_time


def _predict_k_estimators(estimators, estimators_features, X):
    return sum(
        estimator.predict(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )


class RandomForestMSE:
    DEFAULT_SUB_SAMPLE_SIZE = 1.0
    DEFAULT_SUB_FEATURE_SIZE = 1.0 / 3.0

    def __init__(
            self,
            n_estimators,
            sub_sample_size=None,
            sub_feature_size=None,
            bootstrap=True,
            n_jobs=5,
            random_state=None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        sub_feature_size : float
            The size of feature set for each tree. If None then use one-third
            of all features.
        """
        self.n_estimators = n_estimators
        if sub_sample_size is None:
            self.sub_sample_size = self.DEFAULT_SUB_SAMPLE_SIZE
        else:
            self.sub_sample_size = sub_sample_size
        if trees_parameters is None:
            # self.trees_parameters = boost_tree.TreeRegressionMultiMSEParams()
            self.trees_parameters = {'max_depth': 5, 'min_samples_split': 2, 'min_impurity_decrease': 0.0}
        else:
            self.trees_parameters = trees_parameters
        if sub_feature_size is None:
            self.sub_feature_size = self.DEFAULT_SUB_FEATURE_SIZE
        else:
            self.sub_feature_size = sub_feature_size
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimators = None
        self.estimators_features = None
        self.estimators_fit_time = None
        self.estimators_fit_time_mean = None
        self.estimators_fit_time_all = None
        self.estimators_RMSE = None
        self.estimators_RMSE_train = None
        self.sizes = np.full(self.n_jobs, self.n_estimators // self.n_jobs)
        self.sizes[:self.n_estimators % self.n_jobs] += 1
        self.starts = np.hstack([[0], np.cumsum(self.sizes)])

    def fit(self, X, y, G, coefs, trace=False, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        
        G : numpy ndarray
            Array of size n_objects, 1 + n_algorithms
            Where first column is true y values
        
        coefs: numpy ndarray
            Array of size 1 + n_algorithms

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        global_start, global_end = 0, 0

        if trace:
            global_start = time.time()

        rng = np.random.default_rng(self.random_state)
        seeds = _without_replacement(rng, 100000000, self.n_estimators)

        results = Parallel(n_jobs=self.n_jobs)([
            delayed(_fit_k_estimators)(
                self.sizes[i],
                seeds[self.starts[i]:self.starts[i + 1]],
                self, X, y, G, coefs, trace
            )
            for i in range(self.n_jobs)
        ])

        if trace:
            global_end = time.time()

        self.estimators = list(itertools.chain.from_iterable(job[0] for job in results))
        self.estimators_features = list(itertools.chain.from_iterable(job[1] for job in results))

        if trace:
            self.estimators_fit_time = np.array(list(itertools.chain.from_iterable(job[2] for job in results)))
            self.estimators_fit_time_mean = sum(self.estimators_fit_time) / self.n_estimators
            self.estimators_fit_time_all = global_end - global_start
            self.estimators_RMSE_train = self.calc_RMSE(X, y)

        if trace and X_val is not None and y_val is not None:
            self.estimators_RMSE = self.calc_RMSE(X_val, y_val)

        return self

    def calc_RMSE(self, X, y):
        result = []
        temp = np.zeros(X.shape[0])
        for n, estimator, features in zip(
                range(1, self.n_estimators + 1),
                self.estimators,
                self.estimators_features
        ):
            temp += estimator.predict(X[:, features])
            result.append(
                mean_squared_error(y, temp / n, squared=False)
            )
        return np.array(result)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        results = Parallel(n_jobs=self.n_jobs)([
            delayed(_predict_k_estimators)(
                self.estimators[self.starts[i]:self.starts[i + 1]],
                self.estimators_features[self.starts[i]:self.starts[i + 1]],
                X,
            )
            for i in range(self.n_jobs)
        ])
        return sum(results) / self.n_estimators

    def stats(self):
        return {
            'n_trees': np.array(list(range(1, self.n_estimators + 1))),
            'estimators': self.estimators,
            'features': self.estimators_features,
            'fit_time': self.estimators_fit_time,
            'fit_time_all': self.estimators_fit_time_all,
            'fit_time_mean': self.estimators_fit_time_mean,
            'RMSE': self.estimators_RMSE,
            'RMSE_train': self.estimators_RMSE_train
        }

    def make_estimator(self, random_state):
        # return boost_tree.TreeRegressorMultiMSE(self.trees_parameters)
        return UniversalTreeRegressor(**self.trees_parameters)