#!/usr/bin/env python3
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from itertools import product


def clustering_grid_search(X, algorithm, param_grid):
    """Grid search to find the optimal hyperparameters for a clustering algorithm.

    :param X: DataFrame to cluster
    :type X: object
    :param algorithm: Clustering algorithm to use
    :type algorithm: str
    :param param_grid: Parameters and values to use to tune the algorithm
    :type param_grid: dict
    :raises ValueError: Unsupported algorithm specified
    """
    # Initialize the best scores and parameters for each metric
    best_dbscore = np.inf  # Lower Davies-Bouldin Score is better
    best_silscore = -1    # Higher Silhouette Score is better
    best_chscore = -1     # Higher Calinski-Harabasz Score is better
    best_dbparams = best_silparams = best_chparams = None

    # Generate all combinations of parameters
    param_combinations = list(
        product(*(param_grid[name] for name in param_grid)))

    # Iterate over all combinations of parameters
    for params in param_combinations:
        # Unpack parameters
        kwargs = dict(zip(param_grid.keys(), params))

        # Initialize the clustering model based on the input algorithm
        if algorithm == 'DBSCAN':
            model = DBSCAN(**kwargs)
        elif algorithm == 'HAC':
            model = AgglomerativeClustering(**kwargs)
        elif algorithm == 'HDBSCAN':
            model = HDBSCAN(**kwargs)
        elif algorithm == 'MeanShift':
            model = MeanShift(**kwargs)
        elif algorithm == 'GMM':
            model = GaussianMixture(**kwargs)
        else:
            raise ValueError("Unsupported algorithm specified")

        # Fit the model and predict labels
        labels = model.fit_predict(X)

        # Calculate scores only if more than one cluster and not all noise
        if len(set(labels)) > 1 and not (len(set(labels)) == 2 and -1 in labels):
            try:
                silscore = silhouette_score(X, labels)
                if silscore > best_silscore:
                    best_silscore = silscore
                    best_silparams = kwargs
            except ValueError:
                pass  # Silhouette score cannot be calculated for a single cluster

            try:
                dbscore = davies_bouldin_score(X, labels)
                if dbscore < best_dbscore:
                    best_dbscore = dbscore
                    best_dbparams = kwargs
            except ValueError:
                pass

            try:
                chscore = calinski_harabasz_score(X, labels)
                if chscore > best_chscore:
                    best_chscore = chscore
                    best_chparams = kwargs
            except ValueError:
                pass

    print(
        f"Best params for Davies-Bouldin Score in {algorithm}: {best_dbparams}")
    print(f"Best Davies-Bouldin Score: {np.round(best_dbscore, 3)}")
    print(f"Best params for Silhouette Score in {algorithm}: {best_silparams}")
    print(f"Best Silhouette Score: {np.round(best_silscore, 3)}")
    print(
        f"Best params for Calinski-Harabasz Score in {algorithm}: {best_chparams}")
    print(f"Best Calinski-Harabasz Score: {np.round(best_chscore, 3)}")


"""
Example param_grid:
param_grid_dbscan = {
    'eps': [0.170, 0.172, 0.1723, 0.1725, 0.175, 0.177, 0.185],    np.arange(0.170, 0.177, 0.001).tolist()
    'min_samples': [4, 5, 6, 7, 8, 9]
}
clustering_grid_search(X, 'DBSCAN', param_grid_dbscan)

or 

param_grid_dbscan = {
    'eps': np.arange(0.170, 0.177, 0.001).tolist(),
    'min_samples': np.arange(4, 9, 1).tolist()
}
clustering_grid_search(X, 'DBSCAN', param_grid_dbscan)
"""
