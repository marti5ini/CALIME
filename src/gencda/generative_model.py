import pickle

import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.gencda.distribution_fitting import *


class CausalDataGeneration:

    def __init__(self, graph, df):
        """

        :param graph: networkx.DiGraph
        :param df: pandas DataFrame
        """
        self.graph = graph
        self.df = df
        self.independent_dist = dict()
        self.dependent_dist = dict()

    @staticmethod
    def _ensemble_regression(parents, node):
        """
        This function fits base regressors on the original data and then
        aggregate their individual predictions to form a final prediction.
        :param parents: array-like
            The training input samples.
        :param node:  array-like of shape (n_samples,)
            The target values.
        :return object
            Instance of fitted estimator.
        """

        rf_param = {
            'n_estimators': [5, 10, 50, 100],
            'max_features': ['auto', 'sqrt'],
            #'min_samples_split': [3, 5, 10, 20],
            #'max_depth': [3, 5, 10, 20]
        }

        svr_param = {
            'C': list(np.logspace(0, 2, 4)),
            'kernel': ['linear', 'poly', 'rbf'],
            #'epsilon': [0.1, 0.2, 0.3, 0.5],
        }

        knr_param = {
            'n_neighbors': [5, 10],
            #'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'euclidean', 'manhattan']
        }

        gpr_param = {
        #   'kernel': [[RBF(i) for i in np.logspace(-1, 1, 2)], [DotProduct(sigma) for sigma in np.logspace(-1, 1, 2)]],
            'alpha': [1e-2, 1e-3]
        }

        params = {}
        params.update({'rf__' + k: v for k, v in rf_param.items()})
        params.update({'svr__' + k: v for k, v in svr_param.items()})
        params.update({'knn__' + k: v for k, v in knr_param.items()})
        params.update({'gpr__' + k: v for k, v in gpr_param.items()})

        ensemble_model = VotingRegressor(estimators=[('rf', RandomForestRegressor()), ('knn', KNeighborsRegressor()),
                                                     ('gpr', GaussianProcessRegressor()), ('svr', SVR())])

        model = RandomizedSearchCV(ensemble_model, param_distributions=params, scoring='neg_mean_absolute_error')
        model.fit(parents, node)

        return model.best_estimator_

    def _generate_column(self, node, parents_list=None):
        if len(parents_list) == 0:
            dst = Distribution()
            dst.Fit(self.df[node])
            self.independent_dist[node] = (dst.DistributionName, dst.params[dst.DistributionName])
        else:
            model = self._ensemble_regression(self.df[parents_list].values, self.df[node].values)
            self.dependent_dist[node] = pickle.dumps(model)
        return

    def generate_data(self):
        for node in nx.topological_sort(self.graph):
            predecessors = list(self.graph.predecessors(node))
            self._generate_column(node, predecessors)
        return