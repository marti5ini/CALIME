import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sdv.metrics.tabular import LogisticDetection, KSTest, GMLogLikelihood, ContinuousKLDivergence, SVCDetection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class Plausibility:

    def __init__(self, train_data, synthetic_data, explainer_name):
        """
            Args:
                train_data: pandas DataFrame, real data
                synthetic_data: pandas DataFrame, synthesized data
                explainer_name: string, explainer name
        """
        self.train_data = train_data
        self.synthetic_data = synthetic_data
        self.explainer_name = explainer_name

    def avg_statistics_metric(self, aggregate=True):
        """
            Args:
                aggregate: boolean, default = True
                    it returns a single float value or returns a pandas DataFrame
                    containing the score that each metric obtained.
            Returns:
                a single float value or a pandas.DataFrame
        """
        ks = KSTest.compute(self.train_data, self.synthetic_data)
        no_normalized = GMLogLikelihood.compute(self.train_data.fillna(0),
                                                self.synthetic_data.fillna(0))
        gm = GMLogLikelihood.normalize(no_normalized)
        kl = ContinuousKLDivergence.compute(self.train_data, self.synthetic_data)

        if aggregate:
            return np.mean([ks, gm, kl])
        else:
            return pd.DataFrame({'explainer': [self.explainer_name], 'KSTest': [ks], 'GMLogLikelihood': [gm],
                                 'ContinuousKLDivergence': [kl]})

    def avg_detection_metric(self, aggregate=True):
        """
            Args:
                aggregate: boolean, default = True
                    it returns a single float value or returns a pandas DataFrame
                    containing the score that each metric obtained.
            Returns:
                a single float value or a pandas.DataFrame
        """
        ld = LogisticDetection.compute(self.train_data, self.synthetic_data)
        svc = SVCDetection.compute(self.train_data, self.synthetic_data)

        if aggregate:
            return np.mean([ld, svc])
        else:
            return pd.DataFrame({'explainer': [self.explainer_name], 'LogisticDetection': [ld], 'SVCDetection': [svc]})

    def avg_minimum_distance(self):
        # normalization by row for each type of data
        normalized_train_data, normalized_synthetic_data = self.__normalize()

        min_distances = []

        for generated_row in normalized_synthetic_data:
            if np.isnan(generated_row).any():
                pass
            else:
                dist = cdist(generated_row[None, ], normalized_train_data)
                min_distances.append(np.min(dist))

        return np.mean(min_distances)

    def avg_outlier_score(self, aggregate=True, normalize=True):
        """
            Args:
                aggregate: boolean, default = True
                    it returns a single float value or returns a pandas DataFrame
                    containing the score that each metric obtained.
            Returns:
                a single float value or a pandas.DataFrame
        """
        outliers = {}

        if normalize:
            # normalization by row for each type of data
            normalized_train_data, normalized_synthetic_data = self.__normalize()
        else:
            normalized_train_data, normalized_synthetic_data = self.train_data, self.synthetic_data

        estimators = [IsolationForest(), LocalOutlierFactor(novelty=True)]
        estimators_names = ['isolationF', 'lof']

        for estimator, name in zip(estimators, estimators_names):
            if name == 'isolationF':
                model = estimator.fit(self.train_data)
                y_pred = model.predict(self.synthetic_data)
            else:
                model = estimator.fit(normalized_train_data)
                y_pred = model.predict(normalized_synthetic_data)

            if name == 'abod':
                anomalies = y_pred[y_pred == 1].size
            else:
                anomalies = y_pred[y_pred == -1].size

            outliers[name] = anomalies

        if aggregate:
            return sum(outliers.values()) / len(outliers)
        else:
            return pd.DataFrame({'explainer': [self.explainer_name], 'isolationF': [outliers['isolationF']], 'lof': [outliers['lof']]})

    def __normalize(self):

        train_data = self.train_data.values
        normalized_train_data = (train_data - np.min(train_data, axis=1, keepdims=True)) / \
                                (np.max(train_data, axis=1, keepdims=True) -
                                 np.min(train_data, axis=1, keepdims=True))

        synthetic_data = self.synthetic_data.values
        normalized_synthetic_data = (synthetic_data - np.min(synthetic_data, axis=1, keepdims=True)) / \
                                    (np.max(synthetic_data, axis=1, keepdims=True) -
                                     np.min(synthetic_data, axis=1, keepdims=True)) + 1e-8

        return normalized_train_data, normalized_synthetic_data

