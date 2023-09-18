"""
Base class of the tabular explainer
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.competitors.faster_lime.utils import dict_disc_to_bin


class BaseTabularExplainer(ABC):


    def __init__(self, training_data, feature_names=None,
                 categorical_feature_idxes=None,
                 discretizer='quartile', **kwargs):
        """
        Args:
            training_data (np.ndarray): Training data to measure training data statistics
            feature_names (list): List of feature names
            categorical_feature_idxes (list): List of idxes of features that are categorical
            discretizer (str): Discretization resolution

        Assumptions:
            * Data only contains categorical and/or numerical data
            * Categorical data is already converted to ordinal labels (e.g. via scikit-learn's
                OrdinalEncoder)

        """
        self.training_data = training_data
        self.num_features = self.training_data.shape[1]

        # Parse columns
        if feature_names is not None:
            # TODO input validation
            self.feature_names = list(feature_names)
        else:
            self.feature_names = list(range(self.num_features))
        self.categorical_feature_idxes = categorical_feature_idxes
        if self.categorical_feature_idxes:
            self.categorical_features = [self.feature_names[i] for i in
                                         self.categorical_feature_idxes]
            self.numerical_features = [f for f in self.feature_names if
                                       f not in self.categorical_features]
            self.numerical_feature_idxes = [idx for idx in range(self.num_features) if
                                            idx not in self.categorical_feature_idxes]
        else:
            self.categorical_features = []
            self.numerical_features = self.feature_names
            self.numerical_feature_idxes = list(range(self.num_features))

        # Some book-keeping: keep track of the original indices of each feature
        self.dict_num_feature_to_idx = {feature: idx for (idx, feature) in
                                        enumerate(self.numerical_features)}
        self.dict_feature_to_idx = {feature: idx for (idx, feature) in
                                    enumerate(self.feature_names)}
        self.list_reorder = [self.dict_feature_to_idx[feature] for feature in
                             self.numerical_features + self.categorical_features]

        # Get training data statistics
        # Numerical feature statistics
        if self.numerical_features:
            training_data_num = self.training_data[:, self.numerical_feature_idxes]
            self.sc = StandardScaler(with_mean=False)
            self.sc.fit(training_data_num)
            self.percentiles = dict_disc_to_bin[discretizer]
            self.all_bins_num = np.percentile(training_data_num, self.percentiles, axis=0).T

        # Categorical feature statistics
        if self.categorical_features:
            training_data_cat = self.training_data[:, self.categorical_feature_idxes]
            self.dict_categorical_hist = {
                feature: np.bincount(training_data_cat[:, idx]) / self.training_data.shape[0] for
                (idx, feature) in enumerate(self.categorical_features)
            }

        # Another mapping from feature to type
        self.dict_feature_to_type = {
            feature: 'categorical' if feature in self.categorical_features else 'numerical' for
            feature in self.feature_names}

    @abstractmethod
    def explain_instance(self, **kwargs):
        raise NotImplementedError
