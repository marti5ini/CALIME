import numpy as np
from ctgan import CTGANSynthesizer
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.faster_lime.utils import dict_disc_to_bin


def ridge_solve(tup):
    data_synthetic_onehot, model_pred, weights = tup
    solver = Ridge(alpha=1, fit_intercept=True)
    solver.fit(data_synthetic_onehot,
               model_pred,
               sample_weight=weights.ravel())

    # Get explanations
    importance = solver.coef_[
        data_synthetic_onehot[0].ravel() == 1].ravel()
    return importance


class NumpyRobustTabularExplainer:

    def __init__(self, training_data, ctgan_sampler=None, discriminator=None,
                 feature_names=None, categorical_feature_idxes=None,
                 discretizer='decile', ctgan_epochs=100, ctgan_verbose=False, use_cat_for_ctgan=True,
                 ctgan_params={}, measure_distance='mix', nearest_neighbors=0.8, use_onehot=True,
                 discriminator_threshold=0.5, **kwargs):
        """

        Args:
            training_data (np.ndarray): Training data to measure training data statistics
            ctgan_sampler (Optional[CTGANSampler]): A CTGAN model
            discriminator (Optional[Adversarial_Model]): Discriminator
            feature_names (list): List of feature names
            categorical_feature_idxes (list): List of idxes of features that are categorical
            discretizer (str): Discretization resolution
            ctgan_epochs (int): Number of epochs to train the CTGAN
            use_cat_for_ctgan (bool): Whether to use categorical features in training CTGAN
            ctgan_params (dict): Additional params for CTGAN
            measure_distance (str): "raw" - measure distances on raw synthetic samples
                                    "onehot" - measure hamming distances on onehot samples
            nearest_neighbors (float): What proportion of nearest neighbors to keep
            use_onehot (bool): Whether to use one-hot samples
        """
        self.training_data = training_data
        self.num_features = self.training_data.shape[1]
        self.discriminator = discriminator
        self.measure_distance = measure_distance
        self.nearest_neighbors = nearest_neighbors
        self.use_onehot = use_onehot
        self.discriminator_threshold = discriminator_threshold

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
                feature: np.bincount(training_data_cat[:, idx].astype(np.int64)) / float(
                    self.training_data.shape[0]) for
                (idx, feature) in enumerate(self.categorical_features)
            }

        # Another mapping from feature to type
        self.dict_feature_to_type = {
            feature: 'categorical' if feature in self.categorical_features else 'numerical' for
            feature in self.feature_names}

        # Finally, if ctgan provided is None, train a new one
        if ctgan_sampler is None:
            self.ctgan_sampler = CTGANSynthesizer(verbose=ctgan_verbose, **ctgan_params)

            if use_cat_for_ctgan:
                self.ctgan_sampler.fit(training_data, categorical_feature_idxes, ctgan_epochs)
            else:
                training_data_num = training_data[:, self.numerical_feature_idxes]
                self.ctgan_sampler.fit(training_data_num, epochs=ctgan_epochs)

            self.ctgan_uses_cat = use_cat_for_ctgan

        else:
            self.ctgan_sampler = ctgan_sampler

    def kernel_fn(self, distances, kernel_width):
        return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    def discretize(self, X, percentiles=[25, 50, 75], all_bins=None):
        if all_bins is None:
            all_bins = np.percentile(X, percentiles, axis=0).T
        return (np.array([np.digitize(a, bins)
                          for (a, bins) in zip(X.T, all_bins)]).T, all_bins)

    def explain_instance(self, data_row, predict_fn, num_estimators=1, label=0, num_samples=5000,
                         num_features=10,
                         kernel_width=None, **kwargs):
        # Scale the data
        data_row = data_row.reshape((1, -1))

        # Sample data using the CTGAN
        data_samples = None
        while data_samples is None:
            try:
                data_samples = self.ctgan_sampler.sample(
                    int(num_estimators) * int(num_samples), test_instance=data_row)
            except Exception as e:
                data_samples = None

        # Prune using the discriminator
        if self.discriminator:
            if self.ctgan_uses_cat:
                disc_pred = self.discriminator.perturbation_identifier.predict_proba(
                    data_samples[:, self.discriminator.numerical_cols])
            else:
                disc_pred = self.discriminator.perturbation_identifier.predict_proba(
                    data_samples)

            data_samples = data_samples[disc_pred[:, 1] >= self.discriminator_threshold]

        for batch_idx in range(num_estimators):
            if self.ctgan_uses_cat:
                data_samples[batch_idx * num_samples] = data_row.ravel()
            else:
                data_samples[batch_idx * num_samples] = data_row[:,
                                                        self.numerical_feature_idxes].ravel()

        # Update num_samples
        num_samples = data_samples.shape[0]

        # Split data into numerical and categorical data and process
        list_orig = []
        list_disc = []
        if self.numerical_features:
            if self.ctgan_uses_cat:
                data_num_synthetic = data_samples[:, self.numerical_feature_idxes]
            else:
                data_num_synthetic = data_samples
            # Discretize
            data_synthetic_num_disc, _ = self.discretize(data_num_synthetic, percentiles=self.percentiles,
                                                         all_bins=self.all_bins_num)
            list_disc.append(data_synthetic_num_disc)
            list_orig.append(data_num_synthetic)

        if self.categorical_features:
            # Sample from training distribution for each categorical feature
            if self.ctgan_uses_cat:
                data_cat_synthetic = data_samples[:, self.categorical_feature_idxes]
                list_disc.append(data_cat_synthetic)
                list_orig.append(data_cat_synthetic)
            else:
                # Need to generate sample categorical data without CTGAN
                # Sample from training distribution for each categorical feature
                data_cat = data_row[:, self.categorical_feature_idxes]
                list_buf = []
                for feature in self.categorical_features:
                    list_buf.append(np.random.choice(a=len(self.dict_categorical_hist[feature]),
                                                     size=(1, num_samples),
                                                     p=self.dict_categorical_hist[feature]))
                data_cat_original = data_cat_disc = np.concatenate(list_buf).T
                data_cat_original[0] = data_cat.ravel()
                data_cat_disc[0] = data_cat.ravel()
                list_disc.append(data_cat_disc)
                list_orig.append(data_cat_original)

        # Concatenate the data and reorder the columns
        data_synthetic_original = np.concatenate(list_orig, axis=1)
        data_synthetic_disc = np.concatenate(list_disc, axis=1)
        data_synthetic_original = data_synthetic_original[:, self.list_reorder]
        data_synthetic_disc = data_synthetic_disc[:, self.list_reorder]

        # Turn discretized data into onehot
        data_synthetic_onehot = OneHotEncoder(sparse=False).fit_transform(data_synthetic_disc)

        if self.measure_distance == 'raw':
            # Get distances between original sample and neighbors
            if self.numerical_features:
                data_num_scaled = self.sc.transform(data_num_synthetic)
                distances = cdist(data_num_scaled[:1], data_num_scaled).reshape(-1, 1)
            else:
                distances = cdist(data_synthetic_disc[:1], data_synthetic_disc).reshape(-1, 1)
        elif self.measure_distance == 'mix':
            # mix hamming and euclidean distances
            if self.categorical_features:
                data_cat = data_synthetic_onehot[:, self.categorical_feature_idxes]
                cat_distances = cdist(XA=data_cat[:1], XB=data_cat, metric='hamming').reshape(-1, 1)
            else:
                cat_distances = 0
            if self.numerical_features:
                data_num_scaled = self.sc.transform(data_num_synthetic)
                num_distances = cdist(data_num_scaled[:1], data_num_scaled).reshape(-1, 1)
            else:
                num_distances = 0
            distances = cat_distances + num_distances
        else:
            distances = cdist(
                XA=data_synthetic_onehot[:1],
                XB=data_synthetic_onehot,
                metric='hamming'
            ).reshape(-1, 1)

        # Limit to top k% of nearest neighbors
        arg_idx = np.argsort(distances.ravel())[:int(self.nearest_neighbors * len(distances))]
        data_synthetic_disc = data_synthetic_disc[arg_idx]
        distances = distances[arg_idx]
        data_synthetic_onehot = data_synthetic_onehot[arg_idx]

        # Get model predictions (i.e. groundtruth)
        data_synthetic_original = data_synthetic_original[arg_idx]
        model_pred = predict_fn(data_synthetic_original)

        # Weight distances according to some kernel (e.g. Gaussian)
        if kernel_width is None:
            kernel_width = np.sqrt(data_row.shape[1]) * 0.75
        weights = self.kernel_fn(distances, kernel_width=kernel_width).ravel()

        batch_size = num_samples
        importances = []

        if self.use_onehot:
            data = data_synthetic_onehot
        else:
            data = data_synthetic_disc


        iterator = ((data[batch_idx * batch_size:(batch_idx + 1) * batch_size],
                     model_pred[batch_idx * batch_size:(batch_idx + 1) * batch_size, label],
                     weights[batch_idx * batch_size:(batch_idx + 1) * batch_size]) for batch_idx
                    in range(num_estimators))

        for tup in iterator:
            # Solve
            importance = ridge_solve(tup)
            importances.append(importance)

        importances = np.mean(np.stack(importances), axis=0)
        explanations = sorted(list(zip(self.feature_names, importances)),
                              key=lambda x: x[1], reverse=True)[:num_features]

        # Add '<', '>', '=' etc. to the explanations
        def map_explanations(tup):
            feature, score = tup
            feature_idx = self.dict_feature_to_idx[feature]
            feature_type = self.dict_feature_to_type[feature]
            if feature_type == 'categorical':
                exp = '{} = {}'.format(feature, data_row[0][feature_idx])
            else:
                num_bin = int(data_synthetic_disc[0][feature_idx])
                bins = self.all_bins_num[self.dict_num_feature_to_idx[feature]]
                if num_bin == 0:
                    exp = '{} < {}'.format(feature, bins[0])
                elif num_bin >= len(bins) - 1:
                    exp = '{} > {}'.format(feature, bins[-1])
                else:
                    exp = '{} <= {} < {}'.format(bins[num_bin - 1], feature, bins[num_bin])

            return exp, score

        explanations = list(map(map_explanations, explanations))

        return explanations


if __name__ == '__main__':
    from experiments.utils.dataset_utils import get_and_preprocess_compas_data
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    data = get_and_preprocess_compas_data()
    X, y, _ = data['data'], data['target'], data['cols']
    X['unrelated_column'] = np.random.choice([0, 1], size=X.shape[0])
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    categorical_feature_name = ['two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M',
                                'sex_Female', 'sex_Male', 'race', 'unrelated_column']

    categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]

    explainer = NumpyRobustTabularExplainer(
        training_data=X_train.values,
        ctgan_sampler=None,
        feature_names=features,
        categorical_feature_idxes=categorical_feature_indcs,
        ctgan_epochs=2
    )

    exp = explainer.explain_instance(
        data_row=X_test.values[0],
        predict_fn=clf.predict_proba,
        num_samples=100
    )
    print(exp)
