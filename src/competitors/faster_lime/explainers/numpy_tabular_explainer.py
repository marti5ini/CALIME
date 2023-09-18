import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder
import types

from src.competitors.faster_lime.explainers.base_tabular_explainer import BaseTabularExplainer
from src.competitors.faster_lime.utils import ridge_solve, kernel_fn, discretize, map_explanations


class NumpyTabularExplainer(BaseTabularExplainer):
    """
    A basic tabular explainer
    """

    def explain_instance(self, data_row, predict_fn, label=0, num_samples=5000, num_features=10,
                         kernel_width=None, **kwargs):
        """
        Explain a prediction on a given instance

        Args:
            data_row (np.ndarray): Data instance to explain
            predict_fn (func): A function which provides predictions from the target model
            label (int): The class to explain
            num_samples (int): Number of synthetic samples to generate
            num_features (int): Number of top features to return
            kernel_width (Optional[float]): Width of the Gaussian kernel when weighting synthetic samples

        Returns:
            (list) Tuples of feature and score, sorted by the score
        """
        # Scale the data
        data_row = data_row.reshape((1, -1))

        # Split data into numerical and categorical data and process
        list_orig = []
        list_disc = []
        if self.numerical_features:
            data_num = data_row[:, self.numerical_feature_idxes]
            data_num = self.sc.transform(data_num)
            data_synthetic_num = np.tile(data_num, (num_samples, 1))
            # Add noise
            data_synthetic_num = data_synthetic_num + np.random.normal(
                size=(num_samples, data_num.shape[1]))
            data_synthetic_num[0] = data_num.ravel()
            # Convert back to original domain
            data_synthetic_num_original = self.sc.inverse_transform(data_synthetic_num)
            # Discretize
            data_synthetic_num_disc, _ = discretize(data_synthetic_num_original, self.percentiles,
                                                    self.all_bins_num)
            list_disc.append(data_synthetic_num_disc)
            list_orig.append(data_synthetic_num_original)

        if self.categorical_features:
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

        # Get model predictions (i.e. groundtruth)
        model_pred = predict_fn(data_synthetic_original)

        # Get distances between original sample and neighbors
        if self.numerical_features:
            distances = cdist(data_synthetic_num[:1], data_synthetic_num).reshape(-1, 1)
        else:
            distances = cdist(data_synthetic_disc[:1], data_synthetic_disc).reshape(-1, 1)

        # Weight distances according to some kernel (e.g. Gaussian)
        if kernel_width is None:
            kernel_width = np.sqrt(data_row.shape[1]) * 0.75
        weights = kernel_fn(distances, kernel_width=kernel_width).ravel()

        # Turn discretized data into onehot
        data_synthetic_onehot = OneHotEncoder().fit_transform(data_synthetic_disc)

        # Solve
        tup = (data_synthetic_onehot, model_pred[:, label], weights)
        importances, prediction_score = ridge_solve(tup)
        explanations = sorted(list(zip(self.feature_names, importances)),
                              key=lambda x: x[1], reverse=True)[:num_features]

        # Add '<', '>', '=' etc. to the explanations
        explanations = [
            map_explanations(exp, data_row, self.dict_feature_to_idx, self.dict_feature_to_type, data_synthetic_disc[0],
                             self.all_bins_num, self.dict_num_feature_to_idx) for exp in explanations]

        ret_exp = types.SimpleNamespace(score=prediction_score)

        return ret_exp, data_synthetic_num


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import datasets

    data = datasets.load_breast_cancer()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    explainer = NumpyTabularExplainer(
        training_data=X_train,
        feature_names=data['feature_names']
    )

    exp, data = explainer.explain_instance(
        data_row=X_test[0],
        predict_fn=clf.predict_proba
    )
    print(data)
