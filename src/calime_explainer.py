import time
import networkx as nx
import pickle
import scipy
from scipy import stats
from random import choice
from lime.lime_tabular import *


class CALimeTabularExplainer(LimeTabularExplainer):

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         graph,
                         generative_model,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=1000,
                         distance_metric='euclidean',
                         model_regressor=None):

        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).
        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            graph: networkx.DiGraph. Ground Truth of relations between the features.
            generative_model: pickle model
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()

        start_time = time.time()
        data = self.__neighbors_generation(generative_model, graph, data_row, num_samples)
        time_gen = (time.time() - start_time)

        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        distances = sklearn.metrics.pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        yss = predict_fn(data)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                    discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                scaled_data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp, data, time_gen

    def __neighbors_generation(self, generative_model, graph, data_row, num_samples):
        """
        :param generative_model: pickled model
        :param graph: networkx.DiGraph
        :param data_row: 1d numpy array, corresponding to a row
        :param num_samples: size of the neighborhood to learn the linear model
        :return: 2d numpy array, num_samples * num_cols
        """
        n_configurations = 10
        partial_n_samples = num_samples // 10

        node_names = list(graph.nodes)
        positions = {node_names[position]: position for position in range(len(node_names))}
        num_cols = data_row.shape[1]
        neighbors = np.zeros((num_samples, num_cols))

        idx_temp = 0
        for idx in range(1, n_configurations + 1):
            config = self.__get_configuration(graph)
            num_rows = idx * partial_n_samples
            for idx_row in range(idx_temp, num_rows):
                for node in nx.topological_sort(graph):
                    changed_value = self.__get_value(config, node, idx_row, data_row,
                                                     neighbors, generative_model, positions)

                    neighbors[idx_row][positions[node]] = changed_value[0]
            idx_temp = num_rows

        return neighbors

    @staticmethod
    def __get_value(config, node, idx_row, data_row, neighbors, generative_model, positions):

        if config[node][1]:  # change
            if config[node][0]:  # independent column
                dist_name = generative_model.independent_dist[node][0]
                param = generative_model.independent_dist[node][1]
                dist = getattr(scipy.stats, dist_name)
                changed_value = dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=1)
            else:  # dependent column
                parent_names = list(config[node][2])
                parents_positions = [position for parent, position in positions.items() if parent in parent_names]
                parents_value = neighbors[idx_row][[parents_positions]].reshape(1, -1)
                saved_model = pickle.loads(generative_model.dependent_dist[node])
                changed_value = saved_model.predict(parents_value)
        else:
            changed_value = np.array([data_row[node]])

        return changed_value

    @staticmethod
    def __get_configuration(dag):
        configuration = dict()
        # bool value to guarantee that at least one independent variable is changed
        check = False
        while check is False:
            for node in nx.topological_sort(dag):
                predecessors = set(dag.predecessors(node))
                if len(predecessors) == 0:
                    rnd_number = choice([0, 1])
                    if rnd_number == 1:
                        # independent variable that it is changed
                        configuration[node] = (True, True, [])
                        check = True
                    else:
                        # independent variable that it is not changed
                        configuration[node] = (True, False, [])
                else:
                    for elem in predecessors:
                        if configuration[elem][1]:
                            # dependent variable that it is changed
                            configuration[node] = (False, True, predecessors)

        return configuration
