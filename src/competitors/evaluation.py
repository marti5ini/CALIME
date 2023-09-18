from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer

from src.metrics.plausibility import *
from main import *
from src.competitors.dlime.dlime import *
from src.competitors.faster_lime.explainers.numpy_tabular_explainer import *
from tqdm import tqdm
from pathlib import Path

algorithm_names = ["LIME", "CALIME", "FASTER", "DLIME"]

def load_data(dataset_name, path):

    dataset_path = os.path.join('/Users/martina/Documents/PhD/Calime/CALIME', 'data', dataset_name + '.csv')

    print(f"Loading the {dataset_name} dataset...")
    df = pd.read_csv(dataset_path, index_col=False)
    num_samples = 100
    print("Generating the causal model...")
    # Split the DataFrame into features (X) and target labels (y).
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Create lists to store feature names and unique class values.
    feature_names, class_values = list(X.columns), list(y.unique())

    # Split the data into training and testing sets.
    train, test = X.head(len(X) - num_samples), X.tail(num_samples)
    labels_train, labels_test = y.head(len(y) - num_samples), y.tail(num_samples)

    # Further split the training set into two subsets: 'train_bb' and 'train_gen'.
    # This is typically done for causal modeling purposes.
    train_bb, train_gen, y_bb, y_gen = train_test_split(train, labels_train, test_size=0.3, random_state=0)

    # Generate a causal model ('generative_model') and a graph representation ('graph')
    # using the 'train_gen' dataset, feature names, and a specified 'path'.
    graph_path = os.path.join(path, 'ground_truth.gpickle')
    generation_path = os.path.join(path, 'generative_model.pkl')
    if os.path.exists(graph_path) and os.path.exists(generation_path):
        graph = nx.read_gpickle(graph_path)
        with open(generation_path, 'rb') as file:
            generative_model = pickle.load(file)
    else:
        generative_model, graph = get_causal_model(train_gen, feature_names, path)
    print("Causal model and graph generated.")

    print("Fitting the Black Box...")
    # Create a RandomForestClassifier instance as the base estimator.
    estimator = RandomForestClassifier()
    estimator.fit(train_bb, y_bb)

    # Reset the index of the 'test' DataFrame.
    test = test.reset_index()
    # Select a random row from the 'test' DataFrame.
    random_row = test.sample(n=1)
    # Extract the feature values from the selected row ('data_row').
    data_row = random_row.iloc[:, 1:].values.flatten()

    return data_row, estimator, X, y, train_bb, train, generative_model, graph, feature_names, class_values


if __name__ == '__main__':
    datasets = ['banknote', 'wine_red', 'wdbc', 'statslog']
    results_df = pd.DataFrame(columns=["Dataset", "Algorithm", "AMD", "AOS", "ASM", "ADM", "R^2"])

    for dataset_name in datasets:
        print(f'dataset: {dataset_name}')
        path = os.path.join(os.path.dirname(__file__), 'results', dataset_name)

        if not os.path.exists(path):
            os.makedirs(path)

        data_row, estimator, X, y, train_bb, train, generative_model, graph, feature_names, class_values \
            = load_data(dataset_name, path)

        data = None
        for algorithm_name in algorithm_names:
            with tqdm(total=100, desc=f"{algorithm_name}") as progress_bar:
                if algorithm_name == "DLIME":
                    clustering = AgglomerativeClustering().fit(X)
                    clabel = clustering.labels_
                    clustered_data = np.column_stack([X, clabel])

                    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train)
                    distances, indices = nbrs.kneighbors(data_row.reshape(1, -1))

                    p_label = clabel[indices[0]]
                    N = clustered_data[clustered_data[:, len(feature_names)] == p_label]
                    subset = np.delete(N, len(feature_names), axis=1)
                    dlime_explainer = DLimeExplainer(train_bb.values, feature_names=feature_names,
                                                     class_names=class_values, discretize_continuous=True,
                                                     discretizer='quartile')
                    exp, data = dlime_explainer.explain_instance_hclust(data_row, estimator.predict_proba,
                                                                        clustered_data=subset,
                                                                        regressor='linear')

                elif algorithm_name == "LIME":
                    lime_explainer = LimeExplainer(train_bb.values, feature_names=feature_names,
                                                   class_names=class_values, discretize_continuous=False)
                    exp, data, lime_neighbor_gen_time = lime_explainer.explain_instance(data_row, estimator.predict_proba)
                elif algorithm_name == "CALIME":
                    calime_explainer = CALimeExplainer(graph, generative_model, train_bb.values,
                                                       feature_names=feature_names,
                                                       class_names=class_values, discretize_continuous=False)
                    exp, data, calime_neighbor_gen_time = calime_explainer.explain_instance(data_row,
                                                                                            estimator.predict_proba)
                elif algorithm_name == "FASTER":
                    explainer = NumpyTabularExplainer(training_data=train_bb.values, feature_names=feature_names)

                    exp, data = explainer.explain_instance(data_row=data_row, predict_fn=estimator.predict_proba)
                else:
                    pass

                if algorithm_name == 'DLIME':
                    discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
                    discrete_data = discretizer.fit_transform(train_bb)

                    train_bb = pd.DataFrame(discrete_data, columns=list(train_bb.columns))

                data_df = pd.DataFrame(data, columns=list(train_bb.columns))

                result = Plausibility(train_bb, data_df, algorithm_name)
                if algorithm_name == 'DLIME':
                    aos = result.avg_outlier_score(normalize=False)
                    score = exp.score[1]
                else:
                    aos = result.avg_outlier_score()
                    score = exp.score
                result_row = {
                    "Dataset": dataset_name,
                    "Algorithm": algorithm_name,
                    "R^2": score,
                    "AMD": result.avg_minimum_distance(),
                    "AOS": aos,
                    "ADM": result.avg_detection_metric(),
                    "ASM": result.avg_statistics_metric()
                }
                results_df = results_df.append(result_row, ignore_index=True)
                progress_bar.update(100)

    results_df.to_csv("results.csv", index=False)
