from src.metrics.plausibility import *
from main import *
from src.dlime.dlime import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

if __name__ == '__main__':

    dataset_name = 'wine_red'

    path = os.path.join(os.path.dirname(__file__), 'results', dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', dataset_name + '.csv')

    print(f"Loading the {dataset_name} dataset...")
    df = pd.read_csv(dataset_path, index_col=False)
    num_samples = 100
    print("Dataset loaded successfully.")
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

    clustering = AgglomerativeClustering().fit(X)
    clabel = clustering.labels_
    clustered_data = np.column_stack([X, clabel])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train)
    distances, indices = nbrs.kneighbors(data_row.reshape(1, -1))

    p_label = clabel[indices[0]]
    N = clustered_data[clustered_data[:, 11] == p_label]
    subset = np.delete(N, 11, axis=1)

    # DLIME explanation
    dlime_explainer = DLimeExplainer(train_bb.values, feature_names=feature_names,
                                     class_names=class_values, discretize_continuous=True, discretizer='quartile')

    dlime_exp, dlime_data = dlime_explainer.explain_instance_hclust(data_row, estimator.predict_proba,
                                                                    clustered_data=subset, regressor='linear')

    # LIME explanation
    lime_explainer = LimeExplainer(train_bb.values, feature_names=feature_names,
                                   class_names=class_values, discretize_continuous=False)

    lime_exp, lime_data, lime_neighbor_gen_time = lime_explainer.explain_instance(data_row,
                                                                                  estimator.predict_proba)

    # CALIME explanation
    calime_explainer = CALimeExplainer(graph, generative_model, train_bb.values, feature_names=feature_names,
                                       class_names=class_values, discretize_continuous=False)

    calime_exp, calime_data, calime_neighbor_gen_time = calime_explainer.explain_instance(data_row,
                                                                                          estimator.predict_proba)

    lime_data_df = pd.DataFrame(lime_data)
    calime_data_df = pd.DataFrame(calime_data)
    dlime_data_df = pd.DataFrame(dlime_data)

    train_bb_df = pd.DataFrame(train_bb)

    result = Plausibility(train_bb_df, calime_data_df, 'calime')
    print(result.avg_minimum_distance())

    result2 = Plausibility(train_bb_df, lime_data_df, 'lime')
    print(result2.avg_minimum_distance())

    result3 = Plausibility(train_bb_df, dlime_data_df, 'dlime')
    print(result3.avg_minimum_distance())
