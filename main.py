import os
import pickle
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from src.calime_explainer import CALimeExplainer
from src.causal_model import get_causal_model
from src.lime_explainer import LimeExplainer

warnings.simplefilter('ignore')


def save_explanation(explanation, algorithm):
    exp = explanation.as_list(label=1)
    vals = [x[1] for x in exp]
    names = [x[0] for x in exp]
    vals.reverse()
    names.reverse()
    colors = ['green' if x > 0 else 'red' for x in vals]
    pos = np.arange(len(exp)) + .5
    plt.barh(pos, vals, align='center', color=colors)
    plt.yticks(pos, names)
    plt.savefig(path + f'/{algorithm}_explanation.png')
    return


if __name__ == '__main__':
    dataset_name = 'wine_red'

    path = os.path.join(os.path.dirname(__file__), 'results', dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', dataset_name + '.csv')

    print(f"Loading the {dataset_name} dataset...")
    df = pd.read_csv(dataset_path, index_col=False)
    num_samples = 50
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
        nx.draw_networkx(graph)
        plt.show()
        with open(generation_path, 'rb') as file:
            generative_model = pickle.load(file)
    else:
        generative_model, graph = get_causal_model(train_gen, feature_names, path)
    print("Causal model and graph generated.")

    print("Fitting the Black Box...")
    # Create a RandomForestClassifier instance as the base estimator.
    estimator = RandomForestClassifier()
    hyper_param = dict(n_estimators=[100, 300, 500, 1000],
                       max_depth=[5, 8, 15],
                       min_samples_split=[2, 5, 10, 15, 100],
                       min_samples_leaf=[1, 2, 5, 10]
                       )
    # Perform hyperparameter tuning using RandomizedSearchCV with the specified hyperparameters.
    search = RandomizedSearchCV(estimator, hyper_param, verbose=1)

    search.fit(train_bb, y_bb)

    # Get the best estimator (model) from the hyperparameter tuning.
    best_estimator = search.best_estimator_

    # Use the trained 'best_estimator' model to make predictions on the test dataset.
    y_pred = best_estimator.predict(test)

    # Reset the index of the 'test' DataFrame.
    test = test.reset_index()

    # Select a random row from the 'test' DataFrame.
    random_row = test.sample(n=1)

    # Print the index of the selected row from the 'test' DataFrame.
    print('Selected Row Index:', random_row.index[0])

    # Extract the feature values from the selected row ('data_row').
    data_row = random_row.iloc[:, 1:].values.flatten()

    # LIME explanation
    lime_explainer = LimeExplainer(train_bb.values, feature_names=feature_names,
                                   class_names=class_values, discretize_continuous=False)

    lime_exp, lime_data, lime_neighbor_gen_time = lime_explainer.explain_instance(data_row,
                                                                                  best_estimator.predict_proba)

    # CALIME explanation
    calime_explainer = CALimeExplainer(graph, generative_model, train_bb.values, feature_names=feature_names,
                                       class_names=class_values, discretize_continuous=False)

    calime_exp, calime_data, calime_neighbor_gen_time = calime_explainer.explain_instance(data_row,
                                                                                          best_estimator.predict_proba)

    save_explanation(lime_exp, 'lime')
    save_explanation(calime_exp, 'calime')
