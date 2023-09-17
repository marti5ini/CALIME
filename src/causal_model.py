import os

from src.gencda.generative_model import *
from src.gencda.ncda import *


def get_causal_model(train_gen,
                     feature_names,
                     path,
                     target='m',
                     zmax=3,
                     num_bins=4,
                     strategy='quantile',
                     support=5,
                     alpha=0.001,
                     sorting=np.mean,
                     train_size=0.7,
                     standardization=True):
    """
    Args:
        train_gen: pandas DataFrame, training data
        feature_names: list of names (strings) corresponding to the columns
                in the training data
        path: string, the location of the directory to save results
        target: string, default = 'm'
            a character string indicating the type of association mined
            target type:
            s: frequent item sets
            c: closed (frequent) item sets
            m: maximal (frequent) item sets
        zmax: integer, default = 3
            Maximum number of items per item set
        num_bins: integer, default = 4
            the number of bins to produce
        strategy: string, default = 'quantile'
            strategy used to define the widths of the bins
        support: integer, default = 5
            minimum support threshold (percentage of transactions)
        alpha: float, default = 0.001
            independence test threshold
        sorting: function, default = np.mean
            sorting type: [np.mean, np.max, np.min]
            To the aim of returning only the most representative ones,
            we consider only the causal relationships with the highest
            average (or maximum or minimum) level of p-values of the various dependencies
        train_size: float, default = 0.7
            It should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split
        standardization: bool, default = True
            standardize features before passing them to Gaussian Process Regressor

    Returns:
        tuple: (pickled model, networkx.DiGraph)
    """

    ncda = NCDApriori(train_gen)
    itemsets = ncda.fitApriori(target=target, zmax=zmax, nbins=num_bins, strategy=strategy,
                               support=support)
    causal_relations = ncda.fitNCD(itemsets, alpha=alpha, sorting=sorting, train_size=train_size,
                                   standardization=standardization)

    graph = nx.DiGraph()
    graph.add_nodes_from(feature_names)
    graph.add_edges_from(causal_relations)

    file_path = os.path.join(path, 'ground_truth.gpickle')

    nx.write_gpickle(graph, file_path)

    generative_model = CausalDataGeneration(graph, train_gen)
    generative_model.generate_data()

    filename = path + '/generative_model.pkl'
    pickle.dump(generative_model, open(filename, 'wb'))

    return generative_model, graph
