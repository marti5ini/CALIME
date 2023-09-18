import multiprocessing
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_distances


def ridge_solve(tup):
    data_synthetic_onehot, model_pred, weights = tup
    solver = Ridge(alpha=1, fit_intercept=True)
    solver.fit(data_synthetic_onehot,
               model_pred,
               sample_weight=weights.ravel())
    # Get explanations
    importance = solver.coef_
    return importance


class NumpyTextExplainer:

    def __init__(self):
        pass

    def kernel_fn(self, distances, kernel_width):
        return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    def explain_instance(self, text_instance, predict_fn, label=0, num_estimators=1, num_samples=5000,
                         num_features=10, workers=1, kernel_width=None):
        if type(text_instance) is str:
            text_instance = [text_instance]

        # Use Count Vectorizer to vectorize text as BoW
        count_vectorizer = CountVectorizer()
        bow_vec = count_vectorizer.fit_transform(text_instance).toarray()
        bow_vec = bow_vec / bow_vec
        word_dim = bow_vec.shape[1]

        # Build word analyzer
        analyzer = count_vectorizer.build_analyzer()

        # Map the token to its indices and each index to token
        analyzed_text = analyzer(text_instance[0])
        dict_token_idxes = defaultdict(list)
        for (idx, key) in enumerate(analyzed_text):
            dict_token_idxes[key].append(idx)

        # For each sample, choose how many features we will keep
        num_words_keep = np.random.randint(1, word_dim, size=num_samples * num_estimators)

        # First row of every batch is the original sample, hence we keep all words
        for idx in range(num_estimators):
            num_words_keep[idx * num_samples] = word_dim

        # Sample binary data
        bin_samples = np.array(
            list(map(lambda x: np.random.binomial(1, x / word_dim, size=word_dim),
                     num_words_keep))).astype(bool)

        # Now create synthetic raw text
        features = np.array(count_vectorizer.get_feature_names())

        # boolean index the words
        list_words = [features[row] for row in bin_samples]

        # Get the synthetic raw text
        ### TODO: this is probably the most expensive part - can we optimize this?
        def unravel_text(list_word):
            a = []
            for word in list_word:
                a.extend(dict_token_idxes[word])
            a = sorted(a)
            return ' '.join(np.array(analyzed_text)[a])

        list_raw_synthetic = list(map(unravel_text, list_words))

        # Get model predictions
        model_pred = predict_fn(list_raw_synthetic)

        # Get distances between original sample and synthetic ones
        distances = pairwise_distances(bin_samples[0].reshape((1, -1)), bin_samples,
                                       metric='cosine').ravel()

        # Get weights
        if kernel_width is None:
            kernel_width = np.sqrt(word_dim) * 0.75
        weights = self.kernel_fn(distances, kernel_width=kernel_width).ravel()

        # Solve
        importances = []
        iterator = ((bin_samples[batch_idx * num_samples:(batch_idx + 1) * num_samples],
                     model_pred[batch_idx * num_samples:(batch_idx + 1) * num_samples, label],
                     weights[batch_idx * num_samples:(batch_idx + 1) * num_samples]) for batch_idx
                    in range(num_estimators))
        if workers == 1:
            for tup in iterator:
                # Solve
                importance = ridge_solve(tup)
                importances.append(importance)
        else:
            pool = multiprocessing.Pool(workers)
            importances = pool.map(func=ridge_solve, iterable=iterator)
            pool.close()
            pool.join()
            pool.terminate()
            del pool

        # Get explanation
        importances = np.mean(np.stack(importances), axis=0)
        explanations = sorted(list(zip(features, importances)),
                              key=lambda x: x[1], reverse=True)[:num_features]

        return explanations

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer

    categories = [
        'rec.sport.baseball',
        'soc.religion.christian',
        'sci.med'
    ]

    raw_train = datasets.fetch_20newsgroups(subset='train', categories=categories)
    print(list(raw_train.keys()))
    print(raw_train.target_names)
    print(raw_train.target[:10])
    raw_test = datasets.fetch_20newsgroups(subset='test', categories=categories)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(raw_train.data)
    y_train = raw_train.target

    print(X_train.shape)

    X_test = vectorizer.transform(raw_test.data)
    y_test = raw_test.target

    print(X_test.shape)

    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    explainer = NumpyTextExplainer()

    def predict_fn(instance):
        vec = vectorizer.transform(instance)
        return clf.predict_proba(vec)

    pred = predict_fn(raw_test.data[:1])

    print(pred)
    label = np.argmax(pred.ravel()).ravel()[0]
    exp = explainer.explain_instance(
        text_instance=raw_test.data[0],
        predict_fn=predict_fn,
        num_estimators=10,
        label=label
    )
    print(exp)
