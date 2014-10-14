#!/usr/bin/python

"""
==================================================
Topics extraction with Latent Dirichlet Allocation
==================================================

This is an example of showing howto extrat topics from
20 newsgroup dataset. In here, we provide both batch and
online update examples.

The code is modified from scikit learn's
"Topics extraction with Non-Negative Matrix Factorization"
example.

"""

# Authors: Chyi-Kwei Yau

import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals.six.moves import xrange
from lda import OnlineLDA


def lda_batch_example():
    """
    Example for LDA batch update
    """

    # In default, we set topic number to 10, and both hyperparameter
    # eta and alpha to 0.1 (`1 / n_topics`)
    n_topics = 10
    alpha = 1. / n_topics
    eta = 1. / n_topics

    # bach update is slow, so only use top 4000 records in 20 news groups
    n_samples = 4000
    n_features = 1000
    n_top_words = 15

    print('Example of LDA with bath update')
    print("Loading 20 news groups dataset...")
    dataset = fetch_20newsgroups(
        shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

    print("convert text into sparse matrix...")
    vectorizer = CountVectorizer(
        max_df=0.8, max_features=n_features, min_df=3, stop_words='english')

    doc_word_count = vectorizer.fit_transform(dataset.data[:n_samples])

    print("Fitting LDA models with batch udpate...")
    lda = OnlineLDA(
        n_topics=n_topics, alpha=alpha, eta=eta,
        n_jobs=1, random_state=0, verbose=1)

    feature_names = vectorizer.get_feature_names()
    lda.fit(doc_word_count, max_iters=2)
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def lda_online_example():
    """
    Example for LDA online update
    """

    def chunks(l, n):
        for i in xrange(0, len(l), n):
            yield l[i:i + n]

    # In default, we set topic number to 10, and both hyperparameter
    # eta and alpha to 0.1 (`1 / n_topics`)
    n_topics = 10
    alpha = 1. / n_topics
    eta = 1. / n_topics

    # chunk_size is how many records we want to use
    # in each online iteration
    chunk_size = 2000
    n_features = 1000
    n_top_words = 15

    print('Example of LDA with online update')
    print("Loading 20 news groups dataset...")
    dataset = fetch_20newsgroups(
        shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

    vectorizer = CountVectorizer(
        max_df=0.8, max_features=n_features, min_df=3, stop_words='english')

    lda = OnlineLDA(n_topics=n_topics, alpha=alpha, eta=eta, kappa=0.7,
                    tau=512., n_jobs=1, n_docs=1e4, random_state=0, verbose=0)

    for chunk_no, doc_list in enumerate(chunks(dataset.data, chunk_size)):
        if chunk_no == 0:
            doc_mtx = vectorizer.fit_transform(doc_list)
            feature_names = vectorizer.get_feature_names()
        else:
            doc_mtx = vectorizer.transform(doc_list)

        # fit model
        print("\nFitting LDA models with online udpate on chunk %d..." %
              chunk_no)
        lda.partial_fit(doc_mtx)
        print("Topics after training chunk %d:" % chunk_no)
        for topic_idx, topic in enumerate(lda.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))


def _lda_simple_example():
    """
    This is for debug
    """

    from sklearn.feature_extraction.text import CountVectorizer

    test_words = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj']
    test_vocab = {}
    for idx, word in enumerate(test_words):
        test_vocab[word] = idx

    # group 1: aa, bb, cc, dd
    # group 2: ee ff gg
    # group 3: hh ii jj
    test_docs = ['aa bb cc dd aa aa',
                 'ee ee ff ff gg gg',
                 'hh ii hh ii jj jj jj jj',
                 'aa bb cc dd aa aa dd aa bb cc',
                 'ee ee ff ff gg gg',
                 'hh ii hh ii jj jj jj jj',
                 'aa bb cc dd aa aa',
                 'ee ee ff ff gg gg',
                 'hh ii hh ii jj jj jj jj',
                 'aa bb cc dd aa aa dd aa bb cc',
                 'ee ee ff ff gg gg',
                 'hh ii hh ii jj jj jj jj']

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[^\d\W]\w+\b",
                                 max_df=0.9, min_df=1, vocabulary=test_vocab)

    doc_word_count = vectorizer.fit_transform(test_docs)

    # LDA setting
    n_topics = 3
    alpha = 1. / n_topics
    eta = 1. / n_topics
    n_top_words = 3

    lda = OnlineLDA(n_topics=n_topics, eta=eta, alpha=alpha,
                    random_state=0, n_jobs=1, verbose=0)
    lda.fit(doc_word_count)
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

if __name__ == '__main__':
    lda_batch_example()
