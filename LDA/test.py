import numpy as np
import scipy.sparse as sp

from .lda import _split_sparse, LDA


def test_split_sparse():
    # TODO: complete test
    indptr = np.array([0, 2, 3, 6, 8, 10])
    indices = np.array([0, 2, 2, 0, 1, 2, 0, 1, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X = sp.csr_matrix((data, indices, indptr), shape=(5, 3))

    for a, b, c, d in _split_sparse(X, 2):
        print 'len', len(a) - 1
        print a, b, c, d


def lda_test():
    # TODO: complete test
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

    n_topics = 3
    n_top_words = 4
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[^\d\W]\w+\b",
                                 max_df=0.9, min_df=1, vocabulary=test_vocab)

    doc_word_count = vectorizer.fit_transform(test_docs)
    lda = LDA(n_topics=n_topics, kappa=0.7,
              tau0=1024., random_state=0, n_jobs=2)
    lda.fit_transform(doc_word_count)
    print lda.components_
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
