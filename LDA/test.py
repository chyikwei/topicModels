import numpy as np
import scipy.sparse as sp

from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import raises
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_less

from lda import _split_sparse, LDA

random_state = np.random.mtrand.RandomState(0)

def test_split_sparse():
    n_row = random_state.randint(20, 40)
    n_col = random_state.randint(3, 6)
    n_fold = random_state.randint(3, 8)
    max_fold_size = int(np.ceil(float(n_row) / n_fold))

    X = random_state.randint(0, 2, (n_row, n_col))
    X = sp.csr_matrix(X)
    for part in _split_sparse(X, n_fold):
        assert_true(part.shape[0] <= max_fold_size)
        assert_true(part.shape[1] == n_col)


def lda_example():
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


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
