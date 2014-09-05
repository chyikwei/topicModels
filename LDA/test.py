import numpy as np
import scipy.sparse as sp

from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import raises
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_less

from lda import _split_sparse, _min_batch_split, _n_fold_split
from lda import onlineLDA

random_state = np.random.mtrand.RandomState(0)


def test_split_sparse():
    n_row = 18
    n_col = 3
    batch_sizes = [3, 4, 5, 6]

    X = sp.rand(n_row, n_col, density=0.2,  format='csr')
    for size, batch_X in zip(batch_sizes, _split_sparse(X, batch_sizes)):
        assert_true(batch_X.shape[0] == size)
        assert_true(batch_X.shape[1] == n_col)


def test_n_fold_split():
    n_row = random_state.randint(20, 40)
    n_col = random_state.randint(3, 6)
    n_fold = random_state.randint(3, 8)
    X = sp.rand(n_row, n_col, density=0.2,  format='csr')

    max_fold_size = int(np.ceil(float(n_row) / n_fold))
    for part in _n_fold_split(X, n_fold):
        assert_true(part.shape[0] <= max_fold_size)
        assert_true(part.shape[1] == n_col)


def test_min_batch_split():
    n_row = 30
    n_col = 3
    batch_size = 7
    n_batch = 5
    X = sp.rand(n_row, n_col, density=0.2,  format='csr')

    idx = 1
    for part in _min_batch_split(X, batch_size):
        if idx < n_batch:
            assert_true(part.shape[0] == batch_size)
            assert_true(part.shape[1] == n_col)
        else:
            assert_true(part.shape[0] == n_row % batch_size)
            assert_true(part.shape[1] == n_col)
        idx += 1


def test_lda():
    """test LDA with sparse array"""
    pass


def test_lda_dense():
    """test LDA with dense array"""
    pass


def test_lda_top_words():
    """
    Test top words in LDA topics
    Test top words by create 3 topics, each have 3 words
    Top 3 words in each topic should be consistent with the index
    """

    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    block = n_topics * np.ones((3,3))
    X = sp.block_diag([block] * n_topics).tocsr()

    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0, tau=5., random_state=random_state)
    lda.fit_transform(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert_true(tuple(sorted(top_idx)) in correct_idx_grps)


def test_lda_fit_transform():
    """
    Test LDA fit_transform & transform
    fit_transform and transform result should be similar
    """

    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    block = n_topics * np.ones((3,3))
    X = sp.block_diag([block] * n_topics).tocsr()

    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0, tau=5., random_state=random_state)
    X_fit = lda.fit_transform(X)
    X_trans = lda.transform(X)
    assert_array_almost_equal(X_fit, X_trans, 1)


def test_lda_normalize_docs():
    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    block = n_topics * np.ones((3,3))
    X = sp.block_diag([block] * n_topics).tocsr()

    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0, tau=5., random_state=random_state)
    X_fit = lda.fit_transform(X)
    assert_array_almost_equal(X_fit.sum(axis=1), np.ones(X.shape[0]))


def test_lda_partial_fit():
    pass

def test_lda_transform():
    pass

def test_dim_mismatch():
    pass

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
