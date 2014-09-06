import numpy as np
import scipy.sparse as sp

from sklearn.utils.testing import assert_true
from sklearn.utils.testing import raises
from sklearn.utils.testing import assert_array_almost_equal

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


def test_lda_batch():
    """
    Test LDA batch training(`fit` method)

    Test top words by create 3 topics, each have 3 words
    Top 3 words in each topic should be consistent with the index
    """

    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    block = n_topics * np.ones((3, 3))
    X = sp.block_diag([block] * n_topics).tocsr()

    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0,
                    random_state=random_state)
    lda.fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert_true(tuple(sorted(top_idx)) in correct_idx_grps)


def test_lda_online():
    """
    Test LDA online training(`partial_fit` method)
    (same as test_lda_batch)
    """

    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    block = n_topics * np.ones((3, 3))
    X = sp.block_diag([block] * n_topics).tocsr()

    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0,
                    tau=30., random_state=random_state)

    for i in xrange(3):
        lda.partial_fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert_true(tuple(sorted(top_idx)) in correct_idx_grps)


def test_lda_dense_input():
    """
    Test LDA with dense input.
    Similar to test_lda()
    """
    X = np.random.randint(5, size=(20, 10))
    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0,
                    random_state=random_state)
    X_trans = lda.fit_transform(X)
    assert_true((X_trans > 0.0).any())


def test_lda_fit_transform():
    """
    Test LDA fit_transform & transform
    fit_transform and transform result should be the same
    """

    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    block = n_topics * np.ones((3, 3))
    X = sp.block_diag([block] * n_topics).tocsr()

    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0,
                    random_state=random_state)
    X_fit = lda.fit_transform(X)
    X_trans = lda.transform(X)
    assert_array_almost_equal(X_fit, X_trans, 4)


def test_lda_normalize_docs():
    """
    test sum of topic distribution equals to 1 for each doc
    """

    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    block = n_topics * np.ones((3, 3))
    X = sp.block_diag([block] * n_topics).tocsr()

    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0,
                    random_state=random_state)
    X_fit = lda.fit_transform(X)
    assert_array_almost_equal(X_fit.sum(axis=1), np.ones(X.shape[0]))


@raises(ValueError)
def test_lda_partial_fit_dim_mismatch():
    """
    test n_vocab mismatch in partial_fit
    """

    n_topics = random_state.randint(3, 6)
    alpha0 = eta0 = 1. / n_topics

    n_col = random_state.randint(6, 10)
    X_1 = np.random.randint(4, size=(10, n_col))
    X_2 = np.random.randint(4, size=(10, n_col + 1))
    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0,
                    tau=5., random_state=random_state)
    for X in [X_1, X_2]:
        lda.partial_fit(X)


@raises(AttributeError)
def test_lda_transform_before_fit():
    """
    test `transform` before `fit`
    """
    X = np.random.randint(4, size=(20, 10))
    lda = onlineLDA()
    lda.transform(X)


@raises(ValueError)
def test_lda_transform_mismatch():
    """
    test n_vocab mismatch in fit and transform
    """
    X = np.random.randint(4, size=(20, 10))
    X_2 = np.random.randint(4, size=(10, 8))

    n_topics = random_state.randint(3, 6)
    alpha0 = eta0 = 1. / n_topics
    lda = onlineLDA(n_topics=n_topics, alpha=alpha0,
                    eta=eta0, random_state=random_state)
    lda.partial_fit(X)
    lda.transform(X_2)


def test_lda_batch_multi_jobs():
    """
    Test LDA batch training with multi jobs
    """

    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    block = n_topics * np.ones((3, 3))
    X = sp.block_diag([block] * n_topics).tocsr()

    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0,
                    n_jobs=2, random_state=random_state)
    lda.fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert_true(tuple(sorted(top_idx)) in correct_idx_grps)


def test_lda_online_multi_jobs():
    """
    Test LDA online training with multi jobs
    """

    n_topics = 3
    alpha0 = eta0 = 1. / n_topics
    block = n_topics * np.ones((3, 3))
    X = sp.block_diag([block] * n_topics).tocsr()

    lda = onlineLDA(n_topics=n_topics, alpha=alpha0, eta=eta0,
                    n_jobs=2, tau=30., random_state=random_state)
    for i in xrange(3):
        lda.partial_fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert_true(tuple(sorted(top_idx)) in correct_idx_grps)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
