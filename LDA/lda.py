"""Latent Dirichlet Allocation with variational inference"""
# Note: This implementation is modified from Matthew D. Hoffman's onlineldavb
# Link: http://www.cs.princeton.edu/~mdhoffma/code/onlineldavb.tar

# Author: Chyi-Kwei Yau


import numpy as np
import math
from scipy.special import gammaln, psi
import scipy.sparse as sp
from sklearn.utils import check_random_state, atleast2d_or_csr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed


def _dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


def _split_sparse(X, n_fold):
    """
    split sparse matrix by row
    return a new csr_matrix
    """
    if n_fold <= 1:
        yield X

    n_rows, n_cols = X.shape
    fold_size = int(math.ceil(float(n_rows) / n_fold))
    X_indptr = X.indptr
    X_indices = X.indices
    X_data = X.data

    fold = 0
    while fold < n_fold:
        start_index = fold * fold_size
        end_index = min((fold + 1) * fold_size, n_rows) - 1

        indptr = X_indptr[start_index:end_index + 2] - X_indptr[start_index]
        indices = X_indices[X_indptr[start_index]:X_indptr[end_index + 1]]
        data = X_data[X_indptr[start_index]:X_indptr[end_index + 1]]
        shape = (len(indptr) - 1, n_cols)

        yield sp.csr_matrix((data, indices, indptr), shape=shape)

        fold += 1


def _update_gamma(X, expElogbeta, alpha, rng, max_iters, meanchangethresh, cal_delta):
    n_docs, n_vocabs = X.shape
    n_topics = expElogbeta.shape[0]

    # gamma is non-normailzed topic distribution
    gamma = rng.gamma(100., 1. / 100., (n_docs, n_topics))
    expElogtheta = np.exp(_dirichlet_expectation(gamma))
    # diff on component (only calculate it when keep_comp_change is True)
    delta_component = np.zeros(expElogbeta.shape) if cal_delta else None

    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    for d in range(n_docs):
        ids = X_indices[X_indptr[d]:X_indptr[d + 1]]
        cnts = X_data[X_indptr[d]:X_indptr[d + 1]]
        gammad = gamma[d, :]
        expElogthetad = expElogtheta[d, :]
        expElogbetad = expElogbeta[:, ids]
        # The optimal phi_{dwk} is proportional to
        # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
        phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

        # Iterate between gamma and phi until convergence
        for it in range(0, max_iters):
            lastgamma = gammad
            # We represent phi implicitly to save memory and time.
            # Substituting the value of the optimal phi back into
            # the update for gamma gives this update. Cf. Lee&Seung 2001.
            gammad = alpha + expElogthetad * \
                np.dot(cnts / phinorm, expElogbetad.T)
            expElogthetad = np.exp(_dirichlet_expectation(gammad))
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

            meanchange = np.mean(abs(gammad - lastgamma))
            if (meanchange < meanchangethresh):
                break
        gamma[d, :] = gammad
        # Contribution of document d to the expected sufficient
        # statistics for the M step.
        if cal_delta:
            delta_component[:, ids] += np.outer(expElogthetad, cnts / phinorm)

    return (gamma, delta_component)


class LDA(BaseEstimator, TransformerMixin):

    """
    online Latent Dirichlet Allocation implementation with variational inference
    Reference: 
    "Online Learning for Latent Dirichlet Allocation", Matthew D. Hoffman, David M. Blei, Francis Bach

    Parameters
    ----------
    n_topics: int
        number of topics

    X: sparse matrix, [n_docs, n_vocabs]
        The data matrix to be decomposed where X[i, j] is the count of vocab `j` in document `i`

    alpha: float
        Hyperparameter for prior on weight vectors theta

    eta: float
        Hyperparameter for prior on topics beta
    
    kappa: float
        Learning Rate

    e_step_tol: double, default: 1e-4
        Tolerance value used in e-step break conditions.

    prex_tol: double, default: 1e-4
        Tolerance value used in preplexity break conditions.

    n_jobs: int, default: 1
        The number of jobs to use for the E step. this will break input matrix into
        n_jobs parts and computing them in parallel.

    Attributes
    ----------
    `components_`: array, [n_topics, n_vocabs]
        vocab distribution for each topic. components_[i, j] is the probability of
        vocab `j` in topic `i`

    """

    def __init__(self, n_topics=10, alpha0=None, eta0=None, kappa=0.7,
                 e_step_tol=1e-3, pre_tol=1e-2, tau0=1024., n_jobs=1, random_state=None):
        self.n_topics = n_topics
        self.alpha0 = alpha0
        self.alpha = alpha0 or 1.0 / n_topics
        self.eta0 = eta0
        self.eta = eta0 or 1.0 / n_topics
        self.e_step_tol = e_step_tol
        self.prex_tol = pre_tol
        self.kappa = kappa
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.tau0 = tau0
        self.tau = tau0 + 1
        self.n_jobs = n_jobs
        self.updatecnt = 0
        self.total_docs = 3.3e6
        self.meanchangethresh = 0.001

    def _e_step(self, X, cal_delta=True):
        if self.n_jobs == 1:
            gamma, delta_component = _update_gamma(
                X, self.expElogbeta, self.alpha, self.rng, 100, self.meanchangethresh, cal_delta)

        else:
            # parell setting
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_update_gamma)(sub_X, self.expElogbeta, self.alpha, self.rng, 100, self.meanchangethresh, cal_delta)
                    for sub_X in _split_sparse(X, self.n_jobs))
            gammas, deltas = zip(*results)
            gamma = np.vstack(gammas)
            if cal_delta:
                delta_component = np.zeros(self.components_.shape)
                for delta in deltas:
                    delta_component += delta
            else:
                delta_component = None

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        if cal_delta:
            delta_component *= self.expElogbeta

        return (gamma, delta_component)

    def _m_step(self, delta_component, total_docs, n_docs):
        # TODO: check params
        rhot = pow(self.tau + self.updatecnt, -self.kappa)
        #print "rhot", rhot

        self.components_ = ((1 - rhot) * self.components_) + \
            (rhot * (self.eta + self.total_docs * delta_component / n_docs))

        self.Elogbeta = _dirichlet_expectation(self.components_)
        self.expElogbeta = np.exp(self.Elogbeta)
        self.updatecnt += 1

    def _to_csr(self, X):
        """
        check & convert X to csr format
        """
        X = atleast2d_or_csr(X)
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        return X

    def fit_transform(self, X, normalize=False, partial=True):
        """
        Learn a model for X and returns the transformed data
        Parameters
        ----------
        X: array or sparse matrix, shape = [n_docs, n_vocabs]
            Data matrix to be transformed by the model
            n_vacabs must be the same as n_vocabs in fitted model

        normalize: boolean
            normalize topic ditrubution or not. If 'True', sum of topic proportion
            will be 1 for each document

        Returns
        -------
        gamma: array, [n_dics, n_topics]
            Topic distribution for each doc
        """

        X = self._to_csr(X)
        n_docs, n_vocabs = X.shape

        if not hasattr(self, 'components_'):
            # initialize vocabulary & latent variables
            self.n_vocabs = n_vocabs
            self.components_ = self.rng.gamma(
                100., 1. / 100., (self.n_topics, n_vocabs))
            self.Elogbeta = _dirichlet_expectation(self.components_)
            self.expElogbeta = np.exp(self.Elogbeta)
        else:
            # make sure vacabulary size matched
            if self.components_.shape[1] != X.shape[1]:
                raise ValueError("dimension not match")

        # EM update
        gamma, delta_component = self._e_step(X)
        self._m_step(delta_component, self.total_docs, n_docs)

        # TODO: add partial transform

        if normalize:
            gamma /= gamma.sum(axis=1)[:, np.newaxis]

        return gamma

    def partial_fit(self, X, y=None):
        """
        learn model from X

        Parameters
        ----------
        X: sparse matrix, shape = [n_docs, n_vocabs]
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        X = self._to_csr(X)
        # TODO: modified tottal doc size
        #self.total_docs += X.shape[1]
        self.fit_transform(X, normalize=False, partial=True)
        return self

    def fit(self, X, y=None, max_iters=20, tol=1e-2, verbose=True):
        """
        Learn model from X

        Parameters
        ----------
        X: sparse matrix, shape = [n_docs, n_vocabs]
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        X = self._to_csr(X)
        _last_bound = None
        for i in xrange(max_iters):
            gamma = self.fit_transform(X, normalize=False)
            # check approx bound
            _bound = self.approx_bound(X, gamma)
            _perwordbound = (
                _bound * X.shape[0]) / (self.total_docs * sum(X.data))
            if verbose:
                print 'iteration', i, 'bound', _perwordbound
        
            if i > 0 and abs(_last_bound - _perwordbound) < tol:
                break

            _last_bound = _perwordbound

        return self

    def transform(self, X, normalize=True):
        """
        Transform the data X according to the fitted model (run inference)

        Parameters
        ----------
        X: sparse matrix, shape = [n_docs, n_vocabs]
            Data matrix to be transformed by the model
            n_vacabs must be the same as n_vocabs in fitted model

        Returns
        -------
        data: array, [n_docs, n_topics]
            Document distribution
        """
        X = self._to_csr(X)
        n_docs, n_vocabs = X.shape

        if not hasattr(self, 'components_'):
            raise AttributeError(
                "no 'components_' attr in model. Please fit model first.")
        # make sure vocabulary size is the same in fitted model and new doc
        # matrix
        if n_vocabs != self.n_vocabs:
            raise ValueError(
                "feature dimension(vocabulary size) not match.")

        gamma, _ = self._e_step(X, False)

        if normalize:
            gamma /= gamma.sum(axis=1)[:, np.newaxis]

        return gamma

    def approx_bound(self, X, gamma):
        """
        Parameters
        ----------
        X: sparse matrix, [n_docs, n_vocabs]
        
        gamma: array, shape = [n_docs, n_topics]
            document distribution (ca be either normalized & un-normalized)
        
        Returns
        -------
        score: float, score of gamma
        """
        X = self._to_csr(X)
        n_docs, n_topics = gamma.shape
        score = 0
        Elogtheta = _dirichlet_expectation(gamma)

        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

        # E[log p(docs | theta, beta)]
        for d in range(0, n_docs):
            ids = X_indices[X_indptr[d]:X_indptr[d + 1]]
            cnts = X_data[X_indptr[d]:X_indptr[d + 1]]
            phinorm = np.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self.Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
            score += np.sum(cnts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self.alpha - gamma) * Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self.alpha))
        score += sum(
            gammaln(self.alpha * self.n_topics) - gammaln(np.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self.total_docs / n_docs
        # E[log p(beta | eta) - log q (beta | lambda)]
        score += np.sum((self.eta - self.components_) * self.Elogbeta)
        score += np.sum(gammaln(self.components_) - gammaln(self.eta))
        score += np.sum(gammaln(self.eta * self.n_vocabs)
                        - gammaln(np.sum(self.components_, 1)))

        return score
