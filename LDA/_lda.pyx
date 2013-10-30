import numpy as np
cimport numpy as np
cimport cython
from scipy.special import gammaln, psi

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT


cdef np.ndarray __dirichlet_expectation(np.ndarray alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    cdef int n_col
    n_col = alpha.shape[1]
    if n_col == 0:
        return psi(alpha) - psi(np.sum(alpha))
    return psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
def lda_e_step(X, np.ndarray expElogbeta_, double alpha, unsigned int max_iters, int cal_delta=1):
    cdef unsigned int n_docs, n_vocabs, n_topics, d, it
    cdef DOUBLE meanchangethresh = 0.001
    cdef np.ndarray[DOUBLE, ndim=2] gamma, expElogtheta, expElogbetad, expElogbeta
    cdef np.ndarray[DOUBLE, ndim=2] delta_component = None
    cdef np.ndarray[DOUBLE, ndim=1] gammad, lastgamma, phinorm
    cdef np.ndarray[DOUBLE, ndim=1] X_data, cnts
    cdef np.ndarray[INT, ndim=1] X_indices, X_indptr, ids

    n_docs = X.shape[0]
    n_vocabs = X.shape[1]
    n_topics = expElogbeta_.shape[0]
    if n_vocabs != expElogbeta_.shape[1]:
        raise ValueError('components_ and X dimension not matched')
    expElogbeta = expElogbeta_
    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    gamma = np.random.gamma(100., 1. / 100., [n_docs, n_topics])
    expElogtheta = np.exp(__dirichlet_expectation(gamma))

    if cal_delta:
        delta_component = np.zeros([n_topics, n_vocabs])

    for d in range(n_docs):
        ids = X_indices[X_indptr[d]:X_indptr[d + 1]]
        cnts = X_data[X_indptr[d]:X_indptr[d + 1]]
        gammad = gamma[d, :]
        expElogthetad = expElogtheta[d, :]
        expElogbetad = expElogbeta[:, ids]
            
        phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
        for it in range(max_iters):
            lastgamma = gammad
            # We represent phi implicitly to save memory and time.
            # Substituting the value of the optimal phi back into
            # the update for gamma gives this update. Cf. Lee&Seung 2001.
            gammad = alpha + expElogthetad * np.dot(cnts / phinorm, expElogbetad.T)
            expElogthetad = np.exp(__dirichlet_expectation(gammad))
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
            # If gamma hasn't changed much, we're done.
            meanchange = np.mean(abs(gammad - lastgamma))
            if (meanchange < meanchangethresh):
                break
        gamma[d, :] = gammad
        # Contribution of document d to the expected sufficient
        # statistics for the M step.
        if cal_delta:
            delta_component[:, ids] += np.outer(expElogthetad, cnts / phinorm)

    if cal_delta:
        delta_component *= expElogbeta

    return (gamma, delta_component)


