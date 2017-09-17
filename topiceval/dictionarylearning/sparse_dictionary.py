from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
import scipy.sparse
from six.moves import xrange


def learn(X, H, W, nu_param, lambda_param=0., H_init=None):
    N = X.shape[1]
    D = X.shape[0]
    K = W.shape[1]
    if W.shape[0] != N:
        raise ValueError("W's row dimension does not match with X's rows")
    if H.shape[0] != K or H.shape[1] != D:
        raise ValueError("H's dimensions don't match for given X and W")
    if lambda_param < 0:
        raise ValueError("Lambda parameter can't be negative")
    use_hinit = False
    if lambda_param == 0. and H_init is not None:
        warnings.warn("Warning: H_init won't be in effect as lambda parameter is set to 0")
    elif lambda_param > 0 and H_init is None:
        warnings.warn("Warning: Lamda parameter won't be in effect as H_init is None")
    elif lambda_param > 0 and H_init is not None:
        use_hinit = True
        assert (H_init.shape == H.shape), "H_init must have same dimenstions as H"

    # noinspection PyRedundantParentheses
    G = W.transpose().dot(W).toarray()                      # G has shape K x K, numpy 2D array
    XtW = X.dot(W)
    new_H = H.toarray()
    # hk_blocks = []
    for k in xrange(K):
        hk = np.zeros(D)
        # hk = scipy.sparse.dok_matrix((1, D), dtype=np.float64)
        g = G[:, k]                                           # g has shape (K, ), numpy array
        g[k] = 0.
        # vk = W[:, k]                                        # vk has shape N x 1, CSC matrix
        # noinspection PyRedundantParentheses
        # numerator = ((X.T).dot(vk) - (H.T).dot(g))
        numerator = (XtW[:, k] - scipy.sparse.csc_matrix(new_H.T.dot(g.reshape(-1, 1))))
        # numerator = (XtW[:, k] - scipy.sparse.csc_matrix(new_H.T.dot(g.reshape(-1, 1)))).toarray().flatten()
        # numerator = (X.dot(vk) - scipy.sparse.csc_matrix(H.transpose().dot(g.reshape(-1, 1)))).toarray().flatten()
        # numerator has shape D x 1, CSC matrix
        denominator = G[k, k]
        if use_hinit:
            init_term = (H_init[k, :].transpose())*lambda_param
            assert (init_term.shape == numerator.shape), "Init term and numerator shape do not agree"
            numerator = numerator/N + init_term
            denominator = denominator/N + lambda_param
        epsilon = 0.00001
        qk = numerator.multiply(1/(denominator+epsilon))    # qk has shape D x 1, CSC matrix
        # qk = np.multiply(numerator, (1 / (denominator + epsilon)))
        # assert (qk.shape == (D, 1))

        # TODO: Can be faster?
        # noinspection PyTypeChecker
        qk_argsorted = np.argsort(qk.data)[::-1]
        pos_vals = 0
        qk_data_sorted = qk.data[qk_argsorted]
        for dataval in qk_data_sorted:
            if dataval > 0:
                pos_vals += 1
            else:
                break
        num_indices = min(pos_vals, nu_param)
        Iset = qk_argsorted[:num_indices]
        # Iset = np.argsort(qk.data)[-num_indices:][::-1]

        # # Istarset = qk.indices[Iset]
        # Iset = np.argsort(qk)[::-1]
        # Istarset = set()
        # for I in Iset[0:nu_param]:
        #     if qk[I] > 0:
        #         Istarset.add(I)
        #     else:
        #         break
        norm = np.linalg.norm(qk_data_sorted[:num_indices], ord=2)
        # norm = np.linalg.norm(qk, ord=2)

        for I in Iset:
            hk[qk.indices[I]] = qk.data[I]/norm

        # for Istar in Istarset:
            # hk[Istar] = qk[Istar] / norm
        # hk_blocks.append(hk)
        # H = scipy.sparse.lil_matrix(H)
        new_H[k, :] = hk
        # H = scipy.sparse.csr_matrix(H)
    # H = scipy.sparse.vstack(blocks=hk_blocks, format='csr')
    # return scipy.sparse.csr_matrix(H)
    return scipy.sparse.csr_matrix(new_H)
