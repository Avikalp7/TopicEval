"""
This module provides some common utility functions to be shared across modules.
Does not have much as of now.
"""

from __future__ import print_function
from __future__ import division

import scipy.sparse
import numpy as np
from six.moves import xrange

import os
import logging

logger = logging.getLogger(__name__)


def verify_filename(filename):
    if not os.path.isfile(filename):
        raise ValueError("Given filepath doesn't exist.")
    return


def l2_normalize_lil_matrix(X):
    """
        l2 normalize the rows of X matrix

        :param X: scipy.sparse.lilmatrix, denoting the document x term matrix
        :return: scipy.sparse.lilmatrix, l2 normalized-by-row sparse X matrix
        """
    # TODO: Can be parrallelized
    assert scipy.sparse.isspmatrix_lil(X)
    for n in xrange(X.shape[0]):
        X.data[n] = X.data[n] / np.linalg.norm(X.data[n], ord=2)
        # assert (0.99 < np.linalg.norm(X.data[n], ord=2) ** 2 < 1.01), "Xn norm is not near 1"
    return X


def l2_normalize_2d_array(X):
    """
        l2 normalize the rows of X matrix

        :param X: np.array, denoting the document x term matrix
        :return: np.array, l2 normalized-by-row X matrix
        """
    # TODO: Can be parrallelized
    for n in xrange(X.shape[0]):
        X[n] = X[n] / np.linalg.norm(X[n], ord=2)
    return X



def l2_normalize_2d_array_copy(X):
    """
        l2 normalize the rows of X matrix

        :param X: np.array, denoting the document x term matrix
        :return: np.array, l2 normalized-by-row X matrix
        """
    # TODO: Can be parrallelized\
    Xnorm = np.zeros(X.shape)
    for n in xrange(X.shape[0]):
        Xnorm[n, :] = X[n] / np.linalg.norm(X[n], ord=2)
    return Xnorm


def l1_normalize_2darray(X):
    """
        l1 normalize the rows of X matrix

        :param X: np.ndarray, denoting the document x term matrix
        :return: np.ndarray, l1 normalized-by-row sparse X matrix
        """
    # TODO: Can be parrallelized
    for n in xrange(X.shape[0]):
        norm = np.linalg.norm(X[n, :], ord=1)
        if norm == 0:
            logger.warning("0 row encounter during l1 normalization of matrix, setting to uniform distribution")
            # noinspection PyTypeChecker
            X[n, :] = np.repeat(1/X.shape[1], X.shape[1])
        else:
            X[n, :] = X[n, :] / norm
    return X
