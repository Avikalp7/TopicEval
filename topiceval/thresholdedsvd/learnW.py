from __future__ import division
from __future__ import print_function

from topiceval import utils

import threading
import multiprocessing as mp
from math import ceil
import logging

import numpy as np
from six.moves import xrange

logger = logging.getLogger(__name__)


def residual_reduction(uj):
    if uj > 0:
        return uj * uj
    else:
        return 0


def NOMP(s, gamma, epsilon, tau, S):
    k = s.shape[0]
    u = s
    R = 1.
    A = set()
    Ac = set(np.arange(k))
    w = np.zeros(k)

    iteration_num = 0
    while len(A) < gamma:
        assert (len(A & Ac) == 0), "Error, A and Ac are not disjoint."
        iteration_num += 1
        Ac_list = list(Ac)
        reduction_values = [residual_reduction(u[j_candidate]) for j_candidate in Ac_list]
        max_idx = np.argmax(reduction_values)
        # noinspection PyTypeChecker
        j_star = Ac_list[max_idx]
        # noinspection PyTypeChecker
        reduction_value = reduction_values[max_idx]
        logger.log(0, "iteration_num=%d, max reduction value=%f, max reduction index=%d"
                   % (iteration_num, reduction_value, j_star))
        if reduction_value <= epsilon:
            logger.log(0, "iteration_num=%d, R=%f, reduction value=%f reached below eps=%f, returning w with "
                          "%d of %d elements" % (iteration_num, R, reduction_value, epsilon, len(A), gamma))
            return w
        assert (j_star not in A), "Overlap Error in A and Ac sets"
        A.add(j_star)
        Ac.remove(j_star)
        subiteration_num = 0
        while True:
            subiteration_num += 1
            for j in A:
                if u[j] == 0:
                    continue
                delta_j = w[j]
                w[j] = max(u[j] + w[j], 0)
                delta_j = w[j] - delta_j
                Sj = S[:, j]
                assert (u.shape == Sj.shape), "u and Sj don't match in shape"
                u = u - Sj * delta_j
            reduction = 0.
            for j in A:
                reduction += w[j] * (s[j] + u[j])
            R_old = R
            R = 1 - reduction
            if R_old != 0:
                tau_star = 1 - R / R_old
            else:
                tau_star = 0.
            logger.log(0, "iteration_num=%d, subiteration_num=%d, reduction=%f, tau_star=%f"
                          % (iteration_num, subiteration_num, reduction, tau_star))
            if R_old - R < tau * R_old or R <= 0:
                logger.log(0, "\nBreaking from cyclic coordinate descent for iteration number=%d after "
                              "subiteration number=%d\n" % (iteration_num, subiteration_num))
                break
    assert (iteration_num == gamma), "gamma and num_iterations don't match"
    logger.log(0, "\nCompleted gamma=%d iterations, returning with full elements" % iteration_num)
    return w


def NOMP_chunk(sfull, W, N, chunk_size, gamma, thread_num, S):
    start_idx = chunk_size * thread_num
    for i in xrange(int(start_idx), int(min(start_idx + chunk_size, N))):
        W[i, :] = NOMP(s=sfull[:, i], gamma=gamma, epsilon=0.0075, tau=0.005, S=S)
    logger.log(0, "Thread/Process %d finished processing chunk %d : %d"
                  % (thread_num, chunk_size * thread_num, chunk_size * (thread_num + 1)))
    return


def NOMP_chunk_mp(sfull, N, chunk_size, gamma, thread_num, S):
    start_idx = chunk_size * thread_num
    result = [NOMP(s=sfull[:, i], gamma=gamma, epsilon=0.0075, tau=0.005, S=S) for i in
              xrange(start_idx, min(start_idx + chunk_size, N))]
    # noinspection PyTypeChecker
    result.append([start_idx, min(start_idx + chunk_size, N)])
    return result


def learn(X, H, N, gamma, num_threads=20, num_processes=3, parallelization=False):
    W = None
    K = H.shape[0]
    Hnorm = utils.l2_normalize_2d_array_copy(H)
    S = Hnorm.dot(Hnorm.T)
    sfullt = X.dot(Hnorm.T)
    sfull = sfullt.T
    if parallelization:
        if N < 1500:
            use_threading, use_multiprocessing = True, False
            logger.log(0, "Using threading with %d threads" % num_threads)
        else:
            use_multiprocessing, use_threading = True, False
            logger.log(0, "Using multiprocessing with %d processes" % num_processes)
    else:
        use_threading, use_multiprocessing = False, False
        logger.log(0, "Using serial computation")
    if not use_threading and not use_multiprocessing:
        W = np.array([NOMP(s=sfull[:, i], gamma=gamma, epsilon=0.0005, tau=0.001, S=S) for i in xrange(N)])

    elif use_threading:
        W = np.zeros((N, K))
        chunk_size = ceil(N / num_threads)
        logger.log(0, "Chunk size = %f" % chunk_size)
        threads = []
        for thread_num in range(num_threads):
            thread = threading.Thread(target=NOMP_chunk, args=(sfull, W, N, chunk_size, gamma, thread_num, S))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    elif use_multiprocessing:
        W = np.zeros((N, K))
        chunk_size = int(ceil(N / num_processes))
        logger.log(0, "Chunk size = %f" % chunk_size)
        p = mp.Pool(processes=num_processes)

        def collect_results(result):
            start_idx, end_idx = result[-1]
            W[start_idx:end_idx, :] = np.array(result[:-1])
            return

        for i in xrange(num_processes):
            p.apply_async(NOMP_chunk_mp, args=(sfull, N, chunk_size, gamma, i, S), callback=collect_results)
        p.close()
        p.join()

    return W
