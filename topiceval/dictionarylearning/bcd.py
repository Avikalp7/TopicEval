"""
This module implements the spars dictionary learning - sparse coding based BCD algorithm for
topic modelling. The class BCD provides all utilities.
"""
# TODO: Make params file

from __future__ import division
from __future__ import print_function

from topiceval.basemodel import BaseModel
import topiceval.dictionarylearning.sparse_coding as sparse_coding
import topiceval.dictionarylearning.sparse_dictionary as sparse_dictionary
import topiceval.utils as utils
# from topiceval.usereval.task_evaluation import mici_selection

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from six.moves import xrange
from sklearn.feature_selection import SelectKBest

import logging
import sys
import os
import pickle

logger = logging.getLogger(__name__)


def softinit(K, D, nu_frac):
    sparse_rows = []
    indices = np.arange(D)
    for k in xrange(K):
        hk = scipy.sparse.random(m=1, n=D, density=nu_frac, format='csr', dtype=np.float64, random_state=None)
        hk.indices = np.random.choice(indices, size=len(hk.indices), replace=False)
        hk.data = hk.data / np.linalg.norm(hk.data, ord=2)
        assert (0.99 < np.linalg.norm(hk.data, ord=2) ** 2 < 1.01), "Hk norm is not near 1"
        sparse_rows.append(hk)
    H_sofinit = scipy.sparse.vstack(blocks=sparse_rows, format='csr')
    return H_sofinit


def normalize_X(X):
    """
    l2 normalize the rows of X matrix

    :param X: scipy.sparse.lilmatrix, denoting the document x term matrix
    :return: scipy.sparse.lilmatrix, l2 normalized-by-row sparse X matrix
    """
    # TODO: Can be parrallelized
    for n in xrange(X.shape[0]):
        X.data[n] = X.data[n] / np.linalg.norm(X.data[n], ord=2)
        # assert (0.99 < np.linalg.norm(X.data[n], ord=2) ** 2 < 1.01), "Xn norm is not near 1"
    return X


class BCD(BaseModel):
    def __init__(self, datasetname, learn_model=True, X_matrix=None, X_matrix_path=None, A_matrix_path=None,
                 id2word_dict=None,
                 id2word_dict_path=None, corpus=None, corpus_path=None, vocab_filepath=None, docword_filepath=None,
                 vocab_size=None, num_docs=None, num_topics=None, H_matrix_path=None, W_matrix_path=None,
                 evaluation_mode=False, bcd_iters=12, H_init=None, reconstruction_error_lim=1e1, gamma_frac=0.35,
                 nu_frac=0.20, sparse_mode=False):
        # TODO: If A/X matrix path is given, it may be is being revaluated in basemodel, check that
        self.modelname = "bcd"
        super(BCD, self).__init__(datasetname=datasetname, id2word_dict=id2word_dict,
                                  id2word_dict_path=id2word_dict_path, corpus=corpus,
                                  corpus_path=corpus_path, A_matrix_path=A_matrix_path, vocab_filepath=vocab_filepath,
                                  docword_filepath=docword_filepath, vocab_size=vocab_size, num_docs=num_docs,
                                  evaluation_mode=evaluation_mode,
                                  Amatrix_needed=(X_matrix is None and X_matrix_path is None and
                                                  A_matrix_path is None) or evaluation_mode)
        self.num_topics = num_topics

        # Loading the X matrix, requires checking which method for input has been used
        if X_matrix is not None:
            self.X_matrix = X_matrix
        elif X_matrix_path is not None:
            if X_matrix_path[-4:] == ".npy":
                self.X_matrix = np.load(X_matrix_path)
            elif X_matrix_path[-4:] == ".npz":
                self.X_matrix = scipy.sparse.load_npz(X_matrix_path)
            else:
                logger.warning("Unrecognized format, loading through pickle...")
                with open(X_matrix_path, "rb") as handle:
                    self.X_matrix = pickle.load(handle)
        elif A_matrix_path is not None:
            if A_matrix_path[-4:] == ".npy":
                self.X_matrix = np.load(A_matrix_path).T
            elif A_matrix_path[-4:] == ".npz":
                self.X_matrix = scipy.sparse.load_npz(A_matrix_path).transpose()
            else:
                logger.warning("Unrecognized format, loading through pickle...")
                with open(A_matrix_path, "rb") as handle:
                    self.X_matrix = pickle.load(handle).T
        else:
            if scipy.sparse.issparse(self.A_matrix):
                self.X_matrix = self.A_matrix.transpose()
            else:
                self.X_matrix = self.A_matrix.T

        # X matrix is stored in scipy.sparse.lil_matrix format
        if not scipy.sparse.isspmatrix_lil(self.X_matrix):
            self.X_matrix = scipy.sparse.lil_matrix(self.X_matrix)

        if self.num_docs is not None:
            assert (self.X_matrix.shape[0] == self.num_docs), "X shape does not agree with num_docs"
        self.num_docs = self.X_matrix.shape[0]

        if vocab_size is not None:
            assert (self.X_matrix.shape[1] == vocab_size), "X shape does not agree with num_docs"

        if not learn_model and (H_matrix_path is None or W_matrix_path is None):
            raise ValueError("Learn model option is off and H and W matrix paths havent been provided.")

        if learn_model:
            if self.num_topics is None:
                raise ValueError("Num_topics not provided (learn_model option is on)")

            if H_matrix_path is not None or W_matrix_path is not None:
                logger.warning("User given filepath for H/W discarded in favor of learn_model=True")

            logger.info("Starting BCD learning for num_topics = %d, num_iters = %d, gamma=%0.2f*K, "
                        "nu=%0.2f*D, reconstruction_error_lim = %0.2f"
                        % (num_topics, bcd_iters, gamma_frac, nu_frac, reconstruction_error_lim))
            self.H_matrix, self.W_matrix = \
                self.__block_coordinate_descent(num_iters=bcd_iters, H_init=H_init,
                                                reconstruction_error_lim=reconstruction_error_lim,
                                                gamma_frac=gamma_frac, nu_frac=nu_frac, sparse_mode=sparse_mode)
            assert (self.H_matrix.shape[0] == self.num_topics and self.H_matrix.shape[1] == self.vocab_size), \
                "H matrix learned has wrong dimenstions"
            assert (self.W_matrix.shape[0] == self.num_docs and self.W_matrix.shape[1] == self.num_topics), \
                "W matrix learned has wrong dimensions"
        else:
            if H_matrix_path is None or W_matrix_path is None:
                raise ValueError("If learn_model option is off, both H and W matrix paths need to be given")

            utils.verify_filename(H_matrix_path)
            utils.verify_filename(W_matrix_path)
            self.H_matrix = np.load(H_matrix_path)
            self.W_matrix = np.load(W_matrix_path)

            if self.num_topics is not None:
                assert (self.num_topics == self.H_matrix.shape[0] and self.num_topics == self.W_matrix.shape[1]), \
                    "Given num_topics do not match with H and W dimensions"
            else:
                assert (self.H_matrix.shape[0] == self.W_matrix.shape[1]), \
                    "W and H matrix don't exhibit same num_topics"
                self.num_topics = self.H_matrix.shape[0]

        if evaluation_mode and (not sparse_mode):
            self.topic_tuples = self.get_all_topic_tuples()
            self.document_topic_matrix = utils.l1_normalize_2darray(self.W_matrix).T
            self.topic_word_matrix = utils.l1_normalize_2darray(self.H_matrix).T
            self.make_document_topic_wt_matrix()
            # self.make_topic_word_matrix_argsorted(nnz=50)
        return

    def __block_coordinate_descent(self, num_iters, H_init, reconstruction_error_lim, gamma_frac, nu_frac, sparse_mode):
        K = self.num_topics
        N = self.X_matrix.shape[0]
        D = self.X_matrix.shape[1]
        logger.debug("N(num_docs)=%d, D(vocab_size)=%d, K(num_topics)=%d" % (N, D, K))
        X = normalize_X(X=self.X_matrix)

        Xt = scipy.sparse.csc_matrix(X.T)
        assert scipy.sparse.isspmatrix_csc(Xt)

        if H_init is not None:
            H = H_init
        else:
            H = softinit(K, D, nu_frac=nu_frac)
        if not scipy.sparse.isspmatrix_csr(H):
            H = scipy.sparse.csr_matrix(H)

        assert (H.shape == (K, D)), "H sofinit has wrong shape"

        # W = np.zeros((N, K))
        W = None

        R_old = sys.maxsize
        convergence_lim = N / 500
        for iter_num in xrange(num_iters):
            logger.debug("BCD ITERATION NUMBER: %d" % iter_num)
            W = sparse_coding.learn(X=Xt, H=H, N=N, gamma=int(gamma_frac * K), parallelization=True)
            logger.debug("Done with sparse coding phase")

            # X_reconstructed = W.dot(H)
            # assert (X_reconstructed.shape == X.shape), "X_reconstructed and X don't agree shapewise"
            # # reconstruction_error = np.linalg.norm(X - X_reconstructed, ord='fro') ** 2
            # reconstruction_error = scipy.sparse.linalg.norm(X - X_reconstructed, ord='fro') ** 2
            # logger.info("Reconstruction Error after sparse coding phase %d: %f"
            #              % (iter_num, reconstruction_error))
            # if reconstruction_error < reconstruction_error_lim:
            #     logger.info("Reconstruction error limit reached, stopping...")
            #     break

            H = sparse_dictionary.learn(X=Xt, H=H, W=W, nu_param=int(nu_frac * D))
            X_reconstructed = W.dot(H)
            assert (X_reconstructed.shape == X.shape), "x_reconstructed and x don't agree shapewise"
            # reconstruction_error = np.linalg.norm(X - X_reconstructed, ord='fro') ** 2
            reconstruction_error = scipy.sparse.linalg.norm(X - X_reconstructed, ord='fro') ** 2
            logger.debug("Reconstruction Error after dict learning phase %d: %f"
                         % (iter_num, reconstruction_error))
            if reconstruction_error < reconstruction_error_lim:
                logger.debug("Reconstruction error limit reached, stopping...")
                break
            if R_old - reconstruction_error < convergence_lim:
                logger.debug("Convergence, stopping BCD early...")
                break
            R_old = reconstruction_error
        # final_normerror = 0.
        # for i in xrange(N):
        #     final_normerror += (np.linalg.norm(X[i, :] - H.T.dot(W[i, :]), ord=2)) ** 2
        # final_normerror /= N
        # logger.info("Final Norm Error after BCD: %0.4f" % final_normerror)
        if not sparse_mode:
            W = utils.l1_normalize_2darray(W.toarray())
            return H.toarray(), W
        else:
            return H, W

    def save_H_matrix(self, H_matrix_path):
        np.save(H_matrix_path, self.H_matrix)
        return

    def save_W_matrix(self, W_matrix_path):
        np.save(W_matrix_path, self.W_matrix)
        return

    def get_topic_tuples(self, topicid_list=None, wordspertopic=10):
        if topicid_list is None:
            # logger.debug("Using all topics as topicid_list is none")
            topicid_list = xrange(self.num_topics)
        topic_tuples = []
        for topicid in topicid_list:
            topicid = int(topicid)
            topic_dist = self.H_matrix[topicid, :]
            topic_dist_norm = topic_dist / sum(topic_dist)
            topic_topword_indices = np.argsort(self.H_matrix[topicid, :])[::-1][:int(wordspertopic)]
            topic_tuples.append([(self.id2word_dict[int(idx)].strip('\r'), topic_dist_norm[int(idx)])
                                 for idx in topic_topword_indices])
        return topic_tuples

    def plot_topic_topwords(self, topicid_list=None, wordspertopic=10, cmaps=None, title='Dictionary Leanring'):
        topic_tuples = self.get_topic_tuples(topicid_list=topicid_list, wordspertopic=wordspertopic)
        super(BCD, self).plot_topic_topwords_base(topic_tuples=topic_tuples, cmaps=cmaps,
                                                  title=title)
        return

    def plot_dominant_topic_document_distribution(self, upper_threshold=0.4, lower_threshold=0.3,
                                                  kind='vbar'):
        super(BCD, self).plot_dominant_topic_document_distribution_base(W_matrix=self.W_matrix.T,
                                                                        upper_threshold=upper_threshold,
                                                                        lower_threshold=lower_threshold, kind=kind)
        return

    def plot_entropy_distribution(self, all_topics=True, topics=None, *args, **kwargs):
        if all_topics:
            if topics is not None:
                logger.warning("bcd.plot_entropy_distribution: User given topic list discarded as all_topics=True")
            topics = xrange(self.num_topics)
        else:
            assert (topics is not None), "If all_topics option is switched off, list of topics to print must be given"
        distributions = [self.H_matrix[topic_id, :] for topic_id in topics]
        super(BCD, self).plot_entropy_distribution_base(distributions, *args, **kwargs)
        return

    def plot_topic_entropy_colormap(self):
        distributions = [self.H_matrix[topic_id, :] for topic_id in xrange(self.num_topics)]
        super(BCD, self).plot_topic_entropy_colormap(distributions)
        return

    def save_topic_top_words(self, filename, num_topics, wordspertopic=10, separator=","):
        if num_topics > self.num_topics or num_topics == -1:
            if num_topics > self.num_topics:
                logger.warning("Given num_topics > num_topics of model, selecting all topics")
            topicid_list = xrange(self.num_topics)
        else:
            topicid_list = np.random.choice(self.num_topics, num_topics, replace=False)
        topic_tuples = self.get_topic_tuples(topicid_list=topicid_list, wordspertopic=wordspertopic)
        super(BCD, self).save_topic_top_words(word_weight_list=topic_tuples, filename=filename,
                                              separator=separator, datasetname=self.datasetname)
        return

    def get_all_topic_tuples(self, num_topics=None, wordspertopic=30):
        if num_topics is None:
            num_topics = self.num_topics
        if num_topics < self.num_topics:
            topicid_list = np.random.choice(self.num_topics, num_topics, replace=False)
        else:
            topicid_list = np.arange(num_topics)
        topic_tuples = self.get_topic_tuples(topicid_list=topicid_list, wordspertopic=wordspertopic)
        for i, topic_tuple in enumerate(topic_tuples):
            for j, tup in enumerate(topic_tuple):
                topic_tuples[i][j] = (tup[0], tup[1])
        return topic_tuples

    def save_topic_images(self, dirname):
        dirname = dirname + "bcd/"
        if not os.path.exists(dirname):
            logger.debug("Creating directory %s" % dirname)
            os.makedirs(dirname)
        for i, topic_tuple in enumerate(self.topic_tuples):
            new_topic_tuple = []
            for j, tup in enumerate(topic_tuple):
                new_topic_tuple.append((tup[0], tup[1] * tup[1]))
            super(BCD, self).plot_topic_topwords_base(topic_tuples=[new_topic_tuple], cmaps='Uniform',
                                                      title='BCD', save=True, show=False,
                                                      filename=dirname + "topic%d" % i, show_weight=True)
        return

    def get_highest_entropy_words(self, numwords=10):
        indices = self.get_highest_entropy_words_base(self.H_matrix.T, numwords=numwords)
        return indices

    def get_highest_freq_words(self, numwords=10):
        highest_freq_wordidc = np.argsort(self.vocab_corpus_frequency_list)[-numwords:][::-1]
        highest_freq_words = [self.id2word_dict[idx] for idx in highest_freq_wordidc]
        print(highest_freq_words)
        return highest_freq_words

    # def get_documents_avg_topic_dist(self, document_indices):
    #     if len(document_indices) == 0:
    #         raise ValueError("Lenght of document indices (for finding average topic vector) is 0")
    #     avg_topic_dist = np.zeros(self.num_topics)
    #     for idx in document_indices:
    #         topic_dist = self.document_topic_wt_matrix[:, idx]
    #         avg_topic_dist += topic_dist
    #     avg_topic_dist /= len(document_indices)
    #     return avg_topic_dist
    #
    # def get_documents_top_topic_tuples(self, document_indices):
    #     avg_topic_dist = self.get_documents_avg_topic_dist(document_indices)
    #     top3_topics = np.argsort(avg_topic_dist)[-3:][::-1]
    #     top3_topic_tuples = self.get_topic_tuples(topicid_list=top3_topics, wordspertopic=30)
    #     top3_topic_weights = avg_topic_dist[top3_topics]
    #     return top3_topic_tuples, top3_topic_weights

    def get_documents_avg_topic_dist(self, document_indices, imp):
        if len(document_indices) == 0:
            raise ValueError("Lenght of document indices (for finding average topic vector) is 0")
        if self.W_matrix is None:
            raise ValueError("Can't get document vectors, W matrix has not been learned")
        avg_topic_dist = np.zeros(self.num_topics)
        for i, idx in enumerate(document_indices):
            topic_dist = self.document_topic_wt_matrix[:, idx]
            avg_topic_dist += topic_dist*imp[i]
        avg_topic_dist /= len(document_indices)
        avg_topic_dist /= np.linalg.norm(avg_topic_dist, ord=1)
        return avg_topic_dist

    # def get_documents_top_topic_tuples(self, document_indices, imp):
    #     avg_topic_dist = self.get_documents_avg_topic_dist(document_indices, imp)
    #     top5_topics = np.argsort(avg_topic_dist)[-5:][::-1]
    #     top5_topics = np.sort(top5_topics)
    #     ranks = mici_selection(top5_topics, self.topic_word_matrix)
    #     top3_topics = top5_topics[np.argsort(ranks)[-3:][::-1]]
    #     top3_topic_tuples = self.get_topic_tuples(topicid_list=top3_topics, wordspertopic=30)
    #     top3_topic_weights = [max(0.0, x) for x in avg_topic_dist[top3_topics]]
    #     return top3_topic_tuples, top3_topic_weights

    def get_documents_top_topic_tuples(self, document_indices, imp):
        avg_topic_dist = self.get_documents_avg_topic_dist(document_indices, imp)
        X = self.document_topic_matrix.T
        y = []
        set_doc_idc = set(document_indices)
        for i, row in enumerate(X):
            if i in set_doc_idc:
                y.append(True)
            else:
                y.append(False)
        selector = SelectKBest(k=5)
        selector.fit(X, y)
        top5_topics = selector.get_support(indices=True)

        # top5_topics = np.argsort(avg_topic_dist)[-5:][::-1]
        # top5_topics = np.sort(top5_topics)
        # ranks = mici_selection(top5_topics, self.topic_word_matrix)
        # top3_topics = top5_topics[np.argsort(ranks)[-3:][::-1]]
        # noinspection PyTypeChecker
        top5_topic_tuples = self.get_topic_tuples(topicid_list=list(top5_topics), wordspertopic=30)
        top5_topic_weights = [max(0.0, x) for x in avg_topic_dist[top5_topics]]
        order = np.argsort(top5_topic_weights)[-3:][::-1]
        top3_topic_weights = [top5_topic_weights[x] for x in order]
        top3_topic_tuples = [top5_topic_tuples[x] for x in order]
        return top3_topic_tuples, top3_topic_weights
