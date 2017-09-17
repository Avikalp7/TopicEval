"""
This module implements LDA using the gensim library, with extra utilities for topic model diagnostics.
"""

from __future__ import print_function
from __future__ import division

import topiceval.utils as utils
from topiceval.basemodel import BaseModel
from topiceval.lda import params
from topiceval.usereval.task_evaluation import mici_selection

import numpy as np
import gensim
from six.moves import xrange
from sklearn.feature_selection import SelectKBest

import logging
import os

logger = logging.getLogger(__name__)


class LDA(BaseModel):
    def __init__(self, datasetname, learn_model=True, model_path=None, id2word_dict=None, id2word_dict_path=None,
                 corpus=None, corpus_path=None, A_matrix_path=None, vocab_filepath=None, docword_filepath=None,
                 vocab_size=None, num_docs=None, num_topics=20, M_matrix_path=None, W_matrix_path=None,
                 evaluation_mode=False, Amatrix_needed=False):
        # TODO: Given model path name, do an auto search for id2word_dict based on filename.id2word
        self.modelname = "lda"
        super(LDA, self).__init__(datasetname=datasetname, id2word_dict=id2word_dict,
                                  id2word_dict_path=id2word_dict_path, corpus=corpus,
                                  corpus_path=corpus_path, A_matrix_path=A_matrix_path, vocab_filepath=vocab_filepath,
                                  docword_filepath=docword_filepath, vocab_size=vocab_size, num_docs=num_docs,
                                  evaluation_mode=True, Amatrix_needed=Amatrix_needed)
        self.num_topics = num_topics

        if not learn_model:
            assert (model_path is not None), "model_path required if learn_model option is turned off"
            utils.verify_filename(model_path)
            utils.verify_filename(model_path + ".state")
            self.lda_model = gensim.models.ldamodel.LdaModel.load(model_path)
        else:
            self.lda_model = self.run_lda()

        if M_matrix_path is not None:
            utils.verify_filename(M_matrix_path)
            self.M_matrix = np.load(M_matrix_path)
            print("Warning: The topic-word matrix is loaded from user specified file, not directly from lda model")
        else:
            self.M_matrix = self.get_M_matrix()

        assert (self.M_matrix.shape[0] == self.vocab_size and self.M_matrix.shape[1] == self.num_topics), \
            "Topic-Word distribution matrix dimenstions do not agree with supplied num-topics or vocab_size"

        if evaluation_mode:
            if W_matrix_path is not None:
                utils.verify_filename(W_matrix_path)
                self.W_matrix = np.load(W_matrix_path)
                logger.warning("Warning: The document-topic matrix is loaded from user specified file, not directly "
                               "from lda model")
            else:
                self.W_matrix = self.get_W_matrix()
            assert (self.W_matrix.shape[0] == self.num_topics and self.W_matrix.shape[1] == self.num_docs), \
                "Document-Topic distribution matrix dimenstions do not agree with supplied num-topics or num_docs"
            self.topic_tuples = self.get_all_topic_tuples()
            # self.make_topic_word_matrix_argsorted(nnz=50)

        self.topic_word_matrix = self.M_matrix
        self.document_topic_matrix = self.W_matrix
        self.make_document_topic_wt_matrix()
        return

    def run_lda(self):
        if self.num_docs < 1000:
            passes = params.passes_low
        elif self.num_docs < 5000:
            passes = params.passes_mid
        else:
            passes = params.passes_high
        lda_model = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.num_topics, id2word=self.id2word_dict,
                                                    random_state=params.random_state, passes=passes,
                                                    iterations=params.iterations, eval_every=params.eval_every,
                                                    alpha=params.alpha, eta=params.eta,
                                                    gamma_threshold=params.gamma_threshold,
                                                    minimum_probability=params.minimum_probability,
                                                    minimum_phi_value=params.minimum_phi_value)
        return lda_model

    def get_M_matrix(self):
        topics_terms = self.lda_model.state.get_lambda()
        M = topics_terms.T
        M = self.__normalize_M_matrix(M)
        return M

    @staticmethod
    def __normalize_M_matrix(M):
        for col in xrange(M.shape[1]):
            M[:, col] = M[:, col] / sum(M[:, col])
        return M

    def get_W_matrix(self):
        W = np.zeros((self.num_topics, self.num_docs))
        for doc_id in xrange(self.num_docs):
            Wj = self.get_topic_distribution_as_array(self.num_topics,
                                                      self.lda_model.get_document_topics(self.corpus[doc_id],
                                                                                         minimum_probability=0.0))
            W[:, doc_id] = Wj
        return W

    @staticmethod
    def get_topic_distribution_as_array(num_topics, distribution):
        Wj = np.array([tup[1] for tup in distribution])
        assert (len(Wj) == num_topics), "Wj's length doesn't match number of topics, can't fit in W!"
        # noinspection PyTypeChecker
        assert (0.99 < sum(Wj) < 1.01), "Sum of probability distribution off tolerated limit!"
        return Wj

    def plot_dominant_topic_document_distribution(self, upper_threshold=0.4, lower_threshold=0.3,
                                                  kind='vbar'):
        super(LDA, self).plot_dominant_topic_document_distribution_base(W_matrix=self.W_matrix,
                                                                        upper_threshold=upper_threshold,
                                                                        lower_threshold=lower_threshold, kind=kind)
        return

    def plot_topic_wordcloud(self, topicid, num_words=20, word_frequency_dict=None, figsize=(6, 3)):
        topic_topn_tuples = self.lda_model.show_topic(topicid=topicid, topn=num_words)
        word_weight_dict = {}
        for tup in topic_topn_tuples:
            word_weight_dict[str(tup[0])] = tup[1]
        super(LDA, self).plot_topic_wordcloud(topicid=topicid, num_words=num_words,
                                              frequencies=word_weight_dict, figsize=figsize)
        return

    def plot_comparative_topic_wordclouds(self, topicid1, topicid2, num_words=20, figsize=(6, 3)):
        topic1_topn_tuples = self.lda_model.show_topic(topicid=topicid1, topn=num_words)
        topic2_topn_tuples = self.lda_model.show_topic(topicid=topicid2, topn=num_words)
        super(LDA, self).plot_comparative_topic_wordclouds_base(num_words,
                                                                frequencies1=topic1_topn_tuples,
                                                                frequencies2=topic2_topn_tuples, figsize=figsize)
        return

    def get_topic_tuples(self, topicid_list=None, wordspertopic=10):
        if topicid_list is None:
            topicid_list = xrange(self.num_topics)
        topic_tuples = []
        for topicid in topicid_list:
            topic_tuples.append(self.lda_model.show_topic(topicid=topicid, topn=wordspertopic))
        return topic_tuples

    def plot_topic_topwords(self, topicid_list=None, wordspertopic=10, cmaps=None, title='LDA'):
        topic_tuples = self.get_topic_tuples(topicid_list=topicid_list, wordspertopic=wordspertopic)
        super(LDA, self).plot_topic_topwords_base(topic_tuples=topic_tuples, cmaps=cmaps, title=title)
        return

    def plot_entropy_distribution(self, all_topics=True, topics=None, *args, **kwargs):
        if all_topics:
            topics = xrange(self.num_topics)
        else:
            assert (topics is not None), "If all_topics option is switched off, list of topics to print must be given"
        distributions = [self.M_matrix[:, topic_id] for topic_id in topics]
        super(LDA, self).plot_entropy_distribution_base(distributions, *args, **kwargs)
        return

    def plot_topic_entropy_colormap(self):
        distributions = [self.M_matrix[:, topic_id] for topic_id in xrange(self.num_topics)]
        super(LDA, self).plot_topic_entropy_colormap(distributions)
        return

    def save_lda_model(self, filename):
        self.lda_model.save(filename)
        return

    def save_topic_top_words(self, filename, num_topics, wordspertopic=10, separator=","):
        if num_topics > self.num_topics or num_topics == -1:
            if num_topics > self.num_topics:
                logger.warning("Given num_topics > num_topics of model, selecting all topics")
            num_topics = self.num_topics
        words_list = self.lda_model.show_topics(num_topics=num_topics, num_words=wordspertopic, formatted=False)
        word_weight_list = [big_tup[1] for big_tup in words_list]
        super(LDA, self).save_topic_top_words(word_weight_list=word_weight_list, filename=filename,
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
        return topic_tuples

    def save_topic_images(self, dirname):
        dirname = dirname + "lda/"
        if not os.path.exists(dirname):
            logger.debug("Creating directory %s" % dirname)
            os.makedirs(dirname)
        for i, topic_tuple in enumerate(self.topic_tuples):
            super(LDA, self).plot_topic_topwords_base(topic_tuples=[topic_tuple], cmaps='Uniform',
                                                      title='lda', save=True, show=False,
                                                      filename=dirname + "topic%d" % i, show_weight=True)
        return

    def save_M_matrix(self, M_matrix_path):
        np.save(M_matrix_path, self.M_matrix)
        return

    def save_W_matrix(self, W_matrix_path):
        if self.W_matrix is None:
            logger.error("W matrix not learned, cannot save")
        else:
            np.save(W_matrix_path, self.W_matrix)
        return

    # def get_documents_avg_topic_dist(self, document_indices):
    #     if len(document_indices) == 0:
    #         raise ValueError("Lenght of document indices (for finding average topic vector) is 0")
    #     if self.W_matrix is None:
    #         raise ValueError("Can't get document vectors, W matrix has not been learned")
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
        top5_topic_tuples = self.get_topic_tuples(topicid_list=list(top5_topics), wordspertopic=30)
        top5_topic_weights = [max(0.0, x) for x in avg_topic_dist[top5_topics]]
        ord = np.argsort(top5_topic_weights)[-3:][::-1]
        top3_topic_weights = [top5_topic_weights[x] for x in ord]
        top3_topic_tuples = [top5_topic_tuples[x] for x in ord]
        return top3_topic_tuples, top3_topic_weights
