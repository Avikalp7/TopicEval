from __future__ import division
from __future__ import print_function

import topiceval.preprocessing.params as params
from topiceval.preprocessing.emailstructure import EmailNetwork

from gensim import corpora
from gensim.models import phrases
import numpy as np
import scipy.sparse
# import pandas as pd

from collections import defaultdict
import re
import logging
import os
from six.moves import xrange
import operator

logger = logging.getLogger(__name__)


def remove_threads(text):
    """ Remove added messages from threaded mails, retaining only the most recent mail """
    text = re.sub(r'^\t?From: .*', ' ', text, flags=re.MULTILINE | re.DOTALL)
    return text


def remove_signature(text):
    """ Remove signature elements, and following text entirely/until a new message is encountered"""
    signatures = ["best", "thanking you", "thanks", "yours sincerely", "sincerely", "warm regards", "regards",
                  "best regards"]
    for signature in signatures:
        text = re.sub(r'^%s[^a-z]*?$.*?(<meta>){0, 1}' % signature, ' ', text,
                      flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    # for name in names:
    #     text = re.sub(r'^%s.*' % name, ' ', text, flags=re.IGNORECASE)
    return text


def phraser(text, bigram):
    """ Given gensim's phraser.Phraser instance, coalesce found bigrams in text """
    return ' '.join(bigram[text.split()])


def phrase_detection(df):
    """ Given the emails dataframe, form bigrams based on the text in "Body" field """
    sentences = [text.split() for text in df["Body"]]
    phrases_ = phrases.Phrases(sentences, min_count=params.bigrams_min_count, threshold=params.bigrams_threshold)
    bigram = phrases.Phraser(phrases_)
    # for phr, score in phrases_.export_phrases(sentences):
    #     print(u'{0}   {1}'.format(phr, score))
    return bigram


def remove_redundant_threads(df):
    bool_list = []
    for i in xrange(0, df.shape[0] - 1):
        if df.iloc[i]['ConversationID'] == df.iloc[i + 1]['ConversationID']:
            bool_list.append(False)
        else:
            bool_list.append(True)
    bool_list.append(True)
    return bool_list


def make_doc2bow(df, dirname, threaded, wordvecs):
    texts = [[word for word in document.split()] for document in df['CleanBody']]
    texts = prune_word_frequency(texts=texts, least_freq=params.word_least_corpus_frequency)
    id2word_dict = corpora.Dictionary(texts)
    ''' Prune n most frequent words '''
    id2word_dict.filter_n_most_frequent(params.filter_n_most_frequent)
    ''' Remove words in < word_least_document_frequency docs and in more than word_highest_document_fraction of docs, 
        and keep vocab size to maximum of max_vocab '''
    word_least_doc_frequency = params.word_least_document_frequency
    if not threaded:
        word_least_doc_frequency *= 2
    id2word_dict.filter_extremes(no_below=params.word_least_document_frequency,
                                 no_above=params.word_highest_document_fraction, keep_n=params.max_vocab)
    logger.info("Total vocabulary size: %d" % len(id2word_dict))
    ''' Making corpus '''
    corpus = [id2word_dict.doc2bow(text) for text in texts]
    ''' Truncate corpus by removing docs of length < document_min_length '''
    doc_indices_to_keep = truncate_corpus(corpus, min_len_doc=params.document_min_length)
    bool_list_indices_to_keep = make_bool_list(len(corpus), doc_indices_to_keep)
    df = df[bool_list_indices_to_keep].copy()
    df['idx'] = xrange(len(df))
    df = df.set_index('idx', drop=False)
    logger.info("Number of emails retained: %d (of %d)" % (len(doc_indices_to_keep), len(corpus)))
    trunc_corpus = [corpus[i] for i in doc_indices_to_keep]
    A = make_Amatrix_from_corpus(trunc_corpus, id2word_dict)
    email_network = EmailNetwork(df, id2word_dict, wordvecs)
    logger.info("Username detected: %s, Num custom folders: %d, Avg folder len: %0.2f"
                % (email_network.username, len(email_network.custom_folders), email_network.avg_folder_len))
    logger.info("Number of big folders: %d, Total users: %d" % (len(email_network.big_folders),
                                                                len(email_network.all_users)))
    del wordvecs
    ''' These are only aggregate statistics, no emails (or their metadata) can be reconstructed '''
    logger.debug("Saving id2word_dict and corpus...")
    id2word_dict.save(dirname + 'id2word_dict.pickle')
    # scipy.sparse.save_npz(dirname + "Amatrix.npz", A)
    np.save(dirname + "corpus.npy", trunc_corpus)
    # np.save(dirname + "dfindices_in_corpus.npy", doc_indices_to_keep)
    return A, email_network


def prune_word_frequency(texts, least_freq):
    """ Remove words based on corpus frequency.

    :param texts: list of lists, each list corresponds to a document, containing word tokens
    :param least_freq: words with corpus frequency below this are removed

    :return: texts (list of lists), with pruned word tokens
    """
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] >= least_freq] for text in texts]
    return texts


def truncate_corpus(corpus, min_len_doc):
    good_indices = [i for i, doc in enumerate(corpus) if sum(n for _, n in doc) >= min_len_doc]
    return good_indices


# def replace_names(text, names):
#     for name in names:
#         try:
#             text = re.sub(r'%s' % name, '<name>', text)
#         except:
#             continue
#     return text


def make_Amatrix_from_corpus(corpus, id2word_dict):
    # A = np.zeros((self.vocab_size, self.num_docs))
    vocab_size = len(id2word_dict)
    num_docs = len(corpus)
    A = scipy.sparse.dok_matrix((vocab_size, num_docs), dtype=np.float32)
    for doc_num in xrange(num_docs):
        for tup in corpus[doc_num]:
            A[tup[0], doc_num] = tup[1]
    return scipy.sparse.csr_matrix(A)


def make_bool_list(n, indices):
    # bool_list = [False] * n
    # for idx in indices:
    #     bool_list[idx] = True
    bool_list = [idx in indices for idx in xrange(n)]
    return bool_list


def add_reply_field(df):
    reply_field = []
    username = __get_username(df)
    for i in xrange(0, df.shape[0] - 1):
        row = df.iloc[i]
        nextrow = df.iloc[i + 1]
        if row['ConversationID'] == nextrow['ConversationID'] and row["SenderName"] != username\
                and nextrow["SenderName"] == username:
            reply_field.append(True)
        else:
            reply_field.append(False)
    reply_field.append(False)
    logger.info("Number of messages replied to: %d (of %d)" % (reply_field.count(True), len(reply_field)))
    df['replied'] = reply_field
    return df


def add_to_cc_bcc_field(df):
    to_cc_bcc_field = []
    n_cc_bcc = []
    for i in xrange(df.shape[0]):
        row = df.iloc[i]
        to_cc_bcc = (row['To'] + ";" + row["CC"] + ";" + row["BCC"]).replace("<UNKNOWN>", "")
        to_cc_bcc = ';'.join([addr for addr in to_cc_bcc.split(';') if addr != ''])
        to_cc_bcc_field.append(to_cc_bcc)

        ncc, nbcc = 0, 0
        if row["CC"] != "<UNKNOWN>" and row["CC"].strip() != '':
            ncc = len(row["CC"].strip().split(';'))
        if row["BCC"] != "<UNKNOWN>" and row["BCC"].strip() != '':
            nbcc = len(row["BCC"].strip().split(';'))
        n_cc_bcc.append(ncc + nbcc)
    df['to_cc_bcc'] = to_cc_bcc_field
    df['n_cc_bcc'] = n_cc_bcc
    logger.info("Number of mails with atleast 1 CC/BCC entry: %d of %d" % (sum(1 for i in n_cc_bcc if i > 0), len(df)))
    return df


def clean_email_header(text):
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    return text


def load_stops():
    path = os.path.dirname(os.path.abspath(__file__))
    path = '/'.join(path.split("\\")[:-1])
    try:
        stop = np.append(np.load(path + "/data/stopwords_extended.npy"), "<meta>")
        stop = np.append(stop, "dear")  # email specific
        stop = np.append(stop, "sir")  # email specific
        stop = np.append(stop, "maam")  # email specific
    except FileNotFoundError:
        stop = np.append(np.load(path + "/topiceval/data/stopwords_extended.npy"), "<meta>")
        stop = np.append(stop, "dear")  # email specific
        stop = np.append(stop, "sir")  # email specific
        stop = np.append(stop, "maam")  # email specific
    stop = set(list(stop))
    return stop


def __get_username(df):
    """ Assign the sender with most emails sent as the user name """
    sender_stats = df[df["FolderType"] == "sent_items"].groupby("SenderName")["SentOn"].count()
    sender_stats_dict = dict(zip(sender_stats.index, sender_stats.data))
    sorted_sender_stats = sorted(sender_stats_dict.items(), key=operator.itemgetter(1), reverse=True)
    username = str(sorted_sender_stats[0][0])
    if username.upper() == "<UNKNOWN>":
        logger.warning("Username detected as <UNKNOWN>! Changing to second highest sender...")
        try:
            username = str(sorted_sender_stats[1][0])
        except IndexError:
            logger.error("Username could not be detected!")
    return username
