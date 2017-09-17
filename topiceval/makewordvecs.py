#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

from topiceval import w2vparams as params

import gensim.utils
from gensim.models import Word2Vec
import numpy as np

import time
import logging

logger = logging.getLogger(__name__)


def read_corpus(train_data, tokens_only=False):
    for i, line in enumerate(train_data):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


def make_pvdbow(train_data, num_topics):
    train_corpus = list(read_corpus(train_data))
    model = gensim.models.doc2vec.Doc2Vec(documents=train_corpus,
                                          size=num_topics, min_count=params.min_wordcorpusfrequency, dm=0,
                                          window=params.context_window, hs=0, workers=params.num_workers,
                                          sample=params.downsampling, negative=params.negative_noisewords,
                                          seed=1, iter=params.pvd_iters)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return np.array(model.docvecs)


def make_wordvecs(train_data, num_topics):
    start = time.time()
    # Use the NLTK tokenizer to split the paragraph into sentences.
    # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    logger.debug("Parsing sentences from training set...")
    #
    # # Loop over each news article.
    for review in train_data:
        try:
            # Split a review into parsed sentences.
            # sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
            sentences += [review.strip().split(' ')]
        except Exception as e:
            logging.debug("Exception while parsing: %s" % e)
            continue
    # Params for word2vec training
    dimension = num_topics                      # Word vector dimensionality
    min_count = params.min_wordcorpusfrequency                               # Minimum word count
    num_workers = params.num_workers            # Number of threads to run in parallel
    window = params.context_window              # Context window size
    downsampling = params.downsampling          # Downsample setting for frequent words
    negative = params.negative_noisewords       # negative nosie words to be drawn in neg sampling
    iters = params.iters
    logger.debug("Training Word2Vec model with negative sampling, num_sentences=%d, dimension=%d, context-window=%d, "
                 "num_its=%d, negative_noisewords=%d, downsampling=%0.5f, num_workers=%d"
                 % (len(sentences), dimension, window, iters, negative, downsampling, num_workers))
    # Train Word2Vec model.
    model = Word2Vec(sentences, workers=num_workers, hs=0, sg=1, negative=negative, iter=iters,
                     size=dimension, min_count=min_count, window=window, sample=downsampling, seed=1)

    # abs_path = os.path.dirname(os.path.abspath(__file__))

    wordvecs = model.wv
    wordvecs.init_sims(replace=True)
    # Save Word2Vec model.
    endmodeltime = time.time()
    logger.debug("time taken: %0.1f" % (endmodeltime - start))
    return wordvecs
