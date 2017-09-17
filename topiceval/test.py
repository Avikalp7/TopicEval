"""
**** Ignore this file, its use for local testing purposes ****
"""

from __future__ import division
from __future__ import print_function

import topiceval.plotter as plotter
from topiceval.lda.gensimlda import LDA
from topiceval.dictionarylearning.bcd import BCD
from topiceval.thresholdedsvd.tsvd import TSVD
# import topiceval.usereval.topiceval_application as application
from topiceval.coherence import semantic

import scipy.sparse
import numpy as np
from six.moves import xrange

import pickle
import logging

# FORMAT = "[%(asctime)s][%(filename)s - %(funcName)10s()] %(message)s"
# logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt="%I:%M:%S")

def plot_temp(topicid_list, topic_tuples, cmaps):
    if cmaps is None:
        cmaps = ['Blues', 'Greens', 'Reds', 'Purples', 'Greys'] * (int(len(topicid_list) / 5) + 1)
    elif cmaps == 'Uniform':
        cmaps = ['Blues'] * len(topicid_list)
    plotter.plot_text_intensity_plot(topic_tuples, cmaps)

# my_lda = lda(model_path="./data/nips.lda.model", vocab_filepath="./data/docword.nips.vocab.trunc.txt",
#              docword_filepath="./data/docword.nips.proc.txt", num_topics=50)
# my_lda.save_lda_model("./data/nips.lda.model")
# my_lda.save_topic_top_words("./data/topi/csnips.txt", "nips", num_topics=50, topn=10)
# my_lda.plot_dominant_topic_document_distribution(kind='vbar')
# my_lda.plot_topic_wordcloud(topicid=5, num_words=20)
# my_lda.plot_comparative_topic_wordclouds(topicid1=6, topicid2=16, num_words=15)
# my_lda.plot_topic_topwords([7, 18, 39])
# my_lda.plot_entropy_distribution(topics=np.arange(50))
# my_lda.plot_topic_entropy_colormap()


if __name__ == '__main__':
    datasetname = "nyt"
    # with open("./data/id_vocab_dict_" + datasetname + ".pickle", "rb") as handle:
    #   id_vocab_dict = pickle.load(handle)
    # Xt = np.load("./data/A_" + datasetname + ".npy")
    # X = Xt.T
    # print("N=%d, D=%d" % (X.shape[0], X.shape[1]))

    num_topics = 30

    # model = LDA (datasetname=datasetname,
    #              A_matrix_path="C:\\Users\\t-avsriv\\Source\\EnronAnalysis\\data_nips\\A_original.npy",
    #              vocab_filepath="C:\\Users\\t-avsriv\\Source\\EnronAnalysis\\data_nips\\docword.nips.vocab.trunc.txt",
    #              num_topics=num_topics, evaluation_mode=True)
    # model = LDA(datasetname=datasetname,
    #             corpus_path="../data_topiceval/userdata/corpus.npy",
    #             id2word_dict_path="../data_topiceval/userdata/id2word_dict.pickle",
    #             num_topics=num_topics, evaluation_mode=True)
    # model.plot_dominant_topic_document_distribution(kind='vbar')
    # model.plot_dominant_topic_document_distribution(kind='colormap')
    # exit()


    # model = BCD (datasetname=datasetname,
    #             A_matrix_path="C:\\Users\\t-avsriv\\Source\\EnronAnalysis\\data_nips\\A_original.npy",
    #             vocab_filepath="C:\\Users\\t-avsriv\\Source\\EnronAnalysis\\data_nips\\docword.nips.vocab.trunc.txt",
    #             num_topics=num_topics, evaluation_mode=True, bcd_iters=12, gamma_frac=0.15, nu_frac=0.15)
    # print(model.representative_topic_tuples)
    # model.save_topic_top_words(filename="bcdwordsnyt.txt", num_topics=num_topics)

    # model = TSVD(datasetname=datasetname,
    #              docword_filepath="C:\\Users\\t-avsriv\\Source\\Repos\\topic_modeling\\ConsoleApplication1\\"
    #                               "data\\NYTimes_Vocab5k_TrainDocs30k\\TrainData.VocabIndex1.DocIndex1.txt",
    #              vocab_filepath="C:\\Users\\t-avsriv\\Source\\Repos\\topic_modeling\\ConsoleApplication1\\"
    #                             "data\\NYTimes_Vocab5k_TrainDocs30k\\vocab.txt", num_topics=num_topics,
    #              evaluation_mode=False)
    # model.save_topic_top_words(filename="tsvdwordsnyt.txt", num_topics=num_topics)

    model = TSVD(datasetname=datasetname,
                A_matrix_path="C:\\Users\\t-avsriv\\Source\\EnronAnalysis\\data_nips\\A_original.npy",
                vocab_filepath="C:\\Users\\t-avsriv\\Source\\EnronAnalysis\\data_nips\\docword.nips.vocab.trunc.txt",
                num_topics=num_topics, evaluation_mode=True)
    model.plot_dominant_topic_document_distribution(kind='colormap')
    model.plot_dominant_topic_document_distribution(kind='vbar')
    # model.plot_topic_topwords(topicid_list=np.arange(10))
    # model.plot_entropy_distribution()
    # model.plot_topic_entropy_colormap()
    # print(model.topicwise_catchwords[0])
    # model.save_topic_top_words(filename="tsvdwordsnyt.txt", num_topics=num_topics)

    # model = BCD(datasetname=datasetname,
    #             A_matrix_path="C:\\Users\\t-avsriv\\Source\\EnronAnalysis\\data_nips\\A_original.npy",
    #             vocab_filepath="C:\\Users\\t-avsriv\\Source\\EnronAnalysis\\data_nips\\docword.nips.vocab.trunc.txt",
    #             num_topics=num_topics, evaluation_mode=False, bcd_iters=12)

    # model = BCD(datasetname=datasetname,
    #             X_matrix=scipy.sparse.random(20000, 20000, density=0.04, format='lil',
    #                                          dtype=np.float64, random_state=None),
    #             id2word_dict={},
    #             num_topics=num_topics, evaluation_mode=False, bcd_iters=3)

    # model = BCD (datasetname=datasetname,
    #              docword_filepath="C:\\Users\\t-avsriv\\Source\\Repos\\topic_modeling\\ConsoleApplication1\\data\\"
    #                               "NYTimes_Vocab5k_TrainDocs30k\\TrainData.VocabIndex1.DocIndex1.txt",
    #              vocab_filepath="C:\\Users\\t-avsriv\\Source\\Repos\\topic_modeling\\ConsoleApplication1\\data\\"
    #                             "NYTimes_Vocab5k_TrainDocs30k\\vocab.txt",
    #              num_topics=num_topics, evaluation_mode=False, bcd_iters=3)
    exit()

    from scipy.stats import entropy
    from scipy.stats import spearmanr
    import numpy as np
    w = model.W_matrix
    entropies = [entropy(w[:, col]) for col in xrange(w.shape[1])]
    vars = [np.var(w[:, col]) for col in xrange(w.shape[1])]
    topicsbyentropy = np.argsort(entropies)
    topicsbyvar = np.argsort(vars)[::-1]
    for topicnum in topicsbyvar[0:3]:
        print(np.sort(w[:,topicnum])[::-1][0:5])
        print(model.representative_topic_tuples[topicnum][0:8])
    print("\n--------------------------------------------------------------\n")
    for topicnum in topicsbyentropy[0:5]:
        print(model.representative_topic_tuples[topicnum][0:8])

    topics_reqd = []
    doc_lens = []
    for docnum in xrange(w.shape[0]):
        doc = sorted(w[docnum, :], reverse=True)
        docsum = sum(doc)
        doc = [d/docsum for d in doc]
        sumprob = 0.
        elems = 0
        while sumprob < 0.8:
            sumprob += doc[elems]
            elems += 1
        topics_reqd.append(elems)
        doc_lens.append(np.count_nonzero(model.X_matrix[docnum, :]))
        if elems > 6:
            print(">6 (%d): doc len: " % elems, doc_lens[docnum])
        # if elems < 2:
        #     print("<2 (%d): doc len: " % elems, np.count_nonzero(model.X_matrix[docnum, :]))
    print("avg doc size:", np.mean(doc_lens))
    print("spearmanr: ", spearmanr(topics_reqd, doc_lens))
    import matplotlib.pyplot as plt
    plt.hist(topics_reqd, bins=max(topics_reqd))
    plt.axis([0, max(topics_reqd), 0, 600])
    plt.show()
    exit()

    topics = model.representative_topic_tuples
    semantic_pmi = []
    print(model.modelname)
    for n, topic in enumerate(topics):
        semantic_pmi.append(semantic.semantic_coherence(model, topic, numwords=15))
    print(np.mean(semantic_pmi))
    print(np.var(semantic_pmi))
    model.save_topic_top_words(model.modelname + model.datasetname + "_topics.txt", num_topics=num_topics,
                               wordspertopic=15)

    idc = model.get_highest_entropy_words(numwords=1000)
    all_idc = np.arange(model.vocab_size)
    idckeep = set(all_idc) - set(idc)
    idckeeplist = sorted(list(idckeep))
    A = model.A_matrix[idckeeplist, :]
    np.save("tempA.npy", A)
    print("NEW A SHAPE: ", A.shape)
    id2wdict = {}
    idx = 0
    for id in all_idc:
        if id in idckeep:
            id2wdict[idx] = model.id2word_dict[id]
            idx += 1
    print("id2wdict: ", len(id2wdict))
    model2 = BCD(datasetname=datasetname+"trunc", A_matrix_path="tempA.npy", id2word_dict=id2wdict,
                 evaluation_mode=True, bcd_iters=5, num_topics=num_topics)
    topics = model2.representative_topic_tuples
    semantic_pmi = []
    print(model2.modelname)
    for n, topic in enumerate(topics):
        semantic_pmi.append(semantic.semantic_coherence(model2, topic, numwords=15))
    print(np.mean(semantic_pmi))
    print(np.var(semantic_pmi))
    model2.save_topic_top_words(model2.modelname + model2.datasetname + "_topics.txt", num_topics=num_topics,
                                wordspertopic=15)


    # model.get_highest_freq_words(numwords=30)

    # model = TSVD(datasetname=datasetname,
    #              docword_filepath="C:\\Users\\t-avsriv\\Source\\Repos\\topic_modeling\\ConsoleApplication1\\data\\"
    #                               "enron\\docword.enron.txt",
    #              vocab_filepath="C:\\Users\\t-avsriv\\Source\\Repos\\topic_modeling\\ConsoleApplication1\\data\\"
    #                             "enron\\vocab.enron.txt",
    #              num_topics=num_topics, evaluation_mode=False)

    # topics = model.representative_topic_tuples
    # semantic_pmi = []
    # print(model.modelname)
    # for n, topic in enumerate(topics):
    #     semantic_pmi.append(semantic.semantic_coherence(model, topic, numwords=15))
    # print(np.mean(semantic_pmi))
    # print(np.var(semantic_pmi))
    # model.save_topic_top_words(model.modelname+datasetname+"_topics.txt", num_topics=num_topics, wordspertopic=15)

    # with open("modellda.pkl", "wb") as handle:
    #     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # dictionary_learning_topic_model = BCD(datasetname=datasetname, learn_model=True,
    #                                       A_matrix_path="./data/common/A_" + datasetname + ".npy",
    #                                       id2word_dict_path="./data/common/id_vocab_dict_" + datasetname + ".pickle",
    #                                       num_topics=num_topics, evaluation_mode=False, gamma_frac=0.25, nu_frac=0.05)

    # dictionary_learning_topic_model = BCD(datasetname=datasetname, learn_model=True,
    #                                       corpus_path="./data/" + datasetname + "/corpus.npy",
    #                                       id2word_dict_path="./data/" + datasetname + "/id2word_dict.pickle",
    #                                       num_topics=num_topics, evaluation_mode=True, gamma_frac=0.25, nu_frac=0.05)

    # dictionary_learning_topic_model = BCD(learn_model=False, A_matrix_path="./data/A_" + datasetname + ".npy",
    #                                       id2word_dict_path="./data/id_vocab_dict_" + datasetname + ".pickle",
    #                                       num_topics=num_topics, evaluation_mode=False,
    #                                       H_matrix_path="./data/H_matrix_" + datasetname + ".npy",
    #                                       W_matrix_path="./data/W_matrix_" + datasetname + ".npy", vocab_size=5002)
    # H, W = dictionary_learning_topic_model.H_matrix, dictionary_learning_topic_model.W_matrix

    # # dictionary_learning_topic_model.save_HW_matrices("./data/H_matrix_" + datasetname + ".npy", "./data/W_matrix_" +
    # #                                                  datasetname + ".npy")

    # dictionary_learning_topic_model.plot_topic_topwords(np.arange(num_topics), wordspertopic=5)
    # dictionary_learning_topic_model.plot_dominant_topic_document_distribution()
    # dictionary_learning_topic_model.plot_entropy_distribution()
    # dictionary_learning_topic_model.plot_topic_entropy_colormap()
    # dictionary_learning_topic_model.save_topic_top_words(filename="./data/topics"+datasetname+".txt",
    #                                                      datasetname=datasetname, num_topics=-1)
    # for k in xrange(num_topics):
    #     print("\n####### TOPIC NUMBER: %d #######" % k)
    #     topic_topword_indices = np.argsort(H[k, :])[::-1]
    #     for idx in topic_topword_indices[0:15]:
    #         print(dictionary_learning_topic_model.id2word_dict[idx])

    # tsvd_topic_model = TSVD(datasetname=datasetname, learn_model=True,
    #                         A_matrix_path="./data/common/A_" + datasetname + ".npy",
    #                         id2word_dict_path="./data/common/id_vocab_dict_" + datasetname + ".pickle",
    #                         docword_filepath="./data/common/docword." + datasetname + ".proc.txt",
    #                         num_topics=num_topics, evaluation_mode=False)
    # tsvd_topic_model.save_M_matrix()

    # tsvd_topic_model = TSVD(datasetname=datasetname, learn_model=False,
    #                         A_matrix_path="./data/common/A_" + datasetname + ".npy",
    #                         id2word_dict_path="./data/common/id_vocab_dict_" + datasetname + ".pickle",
    #                         docword_filepath="./data/common/docword." + datasetname + ".proc.txt",
    #                         M_matrix_path="./data/tsvd/results_" + datasetname +
    #                                       "_eps1_0.10_eps2_0.33_eps3_5.00_topics_15/M.npy",
    #                         W_matrix_path="./data/tsvd/results_" + datasetname +
    #                                       "_eps1_0.10_eps2_0.33_eps3_5.00_topics_15/W.npy", num_topics=num_topics)
    # tsvd_topic_model.plot_topic_topwords(topicid_list=np.arange(num_topics/2))
    # tsvd_topic_model.plot_dominant_topic_document_distribution(kind='colormap')
    # tsvd_topic_model.plot_dominant_topic_document_distribution(kind='vbar')
    # tsvd_topic_model.save_topic_top_words("./data/tsvd_topics" + datasetname + str(num_topics) +
    #                                       ".txt", num_topics=num_topics)
    # tsvd_topic_model.plot_entropy_distribution(all_topics=True)
    # tsvd_topic_model.plot_topic_entropy_colormap()

    # import pickle
    # with open("./data/id_vocab_dict_" + datasetname + ".pickle", "rb") as handle:
    #     id2word_dict = pickle.load(handle)
    # M = utils.read_tsvd_M_hat_matrix(20, 5002, 1065, "nips", norm=True)
    # topicid_list = np.arange(10)
    # topic_tuples = []
    # for k in xrange(10):
    #     top_words = np.argsort(M[:, k])[::-1][:10]
    #     topic_tuples.append([(id2word_dict[word_idx], M[word_idx, k]) for word_idx in top_words])
    # plot_temp(topicid_list, topic_tuples, cmaps=None)
