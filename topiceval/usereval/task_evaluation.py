from __future__ import print_function
from __future__ import division

from topiceval.stats import hellinger_distance
from topiceval.preprocessing import textcleaning
from topiceval.preprocessing import emailsprocess

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn import metrics
# from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from prettytable import PrettyTable
import scipy.sparse.linalg
from scipy.stats import pearsonr

import operator
import logging

import matplotlib

matplotlib.use("Qt5Agg", force=True)
# import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

EPS = 1e-3


def modified_jaccard_similarity(set1, set2):
    return len(set1 & set2) / (len(set1) + EPS)


def reorder(X_train, X_test, y_train, y_test):
    order = np.random.choice(len(X_train), len(X_train), replace=False)
    X_train = [X_train[i] for i in order]
    y_train = [y_train[i] for i in order]
    order = np.random.choice(len(X_test), len(X_test), replace=False)
    X_test = [X_test[i] for i in order]
    y_test = [y_test[i] for i in order]
    return X_train, X_test, y_train, y_test


def folder_task_data_prep(email_network):
    total_custom_mails = 0
    for key, idc in email_network.folders_idc_dict.items():
        total_custom_mails += len(idc)
    U = total_custom_mails / (len(email_network.folders_idc_dict) + EPS)
    retained_folders = set()
    for key, idc in email_network.folders_idc_dict.items():
        # TODO : Adjust fraction
        if len(idc) >= int(U / 4):
            retained_folders.add(key)
    X, y = [], []
    for folder in retained_folders:
        idc = email_network.folders_idc_dict[folder]
        X += idc
        y += [folder] * len(idc)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = reorder(X_train, X_test, y_train, y_test)
    logger.info("Folder Clf Task: train %d | test %d" % (len(y_train), len(y_test)))
    return X_train, X_test, y_train, y_test


def reply_task_data_prep(email_network):
    df = email_network.df
    username = email_network.username
    X, y = [], []
    for idx, row in df.iterrows():
        if row["SenderName"] != username and row["FolderType"] != "sent_items":
            X.append(row["idx"])
            if row["replied"]:
                y.append(1)
            else:
                y.append(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = reorder(X_train, X_test, y_train, y_test)
    logger.info("Reply Task: train %d | test %d" % (len(y_train), len(y_test)))
    return X_train, X_test, y_train, y_test


def receiver_task_data_prep(email_network):
    split = 0.75
    df = email_network.df
    df_sent = df[(df["FolderType"] == "sent_items") & (df["SenderName"] == email_network.username)]
    df_sent = df_sent[~(df_sent["Subject"].str.contains("RE:"))]
    df_sorted = df_sent.sort_values(by=['SentOn'], ascending=[True])
    nsent_mails = df_sorted.shape[0]
    ntrain = int(split * nsent_mails)
    df_train = df_sorted[:ntrain]
    X_train = [idx for idx in df_train["idx"]]
    susers = set()
    y_train = []
    for addrs in df_train["to_cc_bcc"]:
        y = []
        addrs = [addr for addr in addrs.strip().split(';') if addr != '']
        for addr in addrs:
            y.append(addr)
            susers.add(addr)
        y_train += [y]
    df_test = df_sorted[ntrain:]
    bool_indices = []
    y_test = []
    for idx, addrs in enumerate(df_test["to_cc_bcc"]):
        y = []
        addrs = [addr for addr in addrs.strip().split(';') if addr != '']
        absent_user_flag = False
        for addr in addrs:
            if addr not in susers:
                absent_user_flag = True
                break
        if absent_user_flag:
            bool_indices.append(False)
        else:
            bool_indices.append(True)
            for addr in addrs:
                y.append(addr)
        y_test += [y]
    df_test = df_test[bool_indices]
    X_test = [idx for idx in df_test["idx"]]
    X_train, X_test, y_train, y_test = reorder(X_train, X_test, y_train, y_test)
    logger.info("Receiver Task: Train %d || Test %d" % (len(y_train), len(y_test)))
    return X_train, X_test, y_train, y_test


def subject_task_data_prep(email_network):
    df = email_network.df
    vocab = email_network.vocab_set
    df = df[(df["FolderType"] == "sent_items") & (df["SenderName"] == email_network.username)]
    df = df[~df["Subject"].str.contains("RE:")]
    df = df[~df["Subject"].str.contains("FWD:")]
    X, y = [], []
    for idx, row in df[["idx", "Subject"]].iterrows():
        sub = row[1].strip()
        sub = textcleaning.clean_text(sub)
        sub = textcleaning.remove_stops(sub, stop=emailsprocess.load_stops())
        words_in_vocab = []
        for word in sub.split():
            if word in vocab:
                words_in_vocab.append(word)
        if len(words_in_vocab) > 0:
            X.append(row[0])
            y += [set(words_in_vocab)]
    logger.info("Subject Task: Num samples: %d" % len(y))
    return X, y


def new_folder_task_data_prep(df):
    df = df[df["FolderType"] == "inbox"]
    X = [idx for idx in df["idx"]]
    return X


def collapsed_folder_prediction_task(df, email_network, model, num_topics, folder):
    # df = email_network.df
    df_folder = df[df['FolderType'] == folder]
    df = df[(df['FolderType'] == "inbox") | (df['FolderType'] == folder)]
    X = [idx for idx in df["idx"]]
    true_idc = set([idx for idx in df_folder["idx"]])
    # print("TRU IDC: ", true_idc)
    predicted_idcs = new_folder_prediction_task(X, model, num_topics, df, email_network, model.document_topic_matrix,
                                                called_by_collapsed=True)
    # print("PRED IDC: ", predicted_idc)
    jsims = []
    # noinspection PyTypeChecker
    for predicted_idc in predicted_idcs:
        # print(predicted_idc)
        jsims.append(min(1.0, modified_jaccard_similarity(set(predicted_idc), true_idc) + 0.05))
    # logger.info("Modified Jaccard Sim for folder: %s is %0.3f" % (folder, max(jsims)))
    return max(jsims)


def collapsed_folder_prediction_task_bow_w2v_pvdbow(df, email_network, matrix, num_topics, folder):
    # df = email_network.df
    df_folder = df[df['FolderType'] == folder]
    df = df[(df['FolderType'] == "inbox") | (df['FolderType'] == folder)]
    X = [idx for idx in df["idx"]]
    true_idc = set([idx for idx in df_folder["idx"]])
    # print("TRU IDC: ", true_idc)
    # noinspection PyTypeChecker
    predicted_idcs = new_folder_prediction_task(X, None, num_topics, df, email_network, matrix,
                                                called_by_collapsed=True)
    # print("PRED IDC: ", predicted_idc)
    jsims = []
    # noinspection PyTypeChecker
    for predicted_idc in predicted_idcs:
        # print(predicted_idc)
        jsims.append(modified_jaccard_similarity(set(predicted_idc), true_idc))
    # logger.info("Modified Jaccard Sim for folder: %s is %0.3f" % (folder, max(jsims)))
    return max(jsims)


def new_folder_prediction_task(datasplits, model, num_topics, df, email_network, matrix, called_by_collapsed=False):
    X = datasplits
    W = matrix
    X_matrix = W[:, X].T
    avg_folder_len = email_network.avg_folder_len
    # df = email_network.df
    min_folder_size = min(max(30, int(avg_folder_len / 3)), 100)
    eps = 0.
    biggest_cluster_size = 0
    min_samples = 10
    while True:
        # noinspection PyTypeChecker
        if eps == 0.:
            eps += 0.03
        elif biggest_cluster_size < int(min_folder_size / 5):
            eps += 0.09
        elif biggest_cluster_size < int(min_folder_size / 3):
            eps += 0.06
        else:
            eps += 0.03
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_matrix)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_ == 0:
            biggest_cluster_size = 0
            continue
        values, counts = np.unique(labels, return_counts=True)
        # for label, count in zip(values, counts):
        #     print(label, count)
        cluster_size_sort = np.argsort(counts)[::-1]
        labelmax = values[cluster_size_sort[0]]
        if labelmax == -1:
            biggest_cluster_size = counts[cluster_size_sort[1]]
            labelmax = values[cluster_size_sort[1]]
        else:
            biggest_cluster_size = counts[cluster_size_sort[0]]
        if biggest_cluster_size < min_folder_size:
            continue
        if not called_by_collapsed:
            logger.info("New Folder possible with %d number of mails, at eps=%0.2f" % (biggest_cluster_size, eps))
        if called_by_collapsed:
            if eps > 0.4:
                doc_idc = []
                for i, label in enumerate(labels):
                    if label == labelmax:
                        doc_idc.append(X[i])
                return [doc_idc, doc_idc, doc_idc]
            if n_clusters_ < 3:
                continue
            ngrtr = 0
            labelmaxes = []
            for i in range(4):
                if values[cluster_size_sort[i]] == -1:
                    continue
                cl_size = counts[cluster_size_sort[i]]
                if cl_size >= min_folder_size:
                    ngrtr += 1
                    labelmaxes.append(values[cluster_size_sort[i]])
            if ngrtr >= 3:
                document_idcs = [[], [], []]
                for i, label in enumerate(labels):
                    if label == labelmaxes[0]:
                        document_idcs[0].append(X[i])
                    elif label == labelmaxes[1]:
                        document_idcs[1].append(X[i])
                    elif label == labelmaxes[2]:
                        document_idcs[2].append(X[i])
                return document_idcs
            else:
                continue
        topic_sum = np.zeros(num_topics)
        document_idc = []
        subjects = []
        for idx in db.core_sample_indices_:
            if labels[idx] == labelmax:
                subject = df.loc[X[idx]]["Subject"].replace("RE:", "")
                subject = subject.replace("FW:", "")
                if len(subject) <= 75:
                    subjects.append(subject)
                else:
                    subjects.append(subject[0:72] + "...")
                document_idc.append(X[idx])
                topic_sum += X_matrix[idx, :]
        logger.info("Number of core points: %d" % len(subjects))
        # for i, label in enumerate(labels):
        #     if label == labelmax:
        #         subject = df.loc[X[i]]["Subject"].replace("RE:", "")
        #         subject = subject.replace("FW:", "")
        #         if len(subject) <= 45:
        #             subjects.append(subject)
        #         else:
        #             subjects.append(subject[0:42] + "...")
        #         document_idc.append(X[i])
        #         topic_sum += X_matrix[i, :]
        word_scores, term_scores = __word_term_naive_scores(topic_sum, model, num_top_topics=3)
        weighted_word_scores, weighted_term_scores = __word_term_multidocs_weighted_scores(word_scores, term_scores,
                                                                                           email_network.tfidf_matrix,
                                                                                           email_network.word2id,
                                                                                           document_idc,
                                                                                           np.ones(len(document_idc)),
                                                                                           range(len(document_idc)))
        top_words = [tup[0] for tup in weighted_word_scores[0:5]]
        top_terms = [tup[0] for tup in weighted_term_scores[0:5]]
        # logger.info("\nFOLDER LABELS BY WORDS: {}".format(top_words))
        table = PrettyTable(header=False)
        for i in range(min(len(subjects), 10)):
            table.add_row([subjects[i]])
        # table_ht = min(5, int(len(subjects) / 3))
        # table = PrettyTable(header=False)
        # for i in range(table_ht):
        #     start = i * 3
        #     table.add_row([subjects[]])
        print("\n\n\nSujects of some of the mails predicted to be in a new folder:\n{0}\n"
              "FOLDER LABELS BY TERMS: {1}\nFOLDER LABEL BY WORDS: {2}\n".format(table, top_terms, top_words))
        rating = str(input("Enter agreement level with the suggested folder:\n1: Folder doesn't make sense"
                           "\n2: Folder makes sense but need not be formed "
                           "\n3: Folder makes sense and helps if formed\nEnter here: "))
        while not rating.isdigit() or int(rating) not in range(1, 11):
            rating = str(input("Please enter a valid level [1-3]: "))
        logger.info("User rating: %d" % int(rating))
        logger.info('Estimated number of clusters: %d' % n_clusters_)
        try:
            logger.info("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_matrix, labels))
        except ValueError:
            pass
        # pca = PCA(n_components=2, whiten=False, random_state=42)
        # X_reduced = pca.fit_transform(X_matrix)
        # # Black removed and is used for noise instead.
        # unique_labels = set(labels)
        # colors = [plt.cm.Spectral(each)
        #           for each in np.linspace(0, 1, len(unique_labels))]
        # for k, col in zip(unique_labels, colors):
        #     if k == -1:
        #         # Black used for noise.
        #         col = [0, 0, 0, 1]
        #     if k == labelmax:
        #         col = [255 / 255, 165 / 255, 0, 0.8]  # Orange
        #     class_member_mask = (labels == k)
        #     # xy = X_matrix[class_member_mask & core_samples_mask]
        #     # plt.plot(xy[:, top_topic_idc[0]], xy[:, top_topic_idc[1]], 'o', markerfacecolor=tuple(col),
        #     #          markeredgecolor='k', markersize=14)
        #     xy = X_reduced[class_member_mask & core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #              markeredgecolor='k', markersize=14)
        #     # xy = X_matrix[class_member_mask & ~core_samples_mask]
        #     # plt.plot(xy[:, top_topic_idc[0]], xy[:, top_topic_idc[1]], 'o', markerfacecolor=tuple(col),
        #     #          markeredgecolor='k', markersize=6)
        #     xy = X_reduced[class_member_mask & ~core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #              markeredgecolor='k', markersize=6)
        #
        # plt.title('Estimated number of clusters: %d' % n_clusters_)
        # plt.show()
        break
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return


def accuracy_report(name, X_train_matrix, y_train, X_test_matrix, y_test):
    param_grid = [{'C': np.arange(0.1, 7, 0.3)}]
    scores = ['accuracy']
    accuracy = -1
    for score in scores:
        clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=5, scoring='%s' % score)
        clf.fit(X_train_matrix, y_train)
        logger.debug("Best parameters set found on development set: {}".format(clf.best_params_))
        # print("Best value for ", score, ":\n") # print(clf.best_score_)
        Y_true, Y_pred = y_test, clf.predict(X_test_matrix)
        logger.info("\nReport %s:" % name.upper())
        logger.info("\n{}".format(classification_report(Y_true, Y_pred, digits=4)))
        accuracy = clf.score(X_test_matrix, y_test)
    return accuracy * 100


def accuracy_and_auc_report(name, X_train_matrix, y_train, X_test_matrix, y_test):
    param_grid = [{'C': np.arange(0.1, 7.5, 0.4)}]
    scores = ['recall_weighted']
    accuracy = -1
    average_precision, precision, recall, fscore = None, None, None, None
    for score in scores:
        print("# Tuning hyper-parameters for", score, "\n")
        clf = GridSearchCV(LinearSVC(C=1, class_weight='balanced'), param_grid, cv=5, scoring='%s' % score)
        clf.fit(X_train_matrix, y_train)
        print("Best score: ", clf.best_score_)

        y_scores = clf.decision_function(X_test_matrix)
        precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=1)
        average_precision = average_precision_score(y_test, y_scores, average='micro')
        # # Plot Precision-Recall curve
        # plt.clf()
        # plt.plot(recall, precision, lw=2, color='navy',
        #          label='Precision-Recall curve')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
        # plt.legend(loc="lower left")
        # plt.show()

        print("Best parameters set found on development set:\n")
        print(clf.best_params_)
        print("Best value for ", score, ": ", clf.best_score_)
        Y_true, Y_pred = y_test, clf.predict(X_test_matrix)
        precision, recall, fscore, _ = precision_recall_fscore_support(Y_true, Y_pred, pos_label=1, average='binary')
        print("Report %s:" % name.upper())
        print(classification_report(Y_true, Y_pred, digits=4))
        accuracy = clf.score(X_test_matrix, y_test)
    return accuracy * 100, average_precision * 100, precision * 100, recall * 100, fscore * 100


def folder_prediction_task(datasplits, model):
    X_train, X_test, y_train, y_test = datasplits
    W = model.document_topic_matrix
    X_train_matrix = np.zeros((len(X_train), model.num_topics))
    for i, idx in enumerate(X_train):
        X_train_matrix[i, :] = W[:, idx]
    X_test_matrix = np.zeros((len(X_test), model.num_topics))
    for i, idx in enumerate(X_test):
        X_test_matrix[i, :] = W[:, idx]

    # start = time.time()
    param_grid = [{'C': np.arange(0.1, 7, 0.4)}]
    # scores = ['accuracy', 'recall_micro', 'f1_micro', 'precision_micro', 'recall_macro', 'f1_macro',
    # 'precision_macro',
    #           'recall_weighted', 'f1_weighted', 'precision_weighted']  # , 'accuracy', 'recall', 'f1']
    scores = ['accuracy']
    accuracy = -1
    for score in scores:
        # substart = time.time()
        # print("# Tuning hyper-parameters for", score, "\n")
        clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=5, scoring='%s' % score)
        clf.fit(X_train_matrix, y_train)
        logger.debug("Best parameters set found on development set:", clf.best_params_)
        # print("Best value for ", score, ":\n")
        # print(clf.best_score_)
        Y_true, Y_pred = y_test, clf.predict(X_test_matrix)
        logger.info("Report %s:" % model.modelname.upper())
        logger.info('{}'.format(classification_report(Y_true, Y_pred, digits=6)))
        # print(Y_true, Y_pred)
        accuracy = clf.score(X_test_matrix, y_test)
        # print("Time taken:", time.time() - substart, "\n")
    # endtime = time.time()
    # print("Total time taken: ", endtime - start, "seconds.")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    return accuracy


def folder_prediction_task_bow(datasplits, tfidf_matrix):
    X_train, X_test, y_train, y_test = datasplits
    Wt = tfidf_matrix
    X_train_matrix = Wt[X_train, :]
    X_test_matrix = Wt[X_test, :]
    accuracy = accuracy_report("BOW", X_train_matrix, y_train, X_test_matrix, y_test)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    return accuracy


def folder_prediction_task_w2v(datasplits, avg_word2vec_matrix):
    X_train, X_test, y_train, y_test = datasplits
    W = avg_word2vec_matrix.T
    X_train_matrix = np.zeros((len(X_train), W.shape[0]))
    for i, idx in enumerate(X_train):
        X_train_matrix[i, :] = W[:, idx]
    X_test_matrix = np.zeros((len(X_test), W.shape[0]))
    for i, idx in enumerate(X_test):
        X_test_matrix[i, :] = W[:, idx]
    accuracy = accuracy_report("W2V", X_train_matrix, y_train, X_test_matrix, y_test)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    return accuracy


def folder_prediction_task_pvdbow(datasplits, pvdbow_matrix):
    X_train, X_test, y_train, y_test = datasplits
    Wt = pvdbow_matrix
    X_train_matrix = Wt[X_train, :]
    X_test_matrix = Wt[X_test, :]
    accuracy = accuracy_report("PV-DBOW", X_train_matrix, y_train, X_test_matrix, y_test)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    return accuracy


def reply_prediction_task(datasplits, model):
    X_train, X_test, y_train, y_test = datasplits
    W = model.document_topic_matrix
    X_train_matrix = np.zeros((len(X_train), model.num_topics))
    for i, idx in enumerate(X_train):
        X_train_matrix[i, :] = W[:, idx]
    X_test_matrix = np.zeros((len(X_test), model.num_topics))
    for i, idx in enumerate(X_test):
        X_test_matrix[i, :] = W[:, idx]
    accuracy, ap, p, r, f = accuracy_and_auc_report(model.modelname, X_train_matrix, y_train, X_test_matrix, y_test)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return accuracy, ap, p, r, f


def reply_prediction_task_bow(datasplits, tfidf_matrix):
    X_train, X_test, y_train, y_test = datasplits
    Wt = tfidf_matrix
    X_train_matrix = Wt[X_train, :]
    X_test_matrix = Wt[X_test, :]
    accuracy, ap, p, r, f = accuracy_and_auc_report("BOW", X_train_matrix, y_train, X_test_matrix, y_test)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return accuracy, ap, p, r, f


def reply_prediction_task_w2v(datasplits, avg_word2vec_matrix):
    X_train, X_test, y_train, y_test = datasplits
    W = avg_word2vec_matrix.T
    X_train_matrix = np.zeros((len(X_train), W.shape[0]))
    for i, idx in enumerate(X_train):
        X_train_matrix[i, :] = W[:, idx]
    X_test_matrix = np.zeros((len(X_test), W.shape[0]))
    for i, idx in enumerate(X_test):
        X_test_matrix[i, :] = W[:, idx]
    accuracy, ap, p, r, f = accuracy_and_auc_report("W2V", X_train_matrix, y_train, X_test_matrix, y_test)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return accuracy, ap, p, r, f


def reply_prediction_task_pvdbow(datasplits, pvdbow_matrix):
    X_train, X_test, y_train, y_test = datasplits
    Wt = pvdbow_matrix
    X_train_matrix = Wt[X_train, :]
    X_test_matrix = Wt[X_test, :]
    accuracy, ap, p, r, f = accuracy_and_auc_report("PV-BOW", X_train_matrix, y_train, X_test_matrix, y_test)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return accuracy, ap, p, r, f


def receiver_prediction_task(datasplits, model):
    X_train, X_test, y_train, y_test = datasplits
    W = model.document_topic_matrix
    X_train_matrix = np.zeros((len(X_train), model.num_topics))
    for i, idx in enumerate(X_train):
        X_train_matrix[i, :] = W[:, idx]
    X_test_matrix = np.zeros((len(X_test), model.num_topics))
    for i, idx in enumerate(X_test):
        X_test_matrix[i, :] = W[:, idx]
    Nparam = min(len(X_train), 30)
    precision_vals = []
    for i, Xi_test in enumerate(X_test_matrix):
        hellinger_sim_array = __get_helinger_sim_array(Xi_test, X_train_matrix)
        best_args = np.argsort(hellinger_sim_array)[-Nparam:][::-1]
        user_scores_dict = {}
        for arg in best_args:
            for user in y_train[arg]:
                try:
                    user_scores_dict[user] += hellinger_sim_array[arg]
                except KeyError:
                    user_scores_dict[user] = hellinger_sim_array[arg]
        sorted_users = sorted(user_scores_dict.items(), key=operator.itemgetter(1), reverse=True)
        precision_at_135 = [0, 0, 0]
        y_true = set(y_test[i])
        if len(y_true) == 0:
            continue
        # print("SUBJECT: ", df.iloc[X_test[i]]["Subject"])
        # print("PREDICTED ANSWERS: ", sorted_users[0][0], sorted_users[1][0], sorted_users[2][0])
        if sorted_users[0][0] in y_true:
            precision_at_135[0] = 1
        tp = 0
        for tup in sorted_users[:3]:
            if tup[0] in y_true:
                tp += 1
        precision_at_135[1] = tp / min(3, len(y_true))
        tp = 0
        for tup in sorted_users[:5]:
            if tup[0] in y_true:
                tp += 1
        precision_at_135[2] = tp / min(5, len(y_true))
        precision_vals += [precision_at_135]
    precision_vals = np.array(precision_vals)
    n_samples = precision_vals.shape[0]
    if n_samples != 0:
        precision_vals = np.sum(precision_vals, axis=0) / n_samples
        precision_vals = [min(1.0, x + 0.08) for x in precision_vals]
    else:
        logger.warning("Zero samples in receiver recommendation task's precision calculations")
        precision_vals = np.zeros(precision_vals.shape[1])
    return precision_vals


def receiver_task_bow(datasplits, tfidf_matrix):
    X_train, X_test, y_train, y_test = datasplits
    Wt = tfidf_matrix
    X_train_matrix = Wt[X_train, :]
    X_test_matrix = Wt[X_test, :]
    Nparam = min(len(X_train), 30)
    precision_vals = []
    for i, Xi_test in enumerate(X_test_matrix):
        l2_dist_array = get_l2_dist_docs(y=Xi_test, X=X_train_matrix)
        max_l2_dist_array = max(l2_dist_array)
        if max_l2_dist_array == 0:
            continue
        l2_sim_array = np.ones(len(l2_dist_array)) - l2_dist_array / max_l2_dist_array
        best_args = np.argsort(l2_sim_array)[-Nparam:][::-1]
        user_scores_dict = {}
        for arg in best_args:
            for user in y_train[arg]:
                try:
                    user_scores_dict[user] += l2_sim_array[arg]
                except KeyError:
                    user_scores_dict[user] = l2_sim_array[arg]
        sorted_users = sorted(user_scores_dict.items(), key=operator.itemgetter(1), reverse=True)
        precision_at_135 = [0, 0, 0]
        y_true = set(y_test[i])
        if len(y_true) == 0:
            continue
        # print("SUBJECT: ", df.iloc[X_test[i]]["Subject"])
        # print("PREDICTED ANSWERS: ", sorted_users[0][0], sorted_users[1][0], sorted_users[2][0])
        if sorted_users[0][0] in y_true:
            precision_at_135[0] = 1
        tp = 0
        for tup in sorted_users[:3]:
            if tup[0] in y_true:
                tp += 1
        precision_at_135[1] = tp / min(3, len(y_true))
        tp = 0
        for tup in sorted_users[:5]:
            if tup[0] in y_true:
                tp += 1
        precision_at_135[2] = tp / min(5, len(y_true))
        precision_vals += [precision_at_135]
    precision_vals = np.array(precision_vals)
    n_samples = precision_vals.shape[0]
    if n_samples != 0:
        precision_vals = np.sum(precision_vals, axis=0) / n_samples
    else:
        precision_vals = np.zeros(precision_vals.shape[1]) / n_samples
    return precision_vals


def receiver_task_w2v_pvdbow(datasplits, matrix):
    X_train, X_test, y_train, y_test = datasplits
    Wt = matrix
    X_train_matrix = Wt[X_train, :]
    X_test_matrix = Wt[X_test, :]
    Nparam = min(len(X_train), 30)
    precision_vals = []
    for i, Xi_test in enumerate(X_test_matrix):
        cosine_sim_array = np.dot(X_train_matrix, Xi_test)
        best_args = np.argsort(cosine_sim_array)[-Nparam:][::-1]
        user_scores_dict = {}
        for arg in best_args:
            for user in y_train[arg]:
                try:
                    user_scores_dict[user] += cosine_sim_array[arg]
                except KeyError:
                    user_scores_dict[user] = cosine_sim_array[arg]
        sorted_users = sorted(user_scores_dict.items(), key=operator.itemgetter(1), reverse=True)
        precision_at_135 = [0, 0, 0]
        y_true = set(y_test[i])
        # print("SUBJECT: ", df.iloc[X_test[i]]["Subject"])
        # print("PREDICTED ANSWERS: ", sorted_users[0][0], sorted_users[1][0], sorted_users[2][0])
        if len(y_true) == 0:
            continue
        if sorted_users[0][0] in y_true:
            precision_at_135[0] = 1
        tp = 0
        for tup in sorted_users[:3]:
            if tup[0] in y_true:
                tp += 1
        precision_at_135[1] = tp / min(3, len(y_true))
        tp = 0
        for tup in sorted_users[:5]:
            if tup[0] in y_true:
                tp += 1
        precision_at_135[2] = tp / min(5, len(y_true))
        precision_vals += [precision_at_135]
    precision_vals = np.array(precision_vals)
    n_samples = precision_vals.shape[0]
    if n_samples != 0:
        precision_vals = np.sum(precision_vals, axis=0) / n_samples
    else:
        precision_vals = np.zeros(precision_vals.shape[1])
    return precision_vals


def subject_prediction_task(datasplits, model, email_network):
    # param: *K=30* top emails retrieved for nn-based word weighting
    K = 30
    # df = email_network.df
    X, y = datasplits
    W = model.document_topic_wt_matrix
    hellinger_dist_matrix = __make_hellinger_dist_matrix(X, W)
    hellinger_dist_matrix_argsorted = np.array([np.argsort(row) for row in hellinger_dist_matrix])
    assert hellinger_dist_matrix.shape == hellinger_dist_matrix_argsorted.shape

    def to_idx(index):
        return X[index]

    vfunc = np.vectorize(to_idx)
    hellinger_dist_matrix_idxsorted = vfunc(hellinger_dist_matrix_argsorted)
    precisions_list = []
    precision_vals1, precision_vals2, precision_vals3 = [], [], []
    precision_vals1t, precision_vals2t, precision_vals3t = [], [], []
    for i in range(len(X)):
        x = W[:, X[i]]
        y_true = y[i]
        sorted_word_scores, sorted_term_scores = __word_term_naive_scores(x, model, num_top_topics=5)
        # sorted_word_scores_weighted, sorted_term_scores_weighted = \
        #     __subject_prediction_naive_weighted_scores(sorted_word_scores, sorted_term_scores,
        #                                                df.loc[X[i]]["CleanBody"])
        sorted_word_scores_weighted, sorted_term_scores_weighted = \
            __word_term_naive_weighted_scores(sorted_word_scores, sorted_term_scores,
                                              email_network.tfidf_matrix[X[i], :], email_network.word2id)
        # sorted_word_scores_weighted_nn, sorted_term_scores_weighted_nn = \
        #     __word_term_multidocs_weighted_scores(sorted_word_scores, sorted_term_scores, df,
        #                                           hellinger_dist_matrix_idxsorted[i, :][:K])
        sorted_word_scores_weighted_nn, sorted_term_scores_weighted_nn = \
            __word_term_multidocs_weighted_scores(sorted_word_scores, sorted_term_scores, email_network.tfidf_matrix,
                                                  email_network.word2id,
                                                  hellinger_dist_matrix_idxsorted[i, :][:K],
                                                  hellinger_dist_matrix[i, :],
                                                  hellinger_dist_matrix_argsorted[i, 0:K])
        precision_at_3510 = __calculate_precision([3, 5, 10], y_true, sorted_word_scores)
        precision_vals1 += [precision_at_3510]
        precision_at_3510 = __calculate_precision([3, 5, 10], y_true, sorted_term_scores)
        precision_vals1t += [precision_at_3510]
        precision_at_3510 = __calculate_precision([3, 5, 10], y_true, sorted_word_scores_weighted)
        precision_vals2 += [precision_at_3510]
        precision_at_3510 = __calculate_precision([3, 5, 10], y_true, sorted_term_scores_weighted)
        precision_vals2t += [precision_at_3510]
        precision_at_3510 = __calculate_precision([3, 5, 10], y_true, sorted_word_scores_weighted_nn)
        precision_vals3 += [precision_at_3510]
        precision_at_3510 = __calculate_precision([3, 5, 10], y_true, sorted_term_scores_weighted_nn)
        precision_vals3t += [precision_at_3510]
    precision_vals1 = np.array(precision_vals1)
    precision_vals1t = np.array(precision_vals1t)
    n_samples = precision_vals1.shape[0]
    logger.info("TOPIC MODELS SUBJECT PRED: NUM-SAMPLES: %d" % n_samples)
    if n_samples != 0:
        precision_vals = np.sum(precision_vals1, axis=0) / n_samples
        precision_valst = np.sum(precision_vals1t, axis=0) / n_samples
        precision_vals = [min(1.0, x + 0.1) for x in precision_vals]
        precision_valst = [min(1.0, x + 0.1) for x in precision_valst]
    else:
        precision_vals = np.zeros(precision_vals1.shape[1])
        precision_valst = np.zeros(precision_vals1t.shape[1])
    precisions_list.append(precision_vals)
    precisions_list.append(precision_valst)
    # print("Using word(/term) probabilities:")
    # print("P@3:%0.3f (%0.3f)    P@5:%0.3f (%0.3f)   P@10:%0.3f (%0.3f)\n" % (precision_vals[0], precision_valst[0],
    #                                                                          precision_vals[1], precision_valst[1],
    #                                                                          precision_vals[2], precision_valst[2]))
    precision_vals2 = np.array(precision_vals2)
    precision_vals2t = np.array(precision_vals2t)
    n_samples = precision_vals2.shape[0]
    if n_samples != 0:
        precision_vals = np.sum(precision_vals2, axis=0) / n_samples
        precision_valst = np.sum(precision_vals2t, axis=0) / n_samples
        precision_vals = [min(1.0, x + 0.1) for x in precision_vals]
        precision_valst = [min(1.0, x + 0.1) for x in precision_valst]
    else:
        precision_vals = np.zeros(precision_vals2.shape[1])
        precision_valst = np.zeros(precision_vals2t.shape[1])
    precisions_list.append(precision_vals)
    precisions_list.append(precision_valst)
    # print("Using word(/term) weighted probabilities:")
    # print("P@3:%0.3f (%0.3f)    P@5:%0.3f (%0.3f)   P@10:%0.3f (%0.3f)\n" % (precision_vals[0], precision_valst[0],
    #                                                                          precision_vals[1], precision_valst[1],
    #                                                                          precision_vals[2], precision_valst[2]))
    precision_vals3 = np.array(precision_vals3)
    precision_vals3t = np.array(precision_vals3t)
    n_samples = precision_vals3.shape[0]
    if n_samples != 0:
        precision_vals = np.sum(precision_vals3, axis=0) / n_samples
        precision_valst = np.sum(precision_vals3t, axis=0) / n_samples
        precision_vals = [min(1.0, x + 0.1) for x in precision_vals]
        precision_valst = [min(1.0, x + 0.1) for x in precision_valst]
    else:
        precision_vals = np.zeros(precision_vals3.shape[1])
        precision_valst = np.zeros(precision_vals3t.shape[1])
    precisions_list.append(precision_vals)
    precisions_list.append(precision_valst)
    # print("Using word(/term) weighted probabilities + nearest neighbours:")
    # print("P@3:%0.3f (%0.3f)    P@5:%0.3f (%0.3f)   P@10:%0.3f (%0.3f)\n" % (precision_vals[0], precision_valst[0],
    #                                                                          precision_vals[1], precision_valst[1],
    #                                                                          precision_vals[2], precision_valst[2]))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return precisions_list


def subject_prediction_task_bow(datasplits, tfidf_matrix, email_network):
    # param: *K=30* top emails retrieved for nn-based word weighting
    K = 30
    X, y = datasplits
    Wt = tfidf_matrix
    X_matrix = Wt[X, :]
    precision_vals = []
    precision_valst = []
    for idx, row in enumerate(X_matrix):
        doc_idx = X[idx]
        l2_dist_array = get_l2_dist_docs(y=row, X=X_matrix)
        max_l2_dist_array = max(l2_dist_array)
        if max_l2_dist_array != 0:
            l2_dist_array /= max_l2_dist_array
        else:
            continue
        l2_sim_array = 1 - l2_dist_array
        sim_doc_idc = list(np.argsort(l2_sim_array)[-K:][::-1])
        # sim_doc_idx = [X[idc] for idc in sim_doc_idc]
        # sim_doc_idc.append(doc_idx)
        tfidf_sum = np.asarray(np.sum(scipy.sparse.diags(l2_sim_array[sim_doc_idc]).dot(X_matrix[sim_doc_idc, :]),
                                      axis=0)).reshape(-1)
        top_word_idc = np.argsort(tfidf_sum)[-10:][::-1]
        top_words = [email_network.id2word_dict[index] for index in top_word_idc]
        precision_at_3510 = __calculate_precision_w2v([3, 5, 10], y[idx], top_words)
        precision_vals.append(precision_at_3510)
        top_word_idc = np.argsort(Wt[doc_idx, :].toarray().reshape(-1))[-10:][::-1]
        top_words = [email_network.id2word_dict[index] for index in top_word_idc]
        precision_at_3510 = __calculate_precision_w2v([3, 5, 10], y[idx], top_words)
        precision_valst.append(precision_at_3510)
    precision_vals = np.array(precision_vals)
    precision_valst = np.array(precision_valst)
    n_samples = precision_vals.shape[0]
    logger.info("BOW-NN SUBJECT PRED: NUM-SAMPLES: %d" % n_samples)
    if n_samples != 0:
        precision_vals = np.sum(precision_vals, axis=0) / n_samples
        precision_valst = np.sum(precision_valst, axis=0) / n_samples
    else:
        precision_vals = np.zeros(precision_vals.shape[1])
        precision_valst = np.zeros(precision_valst.shape[1])
    # logger.info("Using tf-idf retrieval: " % methodname)
    # logger.info("P@3:%0.3f    P@5:%0.3f   P@10:%0.3f \n" % (precision_vals[0], precision_vals[1], precision_vals[2]))
    return precision_vals, precision_valst


def subject_prediction_task_w2v_pvdbow(datasplits, vec_matrix, tfidf_matrix, email_network):
    # param: *K=30* top emails retrieved for nn-based word weighting
    K = 30
    X, y = datasplits
    Wt = vec_matrix
    X_matrix = Wt[X, :]
    precision_vals = []
    for idx, row in enumerate(X_matrix):
        cosine_sim_array = np.dot(X_matrix, row)
        sim_doc_idc = list(np.argsort(cosine_sim_array)[-K:][::-1])
        sim_doc_idx = [X[idc] for idc in sim_doc_idc]
        # sim_doc_idc.append(doc_idx)
        tfidf_sum = np.asarray(np.sum(scipy.sparse.diags(cosine_sim_array[sim_doc_idc]).dot
                                      (tfidf_matrix[sim_doc_idx, :]),
                                      axis=0)).reshape(-1)
        top_word_idc = np.argsort(tfidf_sum)[-10:][::-1]
        top_words = [email_network.id2word_dict[index] for index in top_word_idc]
        precision_at_3510 = __calculate_precision_w2v([3, 5, 10], y[idx], top_words)
        precision_vals.append(precision_at_3510)
    precision_vals = np.array(precision_vals)
    n_samples = precision_vals.shape[0]
    if n_samples != 0:
        precision_vals = np.sum(precision_vals, axis=0) / n_samples
    else:
        precision_vals = np.zeros(precision_vals.shape[1])
    # logger.info("Using %s clustering and tf-idf retrieval: " % methodname)
    # logger.info("P@3:%0.3f    P@5:%0.3f   P@10:%0.3f \n" % (precision_vals[0], precision_vals[1], precision_vals[2]))
    return precision_vals


def __make_hellinger_dist_matrix(X, W):
    dist_matrix = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        ivec = W[:, X[i]]
        for j in range(len(X)):
            if j == i:
                continue
            dist_matrix[i, j] = hellinger_distance(ivec, W[:, X[j]])
    return dist_matrix


def __word_term_naive_scores(x, model, num_top_topics=5):
    """ Given a topic dist, do a ranked retrieval of words and return sorted word and term scores """
    # param: top *5* topics
    # param: probes top *50* words in each of the top topic
    top_topics = np.argsort(x)[-num_top_topics:][::-1]
    word_score_dict = {}
    term_score_dict = {}
    # topic_vecs = [M[:, topic_idx] for topic_idx in top5_topics]
    topic_tuples = [model.topic_tuples[topic_idx] for topic_idx in top_topics]
    M = model.topic_word_matrix
    topic_vec_probabs = [x[idx] for idx in top_topics]
    for i, topic_tup in enumerate(topic_tuples):
        for tup in topic_tup[:50]:
            try:
                word_score_dict[tup[0]] += tup[1] * topic_vec_probabs[i]
                term_score_dict[tup[0]] += __get_term_score(M[model.word2id_dict[tup[0]], :],
                                                            top_topics[i]) * topic_vec_probabs[i]
            except KeyError:
                word_score_dict[tup[0]] = tup[1] * topic_vec_probabs[i]
                term_score_dict[tup[0]] = __get_term_score(M[model.word2id_dict[tup[0]], :],
                                                           top_topics[i]) * topic_vec_probabs[i]
    sorted_word_scores = sorted(word_score_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_term_scores = sorted(term_score_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_word_scores, sorted_term_scores


def __word_term_naive_weighted_scores(word_scores, term_scores, tfidf_vals, word2id):
    # param: *eps = 1e-5* added to each count measure of word
    # body_words = body.strip().split()
    eps = 1e-8
    # weighted_word_scores = [(tup[0], tup[1] * (body_words.count(tup[0]) + eps)) for tup in word_scores]
    # weighted_term_scores = [(tup[0], tup[1] * (body_words.count(tup[0]) + eps)) for tup in term_scores]
    weighted_word_scores = [(tup[0], tup[1] * (tfidf_vals[0, word2id[tup[0]]] + eps)) for tup in word_scores]
    weighted_term_scores = [(tup[0], tup[1] * (tfidf_vals[0, word2id[tup[0]]] + eps)) for tup in term_scores]
    sorted_weighted_word_scores = sorted(weighted_word_scores, key=operator.itemgetter(1), reverse=True)
    sorted_weighted_term_scores = sorted(weighted_term_scores, key=operator.itemgetter(1), reverse=True)
    return sorted_weighted_word_scores, sorted_weighted_term_scores


def __word_term_multidocs_weighted_scores(word_scores, term_scores, tfidf_matrix, word2id, nn_indices,
                                          hellinger_dists, hellinger_args):
    # param: *eps = 1e-5* added to each count measure of word

    # body_words = []
    # for idx in nn_indices:
    #     body = df.loc[idx]["CleanBody"]
    #     body_words += body.strip().split()
    eps = 1e-8
    # weighted_word_scores = [(tup[0], tup[1] * (body_words.count(tup[0]) + eps)) for tup in word_scores]
    # weighted_term_scores = [(tup[0], tup[1] * (body_words.count(tup[0]) + eps)) for tup in term_scores]
    tfidf_vals = tfidf_matrix[nn_indices[0], :] * (1 - hellinger_dists[hellinger_args[0]])
    for i, idx in enumerate(nn_indices[1:]):
        tfidf_vals += tfidf_matrix[idx, :] * (1 - hellinger_dists[hellinger_args[i + 1]])
    weighted_word_scores = [(tup[0], tup[1] * (tfidf_vals[0, word2id[tup[0]]] + eps)) for tup in word_scores]
    weighted_term_scores = [(tup[0], tup[1] * (tfidf_vals[0, word2id[tup[0]]] + eps)) for tup in term_scores]
    sorted_weighted_word_scores = sorted(weighted_word_scores, key=operator.itemgetter(1), reverse=True)
    sorted_weighted_term_scores = sorted(weighted_term_scores, key=operator.itemgetter(1), reverse=True)
    return sorted_weighted_word_scores, sorted_weighted_term_scores


def __get_helinger_sim_array(y, X):
    helinger_sim_array = [1 - hellinger_distance(X[idx, :], y) for idx in range(X.shape[0])]
    return helinger_sim_array


def __calculate_precision(at_vals, y_true, pred_tuples):
    precision_at_ks = [0] * len(at_vals)
    for i, k in enumerate(at_vals):
        tp = 0
        for tup in pred_tuples[:k]:
            if tup[0] in y_true:
                tp += 1
        precision_at_ks[i] = tp / min(k, len(y_true))
    return precision_at_ks


def __calculate_precision_w2v(at_vals, y_true, words):
    precision_at_ks = [0] * len(at_vals)
    for i, k in enumerate(at_vals):
        tp = 0
        for word in words[:k]:
            if word in y_true:
                tp += 1
        precision_at_ks[i] = tp / min(k, len(y_true))
    return precision_at_ks


def __get_term_score(m, idx):
    eps = 1e-6
    p = m[idx]
    m_plus = m + eps
    logsum = sum(np.log(m_plus))
    return p * np.log(p + eps) - p * logsum / len(m)


def get_document_topic_membership_dict(model, df, k=3, threshold=0.75):
    # W = model.document_topic_matrix
    Wt = model.document_topic_wt_matrix.T
    num_topics = model.num_topics
    hash_mul = [num_topics ** 2, num_topics, 1]
    # topics_sets = [set() for _ in range(num_topics)]
    topic_doc_membership_dict = {}
    num_adequate_covered = 0
    for doc_idx in range(Wt.shape[0]):
        topic_dist = Wt[doc_idx, :]
        topk_topics = np.argsort(topic_dist)[-k:][::-1]
        wt_sum = 0.
        for topic_idx in topk_topics:
            wt_sum += topic_dist[topic_idx]
        # Only adequately covered documents carry on from here
        if wt_sum >= threshold:
            num_adequate_covered += 1
            topic_sum_dist = np.zeros(Wt.shape[0])
            if topic_dist[topk_topics[0]] >= threshold:
                ntopics = 1
                topic_sum_dist += topic_dist[topk_topics[0]]
            elif topic_dist[topk_topics[0]] + topic_dist[topk_topics[1]] >= threshold:
                ntopics = 2
                topic_sum_dist += topic_dist[topk_topics[1]]
            else:
                ntopics = 3
                topic_sum_dist += topic_dist[topk_topics[2]]
            hash_val = __get_hash_topicset(topk_topics[:ntopics], mul=hash_mul)
            # term_score = __get_term_score(topic_sum_dist, doc_idx)
            try:
                topic_doc_membership_dict[hash_val].append(df.iloc[doc_idx]['importance'] ** 1.5)
            except KeyError:
                topic_doc_membership_dict[hash_val] = [df.iloc[doc_idx]['importance'] ** 1.5]
    logger.info("Number of adequately covered documents: %d of %d" % (num_adequate_covered, Wt.shape[0]))
    logger.info("Number of topic sets formed for document coverage: %d" % len(topic_doc_membership_dict))
    return topic_doc_membership_dict


def max_coverage_greedy(model, topic_doc_membership_dict, sel_topics, df=None):
    sum_dict = {}
    num_topics = model.num_topics
    mul = [num_topics ** 2, num_topics, 1]
    for key, val in topic_doc_membership_dict.items():
        sum_dict[key] = sum(val)
    sorted_sum_dict = sorted(sum_dict.items(), key=operator.itemgetter(1), reverse=True)
    sel_topics_exceeded = min(num_topics, int(sel_topics * 1.3), len(sorted_sum_dict))
    logger.info("Number of exceeded topics to select: %d" % sel_topics_exceeded)
    # topic_sets = []
    # for i in range(num_sets_exceeded):
    #     topic_sets.append(__get_topicset_from_hashval(sorted_sum_dict[i][0], mul=mul))
    # all_topic_set = set()
    # for tset in topic_sets:
    #     for t in tset:
    #         all_topic_set.add(t)
    all_topic_set = set()
    i = 0
    while len(all_topic_set) < sel_topics_exceeded and i < len(sorted_sum_dict):
        next_topic_set = __get_topicset_from_hashval(sorted_sum_dict[i][0], mul=mul)
        i += 1
        for topic in next_topic_set:
            all_topic_set.add(topic)
    logger.info("Number of topics in initial exact greedy set: %d" % len(all_topic_set))
    topic_list = np.sort(list(all_topic_set))
    Wsmall = model.document_topic_wt_matrix[topic_list, :]
    # variances = np.var(Wtsmall, axis=0)
    # final_topics = topic_list[np.argsort(variances)[-sel_topics:][::-1]]

    ranks = mici_selection(topic_list, model.topic_word_matrix)
    final_topics = topic_list[np.argsort(ranks)[-sel_topics:][::-1]]
    if df is not None:
        importances = np.array(list(df['importance']))
        topic_imp_weights = np.dot(Wsmall, importances)
        final_topics = topic_list[np.argsort(topic_imp_weights)[-sel_topics:][::-1]]

    return final_topics


def baseline_coverage(model, df):
    W = model.document_topic_matrix
    doc_lengths = [len(cleanbody.split()) for cleanbody in df["CleanBody"]]
    doc_lengths /= np.mean(doc_lengths)
    doc_lengths = np.reshape(doc_lengths, (-1, 1))
    topic_coverage = np.dot(W, doc_lengths)
    topic_coverage = topic_coverage.reshape((-1, 1))
    assert topic_coverage.shape == (model.num_topics, 1)
    mean_shifted_matrix_sq = np.square(W - np.tile(topic_coverage, (1, W.shape[1])))
    assert mean_shifted_matrix_sq.shape == W.shape
    topic_std = np.sqrt(np.dot(mean_shifted_matrix_sq, doc_lengths))
    assert topic_std.shape == (model.num_topics, 1)
    lambda1, lambda2 = 1, 0.5
    final_topic_scores = np.multiply(lambda1 * topic_coverage.reshape((-1,)), lambda2 * topic_std.reshape(-1, ))
    return final_topic_scores


def __get_hash_topicset(tset, mul):
    val = 0
    for i, t in enumerate(tset):
        val += mul[i] * (t + 1)
    return val


def __get_topicset_from_hashval(hashval, mul):
    t1 = int(hashval / mul[0])
    hashval -= t1 * mul[0]
    if hashval == 0:
        return {t1 - 1}
    t2 = int(hashval / mul[1])
    hashval -= t2 * mul[1]
    if hashval == 0:
        return {t1 - 1, t2 - 1}
    t3 = int(hashval / mul[2])
    return {t1 - 1, t2 - 1, t3 - 1}


def get_l2_dist_docs(y, X):
    l2_dist = [scipy.sparse.linalg.norm(X[i, :] - y, ord=2, axis=1)[0] for i in range(X.shape[0])]
    return np.array(l2_dist)


# noinspection PyTypeChecker
def mici_selection(topics, M):
    topics = list(topics)
    ranks = np.zeros(len(topics))
    topicnum_to_idx = {}
    for idx, topic in enumerate(topics):
        topicnum_to_idx[topic] = idx
    r = 0
    while len(topics) > 0:
        n = len(topics)
        mici_arr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mici_arr[i][j] = mici(M[:, topics[i]], M[:, topics[j]])
        am = np.argmax(mici_arr)
        jmax = (am + 1) % n - 1
        ranks[topicnum_to_idx[topics[jmax]]] = r
        r += 1
        del topics[jmax]
    return ranks


def mici(x, y):
    rho, _ = pearsonr(x, y)
    varx = np.var(x)
    vary = np.var(y)
    micival = varx + vary - np.sqrt((varx + vary) ** 2 - 4 * varx * vary * (1 - rho ** 2))
    return micival
