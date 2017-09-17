"""
This module explores the structure in user's inbox.
"""

from __future__ import division
from __future__ import print_function

from topiceval import makewordvecs

# import pandas
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import operator
import logging
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

logger = logging.getLogger(__name__)


class EmailUser(object):
    def __init__(self, name, nsent_to, nsent_cc, nrecvd_from):
        self.name = name
        self.nsent_to = nsent_to
        self.nsent_cc = nsent_cc
        self.recvd_from = nrecvd_from


class EmailNetwork(object):
    def __init__(self, df, id2word_dict, wordvecs):
        self.df = df
        self.username = self.__get_username()
        self.id2word_dict = id2word_dict
        vocabulary = list(self.id2word_dict.values())
        self.vocab_set = set(vocabulary)
        tfidfvectorizer = TfidfVectorizer(vocabulary=vocabulary)
        self.tfidf_matrix = tfidfvectorizer.fit_transform(list(df["CleanBody"]))
        self.word2id = tfidfvectorizer.vocabulary_
        self.id2word_dict = {}
        for word in self.word2id:
            self.id2word_dict[self.word2id[word]] = word
        self.wordvec_dict = self.__make_word2vec_dict(wordvecs)
        self.avg_word2vec_matrix = self.__make_avg_word2vec_matrix(wordvecs)
        num_topics = len(list(self.wordvec_dict.values())[0])
        self.pvdbow_matrix = makewordvecs.make_pvdbow(df["CleanBody"], num_topics)
        # self.idf = tfidfvectorizer.idf_
        self.sent_to_users, self.cc_to_users, self.recvd_from_users, self.sent_to_users_dict, \
            self.cc_to_users_dict, self.recvd_from_users_dict = self.__get_all_users()
        self.all_users = self.sent_to_users | self.cc_to_users | self.recvd_from_users
        self.user_importance_score_dict = {}
        self.top3_users = self.__get_top3_users()
        self.custom_folders, self.folders_idc_dict = self.__get_custom_folders()
        self.avg_folder_len = self.__get_avg_folder_len()
        self.big_folders = self.__get_big_folders()
        self.three_imp_folders = None
        self.three_time_periods = None
        self.temporally_sound = False
        if len(self.custom_folders) < 3 or self.avg_folder_len < 30 or len(self.big_folders) < 3:
            self.frequent_filer = False
        else:
            self.frequent_filer = True
        return

    def __get_username(self):
        """ Assign the sender with most emails sent as the user name """
        sender_stats = self.df[self.df["FolderType"] == "sent_items"].groupby("SenderName")["SentOn"].count()
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

    def __get_all_users(self):
        sent_to_users = set()
        cc_to_users = set()
        recvd_from_users = set()

        emails_sent = self.df[self.df["SenderName"] == self.username]
        sent_to_users_dict = {}
        for _, row in emails_sent[["To", "idx"]].iterrows():
            text = row[0]
            idx = row[1]
            to = [user.strip() for user in text.split(";") if user != '']
            for item in to:
                item = item.strip()
                if item == self.username or item.upper() == "<UNKNOWN>":
                    continue
                try:
                    sent_to_users_dict[item] += [idx]
                except KeyError:
                    sent_to_users_dict[item] = [idx]
                sent_to_users.add(item)

        cc_to_users_dict = {}
        for _, row in emails_sent[["CC", "idx"]].iterrows():
            text = row[0]
            idx = row[1]
            cc = [user.strip() for user in text.split(";") if user != '']
            for item in cc:
                item = item.strip()
                if item == self.username or item.upper() == "<UNKNOWN>":
                    continue
                try:
                    cc_to_users_dict[item] += [idx]
                except KeyError:
                    cc_to_users_dict[item] = [idx]
                cc_to_users.add(item)

        emails_except_sent = self.df[self.df["FolderType"] != "sent_items"]
        recvd_from_users_dict = {}
        for _, row in emails_except_sent[["SenderName", "idx"]].iterrows():
            text = row[0]
            idx = row[1]
            text = text.strip()
            if text == self.username or text.upper() == "<UNKNOWN>" or text == '':
                continue
            try:
                recvd_from_users_dict[text] += [idx]
            except KeyError:
                recvd_from_users_dict[text] = [idx]
            recvd_from_users.add(text)

        # all_recvd_counts = emails_except_sent.groupby("SenderName")["SentOn"].count()
        # recvd_from_users_dict = dict(zip(all_recvd_counts.index, all_recvd_counts.data))
        # recvd_from_users_dict.pop('<UNKNOWN>', None)
        # for user in all_recvd_counts.index:
        #     if user != self.username and user.upper() != "<UNKNOWN>":
        #         recvd_from_users.add(user)

        all_users = sent_to_users | cc_to_users | recvd_from_users
        for user in all_users:
            try:
                _ = sent_to_users_dict[user]
            except KeyError:
                sent_to_users_dict[user] = []
            try:
                _ = cc_to_users_dict[user]
            except KeyError:
                cc_to_users_dict[user] = []
            try:
                _ = recvd_from_users_dict[user]
            except KeyError:
                recvd_from_users_dict[user] = []

        return sent_to_users, cc_to_users, recvd_from_users, sent_to_users_dict, cc_to_users_dict, recvd_from_users_dict

    def __get_top3_users(self):
        sent_to_users_len_dict = {}
        for key in self.sent_to_users_dict:
            sent_to_users_len_dict[key] = len(self.sent_to_users_dict[key])
        sorted_sent_to_users = sorted(sent_to_users_len_dict.items(), key=operator.itemgetter(1), reverse=True)
        top_users = []
        for tup in sorted_sent_to_users[:3]:
            top_users.append(tup[0])
        return top_users

    def __get_custom_folders(self):
        all_folders = set(list(self.df["FolderType"].unique()))
        try:
            all_folders.remove('inbox')
        except KeyError:
            pass
        try:
            all_folders.remove('sent_items')
        except KeyError:
            pass
        try:
            all_folders.remove('Archive')
        except KeyError:
            pass
        folders_idc_dict = {}
        for folder in all_folders:
            folders_idc_dict[folder] = []
        for _, row in self.df[["FolderType", "idx"]].iterrows():
            folder = row[0]
            if folder not in all_folders:
                continue
            idx = row[1]
            folders_idc_dict[folder] += [idx]
        return all_folders, folders_idc_dict

    def __get_avg_folder_len(self):
        folder_idc = list(self.folders_idc_dict.values())
        total_len = 0
        for idc_list in folder_idc:
            total_len += len(idc_list)
        if len(folder_idc) > 0:
            return total_len / len(folder_idc)
        else:
            return 0

    def __get_big_folders(self):
        big_folders = set()
        for folder in self.folders_idc_dict:
            # noinspection PyTypeChecker
            if len(self.folders_idc_dict[folder]) > max(50, self.avg_folder_len/3):
                big_folders.add(folder)
        return big_folders

    def __make_word2vec_dict(self, wordvecs):
        wordvec_dict = {}
        for word in self.vocab_set:
            try:
                vec = wordvecs[word]
                wordvec_dict[word] = vec
            except KeyError:
                pass
        return wordvec_dict

    def __make_avg_word2vec_matrix(self, wordvecs):
        w2v = wordvecs.syn0
        indices = [idx for idx in range(len(self.id2word_dict)) if self.id2word_dict[idx] in wordvecs]
        w2v = w2v[indices, :]
        matrix = self.tfidf_matrix.dot(w2v)
        return matrix

    def make_user_importance_score_dict(self):
        eps = 1e-3
        user_importance_score_dict = {}
        df = self.df
        for user in self.all_users:
            # noinspection PyBroadException
            try:
                usent = len(df[(df["FolderType"] == "sent_items") & (df["to_cc_bcc"].str.contains(user))])
                urecvd = len(df[(df["FolderType"] != "sent_items") & (df["SenderName"] == user)])
                user_importance_score_dict[user] = (usent+eps)*(usent + urecvd)/(urecvd + 1)
            except:
                user_importance_score_dict[user] = eps
        user_imp_maxval = max(list(user_importance_score_dict.values()))
        for key in user_importance_score_dict:
            user_importance_score_dict[key] = user_importance_score_dict[key]/user_imp_maxval
        self.user_importance_score_dict = user_importance_score_dict
        return

    def make_importance_field(self):
        df = self.df
        offset = 0.01
        importance_field = []
        user_importance_dict = self.user_importance_score_dict
        read_reply_fraction = self.__get_read_reply_fraction()
        for idx, row in df.iterrows():
            imp = 0.
            if row["FolderType"] == 'sent_items':
                imp += 1
                nusers = 0
                userimp = 0.
                for user in row["to_cc_bcc"].strip().split(';'):
                    try:
                        userimp += user_importance_dict[user]
                        nusers += 1
                    except KeyError:
                        pass
                if nusers > 0:
                    imp += userimp / nusers
            else:
                if row['replied']:
                    imp += 1
                if row["UnRead"] == "False":
                    imp += read_reply_fraction
                userimp = 0.
                user = row["SenderName"].strip()
                try:
                    userimp += user_importance_dict[user]
                    imp += userimp
                except KeyError:
                    pass

            importance_field.append(imp+offset)
        self.df["importance"] = importance_field
        return

    def __get_read_reply_fraction(self):
        eps = 1e-3
        df = self.df[self.df["FolderType"] != "sent_items"]
        read = len(df[df["UnRead"] == "False"])
        replied = len(df[df["replied"] == True])
        fraction = replied / (read+eps)
        logger.info("Read-reply fraction = %d/%d = %0.4f" % (replied, read, fraction))
        return fraction

    def make_three_imp_folders(self):
        big_folders_arr = np.array(list(self.big_folders))
        imp = np.array([sum(self.df[self.df['FolderType'] == folder]['importance']) for folder in big_folders_arr])
        maximparg = np.argsort(imp)[-3:][::-1]
        three_imp_folders = big_folders_arr[maximparg]
        self.three_imp_folders = three_imp_folders
        return

    def make_three_time_periods(self):
        months = [[], [], []]
        for idx, row in self.df.iterrows():
            if row["diff"].days < 60:            # change these
                months[0].append(idx)
            elif row["diff"].days < 120:
                months[1].append(idx)
            elif row["diff"].days < 180:
                months[2].append(idx)
        if min(len(months[0]), len(months[1]), len(months[2])) > 20:     # change to 20
            self.temporally_sound = True
        self.three_time_periods = months
        return
