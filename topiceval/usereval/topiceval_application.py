"""
Do not think of tinkering or trying to understand this module. This is one hastily made application
designed to get the job done. The program design is all over the flippin' place.
"""

from __future__ import print_function
from __future__ import division

from topiceval.usereval import senddata
import topiceval.usereval.topicevalGUI as topicevalGUI
from topiceval.coherence import semantic
from topiceval.usereval import task_evaluation

import numpy as np
from prettytable import PrettyTable
from palmettopy.palmetto import Palmetto

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMessageBox

import sys
import os
import pickle
import logging
import operator

logger = logging.getLogger(__name__)


def topic_tuple_overlap(topictups):
    """
    Return an overlap measure over 3 topic distributions

    :param topictups: a list of 3 tuples, each tuple contains word-probability information
    :return: overlap measure given by product of probabilities of words overlapping across all
        three tuples
    """
    topic_dicts = {}, {}, {}
    for i, topictup in enumerate(topictups):
        for tup in topictup:
            topic_dicts[i][tup[0]] = tup[1]
    overlap = 0.0
    for word in topic_dicts[0]:
        try:
            overlap += topic_dicts[0][word] * topic_dicts[1][word] * topic_dicts[2][word]
        except KeyError:
            continue
    return overlap


def all_triplet_overlaps(models):
    """
    Get overlap measure for all possible triplet combination of topics

    :param models: a list of 3 topic models
    :return: a list containing overlap measures for all triplets
    """
    num_topics = len(models[0].topic_tuples)
    all_overlaps = []
    for i in range(num_topics):
        for j in range(num_topics):
            for k in range(num_topics):
                try:
                    all_overlaps = all_overlaps + [
                        [[i, j, k], topic_tuple_overlap([models[0].topic_tuples[i],
                                                         models[1].topic_tuples[j],
                                                         models[2].topic_tuples[k]])]]
                except Exception as e:
                    print(models[0].topic_tuples[0])
                    print(models[1].topic_tuples[0])
                    print(models[2].topic_tuples[0])
                    print(e)
                    exit()
    return all_overlaps


class TopicEvalWindowClass(QtWidgets.QMainWindow, topicevalGUI.Ui_MainWindow):
    def __init__(self, models, dirname, num_topics, threaded, email_network, order, A):

        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        if email_network.frequent_filer:
            self.i = -9
        elif email_network.temporally_sound:
            self.i = -6
        else:
            self.i = -3
        self.all_models = models
        self.dirname = dirname
        self.num_topics = num_topics
        self.threaded = threaded
        self.email_network = email_network
        self.order = order
        self.wordspertopic = 10
        self.states = [0] * 3
        self.user_phase = True
        self.mapping = [0, 1, 2]
        self.selected_items = [[]] * 3
        self.headings = [""] * 3
        self.added_words = [""] * 3
        for model in self.all_models:
            semantic.umass_coherence(dirname, model.topic_tuples, model,  A=A, numwords=10)
        orders = self.select_topics()
        # rep1, rep2, rep3 = self.topics_order()
        # self.rep_order = [rep1, rep2, rep3]
        self.rep_order = orders
        for idx, model in enumerate(self.all_models):
            # noinspection PyUnresolvedReferences
            model.representative_topic_tuples = [model.topic_tuples[index] for index in self.rep_order[idx]]
            model.representative_topics_umass_pmi = [model.representative_topics_umass_pmi[index]
                                                     for index in self.rep_order[idx]]
            # semantic.umass_coherence(dirname, model.representative_topic_tuples, model, numwords=10)
            semantic.w2v_coherences(model, email_network.wordvec_dict, num_topics)
        palmetto = Palmetto()
        semantic.other_coherences(palmetto, self.all_models, self.num_topics, numwords=10)

        self.listWidgets = [self.listWidget_1, self.listWidget_2, self.listWidget_3]
        self.showNext10PushButtons = [self.showNext10PushButton_1, self.showNext10PushButton_2,
                                      self.showNext10PushButton_3]
        self.showPrevious10PushButtons = [self.showPrevious10PushButton_1, self.showPrevious10PushButton_2,
                                          self.showPrevious10PushButton_3]

        self.buttonGroups = [self.buttonGroup_1, self.buttonGroup_2, self.buttonGroup_3]
        self.buttonGroupsThematic = [self.buttonGroup_4, self.buttonGroup_5, self.buttonGroup_6]
        self.allButtonGroups = self.buttonGroups + self.buttonGroupsThematic

        self.comboBoxes = [self.comboBox_1, self.comboBox_2, self.comboBox_3]

        path = os.path.dirname(os.path.abspath(__file__)) + "/resources/icon.png"
        self.setWindowIcon(QtGui.QIcon(path))

        self.fontSpinBox.valueChanged.connect(self.change_font_size)

        path = os.path.dirname(os.path.abspath(__file__)) + "/resources/bkdimage.jpg"
        try:
            self.bkdLabel.setPixmap(QtGui.QPixmap(path))
            self.bkdLabel.setScaledContents(True)
        except AttributeError:
            pass

        self.nextCommandLinkButton.clicked.connect(self.show_next_topic)

        # TODO: Fix this, using lambda in for loop is not working as it takes the final value of i for every button
        self.showNext10PushButtons[0].clicked.connect(lambda: self.show_next_10(0))
        self.showNext10PushButtons[1].clicked.connect(lambda: self.show_next_10(1))
        self.showNext10PushButtons[2].clicked.connect(lambda: self.show_next_10(2))
        self.showPrevious10PushButtons[0].clicked.connect(lambda: self.show_previous_10(0))
        self.showPrevious10PushButtons[1].clicked.connect(lambda: self.show_previous_10(1))
        self.showPrevious10PushButtons[2].clicked.connect(lambda: self.show_previous_10(2))
        # for buttonGroup in self.buttonGroups:
        #     buttonGroup.buttonClicked.connect(self.groupbutton_clicked)
        # for buttonGroup in self.buttonGroupsThematic:
        #     buttonGroup.buttonClicked.connect(self.groupbutton_clicked)
        for buttonGroup in self.allButtonGroups:
            buttonGroup.buttonClicked.connect(self.groupbutton_clicked)
        for i in range(len(self.states)):
            self.update_gui(i)

            # self.topicimageLabel.setPixmap(QtGui.QPixmap(dirname + '/topic%d.png' % self.i))
            # self.topicimageLabel.setScaledContents(True)

    # def show_next_image(self):
    #     if self.i >= self.num_topics:
    #         return
    #     self.topicimageLabel.setPixmap(QtGui.QPixmap(self.dirname + '/topic%d.png' % self.i))
    #     self.topicimageLabel.setScaledContents(True)
    #     self.i += 1
    #     if self.i >= self.num_topics:
    #         self.nextCommandLinkButton.setText("Finish!")
    #     return

    def select_topics(self):
        num_topics = self.num_topics
        if self.order == 0:
            return [range(num_topics)] * 3
        if self.order == 1:
            orders = []
            for model in self.all_models:
                # semantic_order = np.argsort(model.representative_topics_umass_pmi)[-num_topics:][::-1]
                baseline_coverage_scores = task_evaluation.baseline_coverage(model, self.email_network.df)
                baseline_coverage_order = np.argsort(baseline_coverage_scores)[-num_topics:][::-1]
                orders.append(baseline_coverage_order)
            return orders
        if self.order == 2:
            orders = []
            for model in self.all_models:
                # topic_sets = task_evaluation.get_topic_document_membership_sets(model)
                # max_cover_topics = task_evaluation.max_coverage_greedy(topic_sets, model.num_docs,
                #                                                        num_sets=num_topics)
                document_topic_membership_dict = \
                    task_evaluation.get_document_topic_membership_dict(model, self.email_network.df)
                max_cover_topics = task_evaluation.max_coverage_greedy(model,
                                                                       document_topic_membership_dict,
                                                                       sel_topics=num_topics, df=self.email_network.df)
                # pmis = model.representative_topics_umass_pmi
                # pmis_ord = [pmis[max_cover_topics[idx]] for idx in range(len(max_cover_topics))]
                # order = np.argsort(pmis_ord)[::-1]
                # max_cover_topics = [max_cover_topics[idx] for idx in order]
                orders.append(max_cover_topics)
            return orders

    def topics_order(self):
        """
        Decide order in which to show topics based on their word-overlap measure

        :return: 3 lists containing order of topic no. for the corresponding topic model
        """
        all_overlaps = all_triplet_overlaps(self.all_models)
        sorted_overlaps = sorted(all_overlaps, key=operator.itemgetter(1), reverse=True)
        set1, set2, set3 = set(), set(), set()
        rep1, rep2, rep3 = [], [], []
        topics_covered = 0
        for item in sorted_overlaps:
            if topics_covered >= self.num_topics:
                break
            topic_nums = item[0]
            if topic_nums[0] in set1 or topic_nums[1] in set2 or topic_nums[2] in set3:
                continue
            else:
                rep1.append(topic_nums[0])
                rep2.append(topic_nums[1])
                rep3.append(topic_nums[2])
                set1.add(topic_nums[0])
                set2.add(topic_nums[1])
                set3.add(topic_nums[2])
                topics_covered += 1
        assert (len(rep1) == self.num_topics and len(rep2) == self.num_topics and len(rep3) == self.num_topics)
        return rep1, rep2, rep3

    def change_font_size(self):
        size = self.fontSpinBox.value()
        font = QtGui.QFont()
        font.setPointSize(size)
        font.setItalic(True)
        font.setWeight(50)
        for listWidget in self.listWidgets:
            listWidget.setFont(font)
        return

    def show_top_user_topics(self):
        top_users = self.email_network.top3_users
        print("Top Users: ", top_users)
        for user in top_users:
            print("\n\n-------------------------------------- USER: %s -----------------------------------" % user)
            sent_indices = self.email_network.sent_to_users_dict[user]
            recvd_indices = self.email_network.recvd_from_users_dict[user]
            all_indices = set(sent_indices) | set(recvd_indices)
            print(all_indices)
            for model in self.all_models:
                print("\n-------------- MODEL: %s ----------------" % model.modelname)
                tuples = model.get_documents_top_topic_tuples(document_indices=all_indices)
                for tup in tuples:
                    print(tup)
                    print('')
        return

    def top_month_topic_tuples(self, model, monthnum):
        doc_idc = self.email_network.three_time_periods[monthnum]
        tuples, weights = model.get_documents_top_topic_tuples(document_indices=doc_idc,
                                                               imp=list(
                                                                   self.email_network.df["importance"][doc_idc]))
        return tuples, weights

    def top_folder_topic_tuples(self, model, foldernum):
        top_folders = self.email_network.three_imp_folders
        folder = top_folders[foldernum]
        folder_idc = self.email_network.folders_idc_dict[folder]
        tuples, weights = model.get_documents_top_topic_tuples(document_indices=folder_idc,
                                                               imp=list(
                                                                   self.email_network.df["importance"][folder_idc]))
        return tuples, weights

    def top_user_topic_tuples(self, model, usernum):
        top_users = self.email_network.top3_users
        user = top_users[usernum]
        sent_indices = self.email_network.sent_to_users_dict[user]
        recvd_indices = self.email_network.recvd_from_users_dict[user]
        all_indices = set(sent_indices) | set(recvd_indices)
        tuples, weights = model.get_documents_top_topic_tuples(document_indices=all_indices,
                                                               imp=list(
                                                                   self.email_network.df["importance"][all_indices]))
        return tuples, weights

    def scores_update(self):
        """ Update stored score for the topic as given by the user """
        if self.i <= -6:
            # foldernum = self.i + 8
            modelnum = self.i + 8
            self.all_models[modelnum].representative_structures_score[0] = \
                int(self.buttonGroup_2.checkedButton().text())
        elif self.i <= -3:
            modelnum = self.i + 5
            self.all_models[modelnum].representative_structures_score[1] = \
                int(self.buttonGroup_2.checkedButton().text())
        elif self.i <= 0:
            modelnum = self.i + 2
            self.all_models[modelnum].representative_structures_score[2] = \
                int(self.buttonGroup_2.checkedButton().text())
        else:
            for i, group in enumerate(self.buttonGroups):
                modelnum = self.mapping[i]
                if group.checkedId() == -1:
                    score = '-'
                else:
                    score = int(group.checkedButton().text())
                self.all_models[modelnum].representative_topics_semantic_scores += [score]
            for i, group in enumerate(self.buttonGroupsThematic):
                modelnum = self.mapping[i]
                if group.checkedId() == -1:
                    score = '-'
                else:
                    score = int(group.checkedButton().text())
                self.all_models[modelnum].representative_topics_thematic_scores += [score]
        return

    def labels_update(self):
        """ Update stored label-based feedback for the topic as given by the user """
        for i in range(3):
            modelnum = self.mapping[i]
            self.all_models[modelnum].representative_topics_enumlabels.append(self.comboBoxes[i].currentText())
            self.all_models[modelnum].representative_topics_textlabels.append(self.headings[i])
            self.all_models[modelnum].representative_topics_addedwords.append(self.added_words[i])
            # self.all_models[modelnum].representative_topics_selectedwords.\
            #     append([index.row() for index in self.listWidgets[i].selectedIndexes()])
            self.selected_items[i] = []
            if self.states[i] == 1:
                last_idx = self.listWidgets[i].count() - 1
                for j in range(last_idx + 1):
                    if self.listWidgets[i].item(j).isSelected():
                        self.selected_items[i] += [j]
            self.all_models[modelnum].representative_topics_selectedwords += [self.selected_items[i]]
        return

    def unselect_radiobuttons(self):
        for buttonGroup in self.allButtonGroups:
            buttonGroup.setExclusive(False)
            for button in buttonGroup.buttons():
                button.setChecked(False)
            buttonGroup.setExclusive(True)
        return

    def unselect_listwidgets(self):
        for listwidget in self.listWidgets:
            listwidget.clearSelection()
        return

    def reset_comboboxes(self):
        for combobox in self.comboBoxes:
            combobox.setCurrentIndex(0)
        return

    def groupbutton_clicked(self):
        all_selected_flag = True
        none_selected_flag = True
        for group in self.allButtonGroups:
            if group.checkedId() == -1:
                all_selected_flag = False
            else:
                none_selected_flag = False
        if self.i >= 0 and (all_selected_flag or none_selected_flag):
            self.nextCommandLinkButton.setEnabled(True)
        elif self.i < 0 and self.buttonGroup_2.checkedId() != -1:
            self.nextCommandLinkButton.setEnabled(True)
        else:
            self.nextCommandLinkButton.setEnabled(False)
        return

    def show_next_10(self, i):
        if self.states[i] == 1:
            last_idx = self.listWidgets[i].count() - 1
            for j in range(last_idx + 1):
                if self.listWidgets[i].item(j).isSelected():
                    self.selected_items[i] += [j]
            self.headings[i] = self.listWidgets[i].item(2).text()
            self.added_words[i] = self.listWidgets[i].item(last_idx).text()

        self.states[i] += 1
        self.update_gui(i)
        return

    def show_previous_10(self, i):
        self.states[i] -= 1
        self.update_gui(i)
        return

    def show_next_topic(self):
        self.i += 1
        self.scores_update()
        if self.i > 0:
            self.labels_update()
            self.mapping = np.random.choice(len(self.all_models), 3, replace=False)
            print(self.mapping)
        if self.i >= self.num_topics:
            self.states = [-1] * len(self.states)
            self.show_results()
        else:
            self.states = [0] * len(self.states)
        for i in range(len(self.states)):
            self.update_gui(i)
        return

    def show_topic(self, i):
        start = (self.states[i] - 1) * self.wordspertopic
        end = start + self.wordspertopic
        total_weight, strings = self.tuples_to_strings(start=start, end=end, i=i)
        if self.i >= 0:
            self.listWidgets[i].addItems(["Total Wt (%d:%d)" % (start + 1, end), "(%0.3f)" % total_weight, ""])
        elif self.i >= -3:
            usernum = self.i + 3
            username = self.email_network.top3_users[usernum]
            self.listWidgets[i].addItems(["User: %s" % username, "(%0.3f)" % total_weight, ""])
        elif self.i >= -6:
            monthnum = self.i + 6
            months = ['June-July', 'April-May', 'Feb-March']
            self.listWidgets[i].addItems(["Months: %s" % months[monthnum], "(%0.3f)" % total_weight, ""])
        elif self.i >= -9:
            foldernum = self.i + 9
            foldername = self.email_network.three_imp_folders[foldernum]
            self.listWidgets[i].addItems(["Folder: %s" % foldername, "(%0.3f)" % total_weight, ""])

        self.listWidgets[i].item(2).setTextAlignment(132)  # centering
        self.listWidgets[i].item(1).setTextAlignment(132)  # centering
        self.listWidgets[i].item(0).setTextAlignment(132)  # centering
        self.listWidgets[i].item(2).setFlags(self.listWidgets[i].item(2).flags() & ~QtCore.Qt.ItemIsSelectable)
        self.listWidgets[i].item(1).setFlags(self.listWidgets[i].item(1).flags() & ~QtCore.Qt.ItemIsSelectable)
        self.listWidgets[i].item(0).setFlags(self.listWidgets[i].item(0).flags() & ~QtCore.Qt.ItemIsSelectable)
        self.listWidgets[i].addItems(strings)
        if self.states[i] == 1:
            self.listWidgets[i].addItem("")
            lastitem_idx = self.listWidgets[i].count() - 1
            self.listWidgets[i].item(lastitem_idx).setFlags(self.listWidgets[i].item(lastitem_idx).flags() |
                                                            QtCore.Qt.ItemIsEditable)
            self.listWidgets[i].item(lastitem_idx).setFlags(self.listWidgets[i].item(lastitem_idx).flags()
                                                            & ~QtCore.Qt.ItemIsSelectable)
            self.listWidgets[i].item(lastitem_idx).setForeground(QtGui.QBrush(QtGui.QColor(220, 220, 220)))
            self.listWidgets[i].item(2).setFlags(self.listWidgets[i].item(2).flags() | QtCore.Qt.ItemIsEditable)
            self.listWidgets[i].item(2).setToolTip("Write a label for the topic!")
            self.listWidgets[i].item(2).setForeground(QtGui.QBrush(QtGui.QColor(220, 220, 220)))
            if self.headings[i] == "" or self.headings[i] == "(Enter Topic Label)":
                self.listWidgets[i].item(2).setText("(Enter Topic Label)")
            else:
                self.listWidgets[i].item(2).setText(self.headings[i])
            for itemnum in self.selected_items[i]:
                self.listWidgets[i].item(itemnum).setSelected(True)
            if self.added_words[i] == "" or self.added_words[i] == "(Add Words)":
                self.listWidgets[i].item(lastitem_idx).setText("Add Words")
            else:
                self.listWidgets[i].item(lastitem_idx).setText(self.added_words[i])
        return

    def update_gui(self, i):
        state = self.states[i]
        labels = [self.scoreLabel_1, self.scoreLabel_3, self.scoreLabel_4, self.scoreLabel_5, self.scoreLabel_6]
        if self.i < 0:
            for gr in self.allButtonGroups:
                for button in gr.buttons():
                    button.setHidden(True)
            for label in labels:
                label.setHidden(True)
            for cb in self.comboBoxes:
                cb.setHidden(True)
            for button in self.buttonGroup_2.buttons():
                button.setHidden(False)
            self.scoreLabel_2.setText('Concept Coverage')
        else:
            for gr in self.allButtonGroups:
                for button in gr.buttons():
                    button.setHidden(False)
            for label in labels:
                label.setHidden(False)
            for cb in self.comboBoxes:
                cb.setHidden(False)
            self.scoreLabel_2.setText('Semantic Coherence')
        if state == 0:
            self.states[i] = 1
            self.listWidgets[i].clear()
            self.show_topic(i)
            self.showNext10PushButtons[i].setEnabled(True)
            self.showPrevious10PushButtons[i].setEnabled(False)
            self.unselect_radiobuttons()
            self.unselect_listwidgets()
            self.reset_comboboxes()
            self.nextCommandLinkButton.setEnabled(True)
        if state == 1:
            self.listWidgets[i].clear()
            self.show_topic(i)
            self.showNext10PushButtons[i].setEnabled(True)
            self.showPrevious10PushButtons[i].setEnabled(False)
        elif state == 2:
            self.listWidgets[i].clear()
            self.show_topic(i)
            self.showNext10PushButtons[i].setEnabled(True)
            self.showPrevious10PushButtons[i].setEnabled(True)
        elif state == 3:
            self.listWidgets[i].clear()
            self.show_topic(i)
            self.showNext10PushButtons[i].setEnabled(False)
            self.showPrevious10PushButtons[i].setEnabled(True)
        elif state == -1:
            for listWidget in self.listWidgets:
                listWidget.clear()
            for showPrevious10PushButton in self.showPrevious10PushButtons:
                showPrevious10PushButton.setEnabled(False)
            for showNext10PushButton in self.showNext10PushButtons:
                showNext10PushButton.setEnabled(False)
            self.unselect_radiobuttons()
            self.reset_comboboxes()
            self.nextCommandLinkButton.setEnabled(True)
            self.nextCommandLinkButton.setText("Close")
            self.nextCommandLinkButton.clicked.disconnect()
            self.nextCommandLinkButton.clicked.connect(self.close)
        if self.i < 0:
            self.nextCommandLinkButton.setEnabled(False)
        return

    def clean_gui(self):
        for listWidget in self.listWidgets:
            listWidget.clear()
        for showPrevious10PushButton in self.showPrevious10PushButtons:
            showPrevious10PushButton.setEnabled(False)
        for showNext10PushButton in self.showNext10PushButtons:
            showNext10PushButton.setEnabled(False)
        self.unselect_radiobuttons()
        self.reset_comboboxes()
        self.nextCommandLinkButton.setEnabled(True)
        self.nextCommandLinkButton.setText("Close")
        return

    # noinspection PyStringFormat
    def show_results(self):
        # global_table = PrettyTable(['Model', 'Top Collabrs Semantic Score', 'Top Collabrs Thematic Score'])
        table_structures = PrettyTable(["Model", "Folder Score", "Temporal Score", "User Score"])
        table_topics = PrettyTable(["Model", "Semantic", "Thematic", "Label",
                                    "Intruders", "UMass", "UCI PMI", "UCI NPMI", "W2V PAIR", "W2V MST"])
        for model in self.all_models:
            table_structures.add_row([model.modelname, model.representative_structures_score[0],
                                      model.representative_structures_score[1],
                                      model.representative_structures_score[2]])
            print("\n--------------- USER EVALUATION RESULTS FOR {0} -------------------"
                  .format(model.modelname))
            table = PrettyTable(["Topic No.", "Semantic", "Thematic", "Label",
                                 "Intruders", "UMass", "UCI PMI", "UCI NPMI", "W2V PAIR", "W2V MST"])
            for idx in range(self.num_topics):
                table.add_row([idx + 1, model.representative_topics_semantic_scores[idx],
                               model.representative_topics_thematic_scores[idx],
                               model.representative_topics_enumlabels[idx],
                               len(model.representative_topics_selectedwords[idx]),
                               '%0.2f' % model.representative_topics_umass_pmi[idx],
                               '%0.3f' % model.representative_topics_uci_pmi[idx],
                               '%0.4f' % model.representative_topics_uci_npmi[idx],
                               '%0.3f' % model.representative_topics_w2vpairwise[idx],
                               '%0.3f' % model.representative_topics_w2vmst[idx]])
            # logger.info("\nTable for Top 3 Collaborators\n{}".format(table_top_users))
            # semantic_topuser_scores_arr = np.zeros(len(model.representative_users_semantic_scores))
            # for idx, val in enumerate(model.representative_users_semantic_scores):
            #     if val == "-":
            #         semantic_topuser_scores_arr[idx] = np.nan
            #     else:
            #         semantic_topuser_scores_arr[idx] = val
            # mean_topuser_semantic_score = np.nanmean(semantic_topuser_scores_arr)
            # var_topuser_semantic_score = np.nanstd(semantic_topuser_scores_arr)
            # logger.info("Average *top* collaborators' topics semantic score for model is %0.2f(%0.2f)"
            #             % (mean_topuser_semantic_score, var_topuser_semantic_score))
            # thematic_topuser_scores_arr = np.zeros(len(model.representative_users_thematic_scores))
            # for idx, val in enumerate(model.representative_users_thematic_scores):
            #     if val == "-":
            #         thematic_topuser_scores_arr[idx] = np.nan
            #     else:
            #         thematic_topuser_scores_arr[idx] = val
            # mean_topuser_thematic_score = np.nanmean(thematic_topuser_scores_arr)
            # var_topuser_thematic_score = np.nanstd(thematic_topuser_scores_arr)
            # logger.info("Average *top* collaborators' topics thematic score for model is %0.2f(%0.2f)"
            #             % (mean_topuser_thematic_score, var_topuser_thematic_score))
            # global_table.add_row([model.modelname, '%0.2f +- %0.3f'
            #                       % (mean_topuser_semantic_score, var_topuser_semantic_score), '%0.2f +- %0.3f'
            #                       % (mean_topuser_thematic_score, var_topuser_thematic_score)])

            semantic_scores_arr = np.zeros(len(model.representative_topics_semantic_scores))
            for idx, val in enumerate(model.representative_topics_semantic_scores):
                if val == "-":
                    semantic_scores_arr[idx] = np.nan
                else:
                    semantic_scores_arr[idx] = val
            mean_semantic_score = np.nanmean(semantic_scores_arr)
            var_semantic_score = np.nanstd(semantic_scores_arr)
            # logger.info("Average user semantic score for model is %0.2f(%0.2f)" % (mean_semantic_score,
            #                                                                        var_semantic_score))
            thematic_scores_arr = np.zeros(len(model.representative_topics_thematic_scores))
            for idx, val in enumerate(model.representative_topics_thematic_scores):
                if val == "-":
                    thematic_scores_arr[idx] = np.nan
                else:
                    thematic_scores_arr[idx] = val
            mean_thematic_score = np.nanmean(thematic_scores_arr)
            var_thematic_score = np.nanstd(thematic_scores_arr)
            # logger.info("Average user thematic score for model is %0.2f(%0.2f)" % (mean_thematic_score,
            #                                                                        var_thematic_score))
            njunk = model.representative_topics_enumlabels.count("Junk")
            nfused = model.representative_topics_enumlabels.count("Fused")
            nunbalanced = model.representative_topics_enumlabels.count("Unbalanced")
            # logger.info("Number of labels: junk: %d, imbalanced: %d, fused: %d" % (njunk, nunbalanced, nfused))
            nintruder_words = 0
            for added_words in model.representative_topics_selectedwords:
                nintruder_words += len(added_words)
            avg_intruder_words = nintruder_words / len(model.representative_topics_selectedwords)
            # print("Average intruder words per topic: {}\n".format(avg_intruder_words))
            avg_umass_pmi = np.mean(model.representative_topics_umass_pmi[:self.num_topics])
            var_umass_pmi = np.std(model.representative_topics_umass_pmi[:self.num_topics])
            # print("Average UMass PMI of topics: %0.2f(%0.3f)" % (avg_umass_pmi, var_umass_pmi))
            avg_uci_pmi = np.mean(model.representative_topics_uci_pmi[:self.num_topics])
            var_uci_pmi = np.std(model.representative_topics_uci_pmi[:self.num_topics])
            # print("Average UCI PMI of topics: %0.2f(%0.3f)" % (avg_uci_pmi, var_uci_pmi))
            avg_uci_npmi = np.mean(model.representative_topics_uci_npmi[:self.num_topics])
            var_uci_npmi = np.std(model.representative_topics_uci_npmi[:self.num_topics])

            avg_w2v_pairwise = np.mean(model.representative_topics_w2vpairwise[:self.num_topics])
            var_w2v_pairwise = np.std(model.representative_topics_w2vpairwise[:self.num_topics])

            avg_w2v_mst = np.mean(model.representative_topics_w2vmst[:self.num_topics])
            var_w2v_mst = np.std(model.representative_topics_w2vmst[:self.num_topics])
            # print("Average UCI NPMI of topics: %0.3f(%0.5f)" % (avg_uci_npmi, var_uci_npmi))
            # avg_cv = np.mean(model.representative_topics_cv_coherence)
            # var_cv = np.std(model.representative_topics_cv_coherence)
            # print("Average CV coherence of topics: %0.2f(%0.5f)" % (avg_cv, var_cv))
            table_topics.add_row([model.modelname, '%0.2f+-%0.3f' % (mean_semantic_score, var_semantic_score),
                                  '%0.2f+-%0.3f'
                                  % (mean_thematic_score, var_thematic_score),
                                  '%d/%d/%d' % (njunk, nunbalanced, nfused),
                                  '%0.2f' % avg_intruder_words, '%0.2f+-%0.3f' % (avg_umass_pmi, var_umass_pmi),
                                  '%0.2f+-%0.3f' % (avg_uci_pmi, var_uci_pmi),
                                  '%0.3f+-%0.4f' % (avg_uci_npmi, var_uci_npmi),
                                  '%0.2f+-%0.3f' % (avg_w2v_pairwise, var_w2v_pairwise),
                                  '%0.2f+-%0.3f' % (avg_w2v_mst, var_w2v_mst)])
            logger.info("\nTable for Individual Topics\n{}".format(table))
        logger.info("\nSTRUCTURAL EVALUATION RESULTS:\n{}".format(table_structures))
        logger.info("\nINDIVIDUAL TOPIC EVALUATION RESULTS:\n{}".format(table_topics))
        return

    def cleanup(self):
        dirpath = self.dirname
        files = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
        for file in files:
            os.remove(file)
        try:
            os.rmdir(dirpath)
            # TODO: Change brute name to chunk of dirpath
            os.rmdir("./data_topiceval")
            logger.debug("Completed temporary file cleanup.")
        except OSError:
            logger.warning("Could not delete the temp folder. You may do it manually for directory at %s" % dirpath)
        return

    def showdialog(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Thank you! Ready to send results!")
        msg.setInformativeText("The results will be sent via email. Privacy is of utmost importance and only the "
                               "top words shown to you and the scores/other info added by you will be sent.")
        msg.setWindowTitle("Submit Results")
        # TODO: Set details of files being sent
        # msg.setDetailedText("The details of files being sent are as follows: 1) topic matrices: the word-probability"
        #                     " distribution of topics, the top words were shown and ranked by you. "
        #                     "2) Your scores for the topics "
        #                     "3) This data will be sent to t-avsriv@microsoft.com")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        # noinspection PyUnresolvedReferences
        msg.buttonClicked.connect(self.msgbtn)
        sys.exit(msg.exec_())

    def msgbtn(self, i):
        if i.text() == "OK":
            self.submit()
        else:
            buttonReply = QMessageBox.question(self, 'Really Quit', "The scores won't be sent. Quit(Yes) "
                                                                    "or Submit(\"No\")?",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                self.close()
            else:
                self.submit()
        return

    def submit(self):
        threaded = self.threaded
        shown_topic_tuples = []
        shown_topic_semantic_scores = []
        shown_topic_thematic_scores = []
        shown_topic_enumlabels = []
        shown_topic_textlabels = []
        shown_topic_addedwords = []
        shown_topic_selectedwords = []
        for i, model in enumerate(self.all_models):
            shown_topic_tuples.append([model.modelname, threaded,
                                       model.representative_topic_tuples[:self.num_topics]])
            # shown_topic_tuples.append([model.modelname, threaded,
            #                            model.representative_topic_tuples[self.rep_order[i]]])
            shown_topic_semantic_scores.append(model.representative_topics_semantic_scores)
            shown_topic_thematic_scores.append(model.representative_topics_thematic_scores)
            shown_topic_enumlabels.append(model.representative_topics_enumlabels)
            shown_topic_textlabels.append(model.representative_topics_textlabels)
            shown_topic_addedwords.append(model.representative_topics_addedwords)
            shown_topic_selectedwords.append(model.representative_topics_selectedwords)

        with open(self.dirname + "shown_topic_tuples.pkl", "wb") as handle:
            pickle.dump(shown_topic_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.dirname + "shown_topic_semantic_scores.pkl", "wb") as handle:
            pickle.dump(shown_topic_semantic_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.dirname + "shown_topic_thematic_scores.pkl", "wb") as handle:
            pickle.dump(shown_topic_thematic_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.dirname + "shown_topic_enumlabels.pkl", "wb") as handle:
            pickle.dump(shown_topic_enumlabels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.dirname + "shown_topic_textlabels.pkl", "wb") as handle:
            pickle.dump(shown_topic_textlabels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.dirname + "shown_topic_addedwords.pkl", "wb") as handle:
            pickle.dump(shown_topic_addedwords, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.dirname + "shown_topic_selectedwords.pkl", "wb") as handle:
            pickle.dump(shown_topic_selectedwords, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # os.remove(self.dirname + "Amatrix.npz")
        os.remove(self.dirname + "corpus.npy")
        os.remove(self.dirname + "dfindices_in_corpus.npy")
        # np.save(dirname + "shown_topic_tuples.npy", shown_topic_tuples)
        # np.save(dirname + "shown_topic_scores.npy", shown_topic_scores)
        senddata.makezip(self.dirname)
        senddata.sendmail()
        QMessageBox.information(self, 'Success', "Results sent successfully!")
        self.cleanup()
        return

    def tuples_to_strings(self, start, end, i):
        tuple_list, weight = None, None
        if self.i >= self.num_topics:
            return
        if self.i >= 0:
            tuple_list = self.all_models[self.mapping[i]].representative_topic_tuples[self.i][start:end]
            # umass_pmi, umass_npmi = \
            #     semantic_coherence(self.dirname, self.all_models[self.mapping[i]], tuple_list, numwords=10)
            # self.all_models[self.mapping[i]].representative_topics_semcoherences.append(coherence)
            weight = None
        elif self.i >= -3:
            usernum = self.i + 3
            modelnum = self.i + 3
            tuples, weights = self.top_user_topic_tuples(model=self.all_models[modelnum], usernum=usernum)
            tuple_list = tuples[i][start:end]
            weight = weights[i]
        elif self.i >= -6:
            if not self.email_network.temporally_sound:
                self.i = -3
                return self.tuples_to_strings(start, end, i)
            else:
                monthnum = self.i + 6
                modelnum = self.i + 6
                tuples, weights = self.top_month_topic_tuples(model=self.all_models[modelnum], monthnum=monthnum)
                tuple_list = tuples[i][start:end]
                weight = weights[i]
        elif self.i >= -9:
            foldernum = self.i + 9
            modelnum = self.i + 9
            tuples, weights = self.top_folder_topic_tuples(model=self.all_models[modelnum], foldernum=foldernum)
            tuple_list = tuples[i][start:end]
            weight = weights[i]
        # strings = ["%s (%0.3f)" % (tup[0], tup[1]) for tup in tuple_list]
        strings = []
        for tup in tuple_list:
            if tup[1] > 0.0005:
                strings.append("%s (%0.3f)" % (tup[0], tup[1]))
            else:
                strings.append("%s (<0.0005)" % (tup[0]))
        total_weight = 0.
        if self.i >= 0:
            for tup in tuple_list:
                total_weight += tup[1]
        else:
            total_weight = weight
        return total_weight, strings

    def closeEvent(self, event):
        """ Upon closing the application, cleanup all temporary data """
        self.cleanup()
        event.accept()
        return

        # def closeEvent(self, event):
        #
        #     quit_msg = "Are you sure you want to exit the program?"
        #     reply = QMessageBox.question(self, 'Message',
        #                                        quit_msg, QMessageBox.Yes, QMessageBox.No)
        #
        #     if reply == QMessageBox.Yes:
        #         event.accept()
        #     else:
        #         event.ignore()


def main(models, dirname, num_topics, threaded, email_network, order, A):
    # noinspection PyBroadException
    try:
        app = QApplication.instance()
    except:
        app = QApplication(sys.argv)

    window = TopicEvalWindowClass(models=models, dirname=dirname, num_topics=num_topics, threaded=threaded,
                                  email_network=email_network, order=order, A=A)
    window.show()
    app.exec_()
    return
