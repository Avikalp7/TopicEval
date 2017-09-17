"""
This module is used for plotting various results on topic model matrices. Wordclouds have been removed for now.
On Linux, matplotlib import can be problematic, in that case, this module should be omitted when evaluation mode
is off for all topic models
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg", force=True)
import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# import os


# TODO: Make plotting more flexible, incorporate more arguments


def autolabel(rects, ax, vertical=True):
    """
    Attach labels to a horizontal bar graph
    Input:
    rects <list from ax.barh>
    ax <axis object of plt.subplots>
    Output:
    ax <axis object of plt.subplots> with apt text field to display values on bars is returned
    """
    if vertical:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.01,
                    '%0.2f' % height,
                    ha='center', va='bottom')
    else:
        for i, rect in enumerate(rects):
            width = rect.get_width()
            ax.text(1.05*width, rect.get_y() + rect.get_height()/2., '%0.2f' % width)
    return ax


def plot_colormap(color_matrix, nrows=1, figsize=(10, 10),
                  xlabel='', ylabel='', title="Colormap", tight_layout=True, yaxisnormed=True):
    num_rows = color_matrix.shape[0]
    num_cols = color_matrix.shape[1]
    # noinspection PyTypeChecker
    fig, (ax) = plt.subplots(nrows=nrows, figsize=figsize)
    if yaxisnormed:
        ylim = 1
    else:
        ylim = num_cols
    ax.imshow(color_matrix, extent=[0, num_rows, ylim, 0], aspect='auto')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.gca().invert_yaxis()
    if tight_layout:
        plt.tight_layout()
    plt.show()
    return


def plot_bar(values, labeled=True, bar_type='vbar', xlabel='', ylabel='', title='', ticks=None, tight_layout=True,
             const_val_line=None):
    if bar_type not in ['vbar', 'hbar']:
        raise ValueError("Invalid value for attribute type")
    fig, ax = plt.subplots()
    if bar_type == 'hbar':
        y_pos = np.arange(len(values))
        rects = ax.barh(y_pos, values, align='center', color='green', ecolor='black')
        if labeled:
            ax = autolabel(rects, ax, vertical=False)
        ax.set_yticks(y_pos)
        if ticks is None:
            ax.set_yticklabels(tuple(np.arange(1, len(values) + 1)))
        else:
            ax.set_yticklabels(ticks)
    elif bar_type == 'vbar':
        x_pos = np.arange(len(values))
        rects = ax.bar(x_pos, values, align='center', color='green', ecolor='black')
        if labeled:
            ax = autolabel(rects, ax, vertical=True)
        ax.set_xticks(x_pos)
        if ticks is None:
            ax.set_xticklabels(tuple(np.arange(1, len(values) + 1)))
        else:
            ax.set_xticklabels(ticks)
    if const_val_line is not None:
        plt.plot([const_val_line]*len(values), 'r--')
        ax.text(len(values)/2, const_val_line, 'Maximum Possible Value: %0.2f' % const_val_line,
                verticalalignment='bottom', horizontalalignment='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if tight_layout:
        plt.tight_layout()
    plt.show()
    return


def plot_wordcloud(word_frequency_dict, num_words, figsize):
    # TODO: add option for max_font_size
    # eng_stop = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/stopwords_extended.npy")
    # fig, ax = plt.subplots(figsize=figsize)
    # wc = WordCloud(max_words=num_words, stopwords=eng_stop, prefer_horizontal=1.0, max_font_size=40)\
    #     .generate_from_frequencies(word_frequency_dict)
    # ax.imshow(wc, interpolation='bilinear')
    # ax.axis("off")
    # ax.set_title("Wordcloud")
    # plt.tight_layout()
    # plt.show(ax)
    return


def plot_wordcloud_pair(word_frequency_dict1, word_frequency_dict2, num_words, figsize):
    # TODO: see to figsize, add option for max_font_size
    # eng_stop = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/stopwords_extended.npy")
    # wordcloud1 = WordCloud(max_words=num_words, stopwords=eng_stop, max_font_size=40).\
    #     generate_from_frequencies(word_frequency_dict1)
    # plt.subplot(1, 2, 1)
    # plt.imshow(wordcloud1, interpolation='bilinear')
    # plt.axis("off")
    # plt.subplot(1, 2, 2)
    # wordcloud2 = WordCloud(max_words=num_words, stopwords=eng_stop, max_font_size=40).\
    #     generate_from_frequencies(word_frequency_dict2)
    # plt.imshow(wordcloud2, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()
    return


def plot_text_intensity_plot(groups, cmaps, title, save=False, show=True, filename=None, show_weight=False):
    # groups.sort(key=itemgetter(1))
    group1 = groups[:int(len(groups)/2)]
    num_groups = len(group1)
    if len(cmaps) < num_groups:
        raise ValueError("Insufficient cmap values for given number of topics to be plotted.")
    wordspergroup = len(group1[0])
    # fig, ax = plt.subplots(figsize=(num_groups*2 + 0.5, 7))
    ax = plt.subplot(211)
    ax.margins(tight=True)
    ax.axis([8.5, 10 * 2 - 8.75, 0, wordspergroup + 1])
    for i, group in enumerate(group1):
        assert (wordspergroup == len(group)), "Different sized groups in topics to display"
        cmap = plt.get_cmap(cmaps[i])
        max_size = group[0][1]
        min_size = group[-1][1]
        color_intensities = [0.6*(tup[1]-min_size)/(max_size-min_size)+0.5 for tup in group]
        words = [tup[0] for tup in group]
        for j, word in enumerate(words):
            if show_weight:
                word = word + " (%0.3f)" % group[j][1]
            ax.text(i*2.5, len(words)-j, word, verticalalignment='bottom', horizontalalignment='left',
                    color=cmap(color_intensities[j]), fontsize=15)
    plt.axis("off")
    plt.title(title, size='medium')

    group2 = groups[int(len(groups) / 2):]
    num_groups = len(group2)
    if len(cmaps) < num_groups:
        raise ValueError("Insufficient cmap values for given number of topics to be plotted.")
    wordspergroup = len(group2[0])
    # fig, ax = plt.subplots(figsize=(num_groups*2 + 0.5, 7))
    ax = plt.subplot(212)
    ax.margins(tight=True)
    ax.axis([8.5, 10 * 2 - 8.75, 0, wordspergroup + 1])
    for i, group in enumerate(group2):
        assert (wordspergroup == len(group)), "Different sized groups in topics to display"
        cmap = plt.get_cmap(cmaps[i])
        max_size = group[0][1]
        min_size = group[-1][1]
        color_intensities = [0.6 * (tup[1] - min_size) / (max_size - min_size) + 0.5 for tup in group]
        words = [tup[0] for tup in group]
        for j, word in enumerate(words):
            if show_weight:
                word = word + " (%0.3f)" % group[j][1]
            ax.text(i * 2.5, len(words) - j, word, verticalalignment='bottom', horizontalalignment='left',
                    color=cmap(color_intensities[j]), fontsize=15)

    plt.axis("off")
    if show:
        plt.tight_layout(pad=0.75, h_pad=0.1, w_pad=0.1)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    if save:
        assert (filename is not None), "No filename given for saving the topic topwords plot"
        plt.savefig(filename, bbox_inches='tight')
    # plt.close(fig)
    plt.close()
    return
