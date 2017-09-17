"""
This module use the win32com.client library for python to extract user emails
from outlook and organize them in a pandas dataframe structure.
"""

from __future__ import division
from __future__ import print_function

from topiceval.preprocessing import textcleaning
import topiceval.preprocessing.emailsprocess as emailprocess
# from topiceval import config
from topiceval import makewordvecs

import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from six.moves import xrange

import win32com.client
import multiprocessing as mp
import os
import logging
import pickle

logger = logging.getLogger(__name__)


def extract_usermails(threaded, num_topics, reuse, excludefolders):
    """
    Extract mails from inbox, store as a dataframe and return the directory name and the dataframe.

    The steps followed are:

    1. Make a data directory at current location to store temporary data

    2. Extract user mails from outlook using win32.client API. Always extracts from all folders except
        some default one. Private folders that the users want not to be explored can be mentioned
        explicitly via the command-line. These are stored in a dataframe with their meta-data information.
        Additional fields like message replied to or not, message importance etc are also appended.

    3. If threaded option is True, then from all mails belonging to a single mail thread, only
        the largest is retained and the rest are discarded.
        If threaded is False, then previous conversation content is removed from all mails, maintaining
        only the most recent message in the extracted mail => all mails in a conversation form separate
        docs

    4. Signature Removal: Some common phrases that form signatures are removed

    5. The clean_text() function of textcleaning.py module is applied. It tokenizes text, removes special
        characters, identifies metadata info, url's, email addresses, numbers, money figures, weekdays etc.,
        along with additional option of lemmatization. Some effort is made to remove template text by removing
        information that folllows multiple special character of the same kind (like --------)

    6. Phrase Detection: Gensim's phrase detection module is used to form bigrams # TODO: Trigrams etc

    7. Next step is removal of stopwords. The popular extended list of stopwords available online is used
        for this task since we aim at removal of topically poor words.

    Parameters
    ----------
    :param threaded: bool, if True, treats threaded conversation as a single document
    :param num_topics: number of topics for topic model, used for setting dimensions for word2vec learning
    :param reuse: bool, if True, stores extracted email items as pickle file in current directory for reuse
    :param excludefolders: string, comma separated list of extra folders to exclude

    :return: string, pandas.DataFrame corresponding to the directory path to store temporary data
        and dataframe holding user's email information
    """
    ''' Make data directory at current location for temporary storage of data '''
    dirname = make_user_dir()
    ''' Get email items from inbox, sent_items and any extrafolders mentioned through command line
        Currently configured to use 2 processes for parallel extraction of mails from inbox and sent mails
        extrafolders are traversed serially '''
    # TODO: Turn on multiprocessing after sufficient testing and benchmarking

    # preload_flag is used to determine whether email items are load from previously saved items (flag=True)
    # or freshly extracted from outlook
    preload_flag = False
    # If items file exists in current directory, load mails from that instead of extracting again
    if os.path.isfile("./topiceval_items.pickle"):
        try:
            with open("./topiceval_items.pickle", "rb") as handle:
                items = pickle.load(handle)
            preload_flag = True
        except Exception as e:
            logger.error("COULD NOT LOAD FROM PICKLE FILE, ERROR: {}".format(e))
            items = extract(excludefolders, use_multiprocessing=False)
    else:
        items = extract(excludefolders, use_multiprocessing=False)

    logger.info("Total number of user emails extracted : %d" % len(items))

    # If reuse option is on, we save the extracted mails to a file in the current directory
    if reuse and not preload_flag:
        logger.debug("reuse option is on, saving items in current directory...")
        try:
            with open("./topiceval_items.pickle", "wb") as handle:
                pickle.dump(items, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            os.remove("./topiceval_items.pickle")
            logger.error("COULD NOT STORE EMAIL ITEMS, PICKLE ERROR: {}".format(e))
    preload_flag = False

    # If reuse option is on and dataframe file and wordvecs are present, load them, else construct them
    if reuse and os.path.isfile("./df.pkl") and os.path.isfile("./wordvecs"):
        print("Reading presaved dataframe and trained wordvecs..")
        df = pd.read_pickle('./df.pkl')
        wordvecs = KeyedVectors.load('./wordvecs')
        preload_flag = True
    else:
        # df is a pandas dataframe consisting of various email header information with the email body
        # wordvecs is a gensim.models.keyedvectors.KeyedVector instance
        df, wordvecs = makedf(items, threaded, num_topics)

    if not preload_flag and reuse:
        df.to_pickle('./df.pkl')
        wordvecs.save('./wordvecs')

    # A is the doc-term BoW matrix, email_network is a preprocessing.emailstructure.EmailNetwork instance
    A, email_network = emailprocess.make_doc2bow(df, dirname, threaded, wordvecs)

    # Compute the replied-to-or-not boolean field and append to the email_network's dataframe
    # email_network.df = emailprocess.add_reply_field(email_network.df, email_network)

    # Compute the combined to-cc-bcc field and append to the email_network's dataframe
    email_network.df = emailprocess.add_to_cc_bcc_field(email_network.df)
    # Compute the num_days from today field and append to the email_network's dataframe
    email_network.df["diff"] = (pd.datetime.now() - email_network.df['SentOn'])
    # Compute the email importance field and append to the email_network's dataframe
    email_network.make_user_importance_score_dict()
    email_network.make_importance_field()
    # Make the email_network object compute some extra things now that email importance and other fields are available
    email_network.make_three_imp_folders()
    email_network.make_three_time_periods()
    return dirname, email_network, A


def encodeit(s):
    if isinstance(s, str):
        return (s.encode('utf-8')).decode('utf-8')
    else:
        return s


def extract_helper(messages, folder_type, items, max_items):
    message = messages.GetLast()
    i = 0
    while message and i < max_items:
        try:
            d = dict()
            d['Subject'] = encodeit(getattr(message, 'Subject', '<UNKNOWN>'))
            d['SentOn'] = str(encodeit(getattr(message, 'SentOn', '<UNKNOWN>')))
            d['SenderName'] = encodeit(getattr(message, 'SenderName', '<UNKNOWN>'))
            d['CC'] = encodeit(getattr(message, 'CC', '<UNKNOWN>'))
            d['BCC'] = encodeit(getattr(message, 'BCC', '<UNKNOWN>'))
            d['To'] = encodeit(getattr(message, 'To', '<UNKNOWN>'))
            d['Body'] = encodeit(getattr(message, 'Body', '<UNKNOWN>'))
            d['ConversationID'] = encodeit(getattr(message, 'ConversationID', '<UNKNOWN>'))
            d['ConversationIndex'] = encodeit(getattr(message, 'ConversationIndex', '<UNKNOWN>'))
            d['UnRead'] = encodeit(getattr(message, 'UnRead', '<UNKNOWN>'))
            d['FolderType'] = folder_type
            if d['SentOn'] != '<UNKNOWN>' and d['ConversationID'] != '<UNKNOWN>':
                i += 1
                items.append(d)
        except Exception as inst:
            print("Error processing mail", inst)

        message = messages.GetPrevious()
    return items


def extract_helper_mp(folder_num, folder_type):
    items = []
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    messages = outlook.GetDefaultFolder(folder_num).Items
    message = messages.GetFirst()
    while message:
        try:
            d = dict()
            d['Subject'] = encodeit(getattr(message, 'Subject', '<UNKNOWN>'))
            d['SentOn'] = str(encodeit(getattr(message, 'SentOn', '<UNKNOWN>')))
            d['SenderName'] = encodeit(getattr(message, 'SenderName', '<UNKNOWN>'))
            d['CC'] = encodeit(getattr(message, 'CC', '<UNKNOWN>'))
            d['BCC'] = encodeit(getattr(message, 'BCC', '<UNKNOWN>'))
            d['To'] = encodeit(getattr(message, 'To', '<UNKNOWN>'))
            d['Body'] = encodeit(getattr(message, 'Body', '<UNKNOWN>'))
            d['ConversationID'] = encodeit(getattr(message, 'ConversationID', '<UNKNOWN>'))
            d['ConversationIndex'] = encodeit(getattr(message, 'ConversationIndex', '<UNKNOWN>'))
            d['UnRead'] = encodeit(getattr(message, 'UnRead', '<UNKNOWN>'))
            d['FolderType'] = folder_type
            if d['SentOn'] != '<UNKNOWN>' and d['ConversationID'] != '<UNKNOWN>':
                items.append(d)
        except Exception as inst:
            print("Error processing mail", inst)

        message = messages.GetNext()
    return items


def extract(excludefolders, use_multiprocessing=False):
    items = []
    folders = {"inbox": 6, "sent_items": 5}     # append "deleted_items": 3 if wanted
    ''' Mails from folders in default folder set won't be extracted during extraction from extra-folders. 
       "archive" has been removed from default folder set and will be extracted unless excluded '''
    default_folder_set = {'deleted items', 'inbox', 'outbox', 'sent items', 'personmetadata', 'tasks', 'junk email',
                          'drafts', 'calendar', 'rss subscriptions', 'quick step settings', 'yammer root',
                          'conversation action settings', 'externalcontacts', 'important', 'journal', 'files',
                          'contacts', 'conversation history', 'social activity notifications', 'sync issues', 'notes',
                          'reminders',
                          'the file so that changes to the file will be reflected in your item.'}
    exclude_folder_set = set([foldername.lower() for foldername in excludefolders.split(',')])

    if use_multiprocessing:
        logger.log(0, "Using multiprocessing while extracting mails...")
        folders_mp = [tup for tup in folders.items()]
        num_processes = len(folders_mp)
        p = mp.Pool(processes=num_processes)
        for i in xrange(num_processes):
            res = p.apply_async(extract_helper_mp, args=(folders_mp[i][1], folders_mp[i][0]))
            items.extend(res.get())
        p.close()
        p.join()

    else:
        outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
        for folder_type in folders:
            logger.debug("Extracting emails from {0}".format(folder_type))
            folder = outlook.GetDefaultFolder(folders[folder_type])
            messages = folder.Items
            logger.debug("extacting messages...")
            items = extract_helper(messages=messages, folder_type=folder_type, items=items, max_items=6000)
            for subfolder in folder.Folders:
                logger.debug("Extracting emails from subfolder {0}".format(str(subfolder)))
                messages = subfolder.Items
                items = extract_helper(messages=messages, folder_type=str(subfolder), items=items, max_items=6000)
            logger.debug("done extacting messages")

    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    all_folders = [folder for folder in outlook.Folders[0].folders]
    extrafolders = [folder for folder in all_folders if (str(folder).lower() not in default_folder_set
                                                         and str(folder).lower() not in exclude_folder_set)]
    if len(extrafolders) > 0:
        for folder in extrafolders:
            logger.debug("Extracting emails from {0}".format(folder))
            messages = folder.Items
            items = extract_helper(messages=messages, folder_type=str(folder), items=items, max_items=600)
            for subfolder in folder.Folders:
                logger.debug("Extracting emails from subfolder {0}".format(str(subfolder)))
                messages = subfolder.Items
                items = extract_helper(messages=messages, folder_type=str(subfolder), items=items, max_items=500)
            if len(items) > 35000:
                break
    return items


def make_user_dir():
    path = "./data_topiceval/userdata/"
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        os.makedirs(path)
    logger.debug("Saving temp data at {0}{1}".format(os.path.dirname(os.path.abspath(__file__)), path[1:]))
    return path


def makedf(items, threaded, num_topics):
    # items.sort(key=lambda tup: tup['SentOn'])
    keys = ["ConversationID", "SentOn", "SenderName", "To", "CC", "BCC", "Subject", "Body", "UnRead",
            "FolderType"]
    df = pd.DataFrame()
    logger.log(0, "Making user-emails' dataframe...")
    for key in keys:
        try:
            df[key] = [str(d[key]) for d in items]
        except Exception as e:
            logger.log(0, "Exception: {}".format(e))
            df[key] = [d[key].encode('utf-8') for d in items]
    df.set_index(keys=["ConversationID"], inplace=True, drop=False)
    # df = df[(~df["Subject"].str.contains("sent you a message in Skype for Business")) &
    #         (~df["Subject"].str.contains("I've shared files with you")) &
    #         (~df["Subject"].str.contains("Missed conversation with"))]
    df['SentOn'] = pd.to_datetime(df['SentOn'], infer_datetime_format=True)
    # df = df[~(df['Subject'].isin(['sent you a message in Skype for Business', 'I\'ve shared files with you',
    #                              'Missed conversation with']))].copy()
    df = df[~df["Subject"].str.contains("I've shared files with you")]
    df = df[~df["Subject"].str.contains("sent you a message in Skype for Business")]
    df = df[~df["Subject"].str.contains("Missed conversation with")]
    # df = df[~df["Subject"].str.contains("I've shared files with you")]
    # df = df[~df["Subject"].str.contains("Missed conversation with")]
    df.sort_values(by=['ConversationID', 'SentOn'], ascending=[True, True], inplace=True)
    logger.log(0, "Cleaning email bodies in dataframe...")
    # names = get_names(df)
    if threaded:
        bool_list = emailprocess.remove_redundant_threads(df)
        df = df[bool_list]
    else:
        df['Body'] = df['Body'].apply(emailprocess.remove_threads)
    df['Body'] = df['Body'].apply(emailprocess.remove_signature)
    bigram_phraser = emailprocess.phrase_detection(df)
    df['CleanBody'] = df['Body'].apply(textcleaning.clean_text)
    ''' Adding subject to clean body '''
    # for idx, row in df[["Subject", "CleanBody"]].iterrows():
    #     cleanbody = row[1] + textcleaning.clean_text(row[0]) * config.ADD_SUBJECT_TO_BODY
    #     df.set_value(idx, "CleanBody", cleanbody)
    df['CleanBody'] = df['CleanBody'].apply(emailprocess.phraser, args=(bigram_phraser,))
    word_vecs = makewordvecs.make_wordvecs(df["CleanBody"], num_topics)
    stopwords = emailprocess.load_stops()
    df['CleanBody'] = df['CleanBody'].apply(textcleaning.remove_stops, args=(stopwords, ))
    df['Subject'] = df['Subject'].apply(emailprocess.clean_email_header)
    df['To'] = df['To'].apply(emailprocess.clean_email_header)
    df['CC'] = df['CC'].apply(emailprocess.clean_email_header)
    df['BCC'] = df['BCC'].apply(emailprocess.clean_email_header)
    df = emailprocess.add_reply_field(df)
    # df['CleanBody'] = df['CleanBody'].apply(emailprocess.replace_names, args=(names, ))
    logger.log(0, "Done cleaning email bodies in dataframe")
    return df, word_vecs


def get_names(df):
    return df["SenderName"].unique().tolist()


# def make_emails_dict(df):
#     emails_dict = {}
#     for elem in df.index.unique():
#         emails_dict[elem] = []
#     for idx, row in df.iterrows():
#         emails_dict[row["ConversationID"]].append((row["ConversationIndex"], row["Body"]))
#     return emails_dict
