from __future__ import division
import warnings
import logging
import argparse
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

logger = logging.getLogger('topiceval')
logger.setLevel(logging.DEBUG)

# create console handler with debug log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
FORMAT = "[%(asctime)s][%(filename)s - %(funcName)10s()] %(message)s"
formatter = logging.Formatter(FORMAT, datefmt="%I:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

parser = argparse.ArgumentParser(description='topic-model-evaluation-package')
parser.add_argument('-numtopics', help='specify number of topics to learn', type=int, default=100)
parser.add_argument('-numtopicseval', help='specify number of topics to evaluate per model', type=int, default=15)
parser.add_argument('-usethreads', help='whether to use threaded mails or single ones', type=int, default=0)
parser.add_argument('-makeWtsvd', help='whether to make W matrix', type=int, default=1)
parser.add_argument('-excludefolders', help='comma separated folder names to exclude', type=str, default="")
parser.add_argument('-skipeval', help='skip user evaluation part', type=int, default=1)
parser.add_argument('-reuse', help='store items for reuse', type=int, default=0)

args = vars(parser.parse_args())

num_topics = args['numtopics']
num_topics_eval = args['numtopicseval']
threaded = args['usethreads']

# create file handler which logs only info messages
logging_filename = './topiceval_results_%dthreaded_%dtopics_%dtopicsEval.log' % (threaded, num_topics, num_topics_eval)
fh = logging.FileHandler(logging_filename, 'w')
# FORMAT = "[%(filename)s - %(funcName)10s()] %(message)s"
FORMAT = "[%(asctime)s][%(filename)s - %(funcName)10s()] %(message)s"
formatter = logging.Formatter(FORMAT)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)
