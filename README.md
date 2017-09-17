# Topiceval

Topiceval is a python library with functionalities to learn topic models (namely gensim LDA, Thresholded SVD, Sparse Coding - Sparse Dictionary Learning) over user mails stored in Outlook with an end interface for user-study on topic quality evaluation, as well as providing topic model classes for usage on independent data-sets. 

Added functions include ability to run diagnostics on learned models, which include visualizations on topic-document distribution, topic entropy matrix, topic weights required to cross thresholds etc. 

# Requires
- python 3.x or python >= 2.6
- For user-email evaluation - Windows and Outlook

# Dependencies
- scipy>=0.19.0 
- gensim 
- numpy
- matplotlib 
- pandas
- scikit-learn 
- nltk (optional)

# Installation

    $ pip install topiceval
    or
    Download from source and "$ python setup.py install" (not yet tested)

# Usage 
## User Email Study

    $ python -m topiceval

You can also add options:
  - -numtopics <int>: Number of topics to learn 
  - -usethreads <0/1>: Whether to treat each message in threaded conversation as different document (0) or not (1)
  - -uselemma <0/1>: Whether to lemmatize words, set to 1 only if you have downloaded wordnet corpora for nltk package.
  - -numtopicseval <int>: How many topics out of the learned ones to be shown to the user for evaluation.
  - excludefolders <str>: Comma separated name of folders to exclude during mail extraction
  - makeWtsvd <0/1>: Whether to make W matrix for TSVD, setting at 1 not yet tested

For example, to learn 20 topics from user mails using lemmatization, excluding folders named 'private' & 'work', and displaying 10 topics for evaluation, you have to use:
    
    $ python -m topiceval -numtopics 20 -numtopicseval 10 -excludefolders private,work -uselemma 1

### User Evaluation Interface
![](https://raw.githubusercontent.com/Avikalp7/CG-Accumulator/master/Screenshots/email_interface.PNG)

## TSVD Topic Model

### Case 1
You have the docword file and vocab file.
"docword file" is in the format *doc_id word_id word_freq* per each line. Both *doc_id* and *word_id* are 1-indexed
"vocab file" has a vocabulary word in each line, where line number is the word_id.

    from topiceval.thresholdedsvd.tsvd import TSVD
    
    model = TSVD(docword_filepath="<your_docword_filepath>", vocab_filepath="<your_vocab_filepath>",    num_topics = 20)
    
    learned_M_matrix = model.M_matrix
    topic_catchwords = model.topicwise_catchwords
    
    model.save_topic_top_words(filename="tsvdtopics.txt")
    model.plot_topic_topwords(topicid_list=range(10))
    model.plot_entropy_distribution()
    model.plot_topic_entropy_colormap()
    
    model.save_M_matrix(filename="<desired_filepath>")

### Outputs
The following are outputs for TSVD models on NIPS dataset from UCI BoW repository
- Snap from text file made by model.save_topic_topwords:
![](https://raw.githubusercontent.com/Avikalp7/CG-Accumulator/master/Screenshots/tsvd_save_topic_topwords.PNG)

- Plot from model.plot_topic_topwords
![](https://raw.githubusercontent.com/Avikalp7/CG-Accumulator/master/Screenshots/tsvd_plot_topic_topwords.PNG)

- Plot from model.plot_entropy_distribution
![](https://raw.githubusercontent.com/Avikalp7/CG-Accumulator/master/Screenshots/tsvd_plot_entropy_distribution.PNG)

- Plot from model.plot_topic_entropy_colormap
![](https://raw.githubusercontent.com/Avikalp7/CG-Accumulator/master/Screenshots/tsvd_plot_topic_entropy_colormap.PNG)

### Case 2
You have corpus in matrix-market format

    from topiceval.thresholdedsvd.tsvd import TSVD
    
    model = TSVD(corpus_filepath="<your_corpus_filepath>", vocab_filepath="<your_vocab_filepath>",      num_topics = 20)

or, if you have loaded corpus in your program, just use argument *corpus=<your_corpus_variable>*

### Case 3
You have made the A matrix

    from topiceval.thresholdedsvd.tsvd import TSVD
    
    model = TSVD(A_matrix_path="<your_Amatrix_filepath>", vocab_filepath="<your_vocab_filepath>",       num_topics = 20)
or, if you have loaded Amatrix in your program, just use argument *A_matrix=<your_Amatrix_variable>*

## BCD Topic Model

### Case 1
You have the docword file and vocab file.

    from topiceval.dictionarylearning.bcd import BCD
    
    model = BCD(docword_filepath="<your_docword_filepath>", vocab_filepath="<your_vocab_filepath>",     num_topics = 20)
    
    # and sparsity, num iterations can be controlled by:
    model = BCD(docword_filepath="<your_docword_filepath>", vocab_filepath="<your_vocab_filepath>",     num_topics = 20, gamma_frac=0.35, nu_frac=0.15, bcd_iters=10)

### Case 2
You have corpus in matrix-market format

    from topiceval.thresholdedsvd.tsvd import TSVD
    
    model = BCD(corpus_filepath="<your_corpus_filepath>", vocab_filepath="<your_vocab_filepath>",       num_topics = 20)

or, if you have loaded corpus in your program, just use argument *corpus=<your_corpus_variable>*

### Case 3
You have made the X matrix (num_docs x vocab) or A matrix (vocab x num_docs)

    from topiceval.thresholdedsvd.tsvd import TSVD
    
    # for A matrix
    model = BCD(A_matrix_path="<your_Amatrix_filepath>", vocab_filepath="<your_vocab_filepath>",        num_topics = 20)
    # for X matrix
    model = BCD(X_matrix_path="<your_Xmatrix_filepath>", vocab_filepath="<your_vocab_filepath>",        num_topics = 20)
    
or, if you have loaded Amatrix in your program, just use argument *A_matrix=<your_Amatrix_variable>*. Similarly for X matrix.