from __future__ import division
# Number of passes over the data if num documents are less
passes_low = 5
# Number of passes over the data if num documents are moderate
passes_mid = 3
# Number of passes over the data if num documents are high
passes_high = 1
# Sparsity of document-topic prior
alpha = 'symmetric'
# prior over topic word distribution
eta = None
# number of iterations per pass
iterations = 100
# --
gamma_threshold = 0.001
# controls filtering the topics returned for a document (bow)
minimum_probability = 0.01
# seeding
random_state = 1
# --
minimum_phi_value = 0.01
# evaluation of perplixity
eval_every = None
