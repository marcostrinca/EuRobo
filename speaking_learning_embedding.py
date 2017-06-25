import math
import os, sys
import random
import string
import collections
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time

start_time = time.time()
def elapsed(sec):
	if sec<60:
		return str(sec) + " sec"
	elif sec<(60*60):
		return str(sec/60) + " min"
	else:
		return str(sec/(60*60)) + " hr"

# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

# import text
text = open('Data/_Portugue_Literature_Dataset.txt', encoding='utf8').read()

###
def get_vocab(text):
	# se der erro de charmap no windows tem que user o comando chcp 65001 pra deixar o console em utf8
	words = text.split()
	no_hiphen = [i.replace('-','') for i in words]
	no_point_and_coma =  [i.replace(';',',') for i in words]
	return no_point_and_coma[:100]

training_data = get_vocab(text)
vocab_size = len(training_data)

###
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

print(vocab_size)

### GRAPH
learning_rate = 0.0001
n_hidden = 300
embedding_size = 150
n_iterations = 2400000
n_inputs = 10 # qdd de palavras no inputs
batch_size = 50 # qdd de vezes que n_inputs vai ser processada a cada iteracao

# Placeholders
x = tf.placeholder(tf.int32, [None, n_inputs, 1])
seqlen = tf.placeholder(tf.int32, [batch_size])
y = tf.placeholder(tf.float32, [None, vocab_size])
keep_prob = tf.constant(1.0) # for dropout

# RNN output node weights and biases
weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
	'out': tf.Variable(tf.random_normal([vocab_size]))
}

# Embedding layer
embeddings = tf.get_variable('embedding_matrix', [vocab_size, embedding_size])
rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

# reshape to [1, n_input]
rnn_inputs = tf.reshape(x, [-1, n_inputs])

# Generate a n_input-element sequence of inputs
# (eg. [had] [a] [general] -> [20] [6] [33])
rnn_inputs = tf.split(x,n_inputs,1)

# RNN
cell = rnn.GRUCell(n_hidden)
init_state = tf.get_variable('init_state', [1, n_hidden], initializer=tf.constant_initializer(0.0))
init_state = tf.tile(init_state, [batch_size, 1])
# rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen, initial_state=init_state)


# Add dropout, as the model otherwise quickly overfits
rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)	

# we just want the last ouput
tf.matmul(rnn_outputs[-1], weights['out']) + biases['out']














