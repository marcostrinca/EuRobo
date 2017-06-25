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

def get_vocab(text):
	# se der erro de charmap no windows tem que user o comando chcp 65001 pra deixar o console em utf8
	words = text.split()
	no_hiphen = [i.replace('-','') for i in words]
	no_point_and_coma =  [i.replace(';',',') for i in words]
	return no_point_and_coma[:100]

training_data = get_vocab(text)
vocab_size = len(training_data)

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
n_iterations = 1000
n_inputs = 10 # qdd de palavras no inputs
batch_size = 50 # qdd de vezes que n_inputs vai ser processada a cada iteracao


x = tf.placeholder(tf.int32, [None, n_inputs, 1]) 
y = tf.placeholder(tf.int32, [None, vocab_size])

W = tf.get_variable('W', [n_hidden, vocab_size])
b = tf.get_variable('b', [vocab_size], initializer=tf.constant_initializer(0))

# Embedding layer
embedding_size = 150
embeddings = tf.get_variable('embedding_matrix', [vocab_size, embedding_size])
rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

def RNN(rnn_inputs):

	# reshape to [1, n_input]
	rnn_inputs = tf.reshape(rnn_inputs, [-1, n_inputs])

	# Generate a n_input-element sequence of inputs
	# (eg. [had] [a] [general] -> [20] [6] [33])
	rnn_inputs = tf.split(rnn_inputs, n_inputs, 1)

	# 2-layer LSTM, each layer has n_hidden units.
	# Average Accuracy= 95.20% at 50k iter
	rnn_cell = rnn.GRUCell(n_hidden)

	# generate prediction
	outputs, states = rnn.static_rnn(rnn_cell, rnn_inputs, dtype=tf.int32)

	# there are n_input outputs but
	# we only want the last output
	return tf.matmul(outputs[-1], W) + b

logits = RNN(rnn_inputs)
preds = tf.nn.softmax(logits)

# Model evaluation
correct = tf.equal(tf.argmax(preds,1), y)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)



### EMBEDDINGS: HOW TO USE THIS!?
# Look up embeddings for inputs.
# embedding_size = 128  # Dimension of the embedding vector.
# init_embeddings = tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
# embeddings = tf.Variable(init_embeddings)
# embed = tf.nn.embedding_lookup(embeddings, train_in puts)
# Compute the cosine similarity between minibatch examples and all embeddings.
# norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
# normalized_embeddings = embeddings / norm
# valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
# similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as session:
	session.run(init)
	# saver.restore(session, 'Data/rnn/final_rnn.ckpt')

	step = 0
	display_step = 1000
	offset = random.randint(0,n_inputs+1)
	end_offset = n_inputs + 1
	acc_total = 0
	loss_total = 0

	writer.add_graph(session.graph)

	while step < n_iterations:

		# Generate a minibatch. Add some randomness on selection process.
		if offset > (len(training_data)-end_offset):
			offset = random.randint(0, n_inputs+1)

		symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_inputs) ]
		symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_inputs, 1])

		symbols_out_onehot = np.zeros([vocab_size], dtype=float)
		symbols_out_onehot[dictionary[str(training_data[offset+n_inputs])]] = 1.0
		symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

		_, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
												feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
		loss_total += loss
		acc_total += acc
		if (step+1) % display_step == 0:
			print("Iter= " + str(step+1) + ", Average Loss= " + \
				  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
				  "{:.2f}%".format(100*acc_total/display_step))
			acc_total = 0
			loss_total = 0
			symbols_in = [training_data[i] for i in range(offset, offset + n_inputs)]
			symbols_out = training_data[offset + n_inputs]
			symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
			print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
		step += 1
		offset += (n_inputs+1)

	save_path = saver.save(session, 'Data/rnn/final_rnn.ckpt')

	print("Optimization Finished!")
	print("Elapsed time: ", elapsed(time.time() - start_time))
	print("Run on command line.")
	print("\ttensorboard --logdir=%s" % (logs_path))
	print("Point your web browser to: http://localhost:6006/")
	while True:
		prompt = "%s words: " % n_inputs
		sentence = input(prompt)
		sentence = sentence.strip()
		words = sentence.split(' ')
		if len(words) != n_inputs:
			continue
		try:
			symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
			for i in range(8):
				keys = np.reshape(np.array(symbols_in_keys), [-1, n_inputs, 1])
				onehot_pred = session.run(pred, feed_dict={x: keys})
				onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
				sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
				symbols_in_keys = symbols_in_keys[1:]
				symbols_in_keys.append(onehot_pred_index)
			print(sentence)
		except:
			print("Word not in dictionary")