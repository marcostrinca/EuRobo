import math
import os, sys
import random
import string
import collections
import numpy as np
import numpy.random as rnd
import tensorflow as tf
from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def plot_with_labels(low_dim_embs, labels):
	# assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	plt.figure(figsize=(18, 18))  #in inches
	print(len(labels))
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i,:]
		plt.scatter(x, y)
		plt.annotate(label,
					 xy=(x, y),
					 xytext=(5, 2),
					 textcoords='offset points',
					 ha='right',
					 va='bottom')

	plt.show()


# import text
text = open('data/_Portugue_Literature_Dataset.txt', encoding='utf8').read()

def get_vocab(text):
	# se der erro de charmap no windows tem que user o comando chcp 65001 pra deixar o console em utf8
	words = text.split()
	no_hiphen = [i.replace('-','') for i in words]
	no_point_and_coma =  [i.replace(';',',') for i in words]
	return no_point_and_coma

words = get_vocab(text)
vocabulary_size = len(words)

def build_dataset(words, vocabulary_size):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
# data - texto substituindo a palavra por um id
# count - lista das palavras que mais aparecem para as que menos aparecem
# dictionary - dicionário com key palavra e value id
# reverse dictionary - o oposto do dicionário

import random
from collections import deque

def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1 # [ skip_window target skip_window ]
	buffer = deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = skip_window  # target label at the center of the buffer
		targets_to_avoid = [ skip_window ]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels

data_index=0
batch, labels = generate_batch(8, 2, 1)
print(batch, [reverse_dictionary[word] for word in batch])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = rnd.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

learning_rate = 0.01

# Input data.
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Look up embeddings for inputs.
init_embeddings = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeddings)
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.
loss = tf.reduce_mean(
	tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
				   num_sampled, vocabulary_size))

# Construct the Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Add variable initializer.
init = tf.global_variables_initializer()
saver = tf.train.Saver()

num_steps = 10001

with tf.Session() as session:
	#init.run()
	saver.restore(session, 'Data/rnn/final_embeddings.ckpt')

	average_loss = 0
	for step in range(num_steps):
		print("\rIteration: {}".format(step), end="\t")
		batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

		# We perform one update step by evaluating the training op (including it
		# in the list of returned values for session.run()
		_, loss_val = session.run([training_op, loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
			# The average loss is an estimate of the loss over the last 2000 batches.
			print("Average loss at step ", step, ": ", average_loss)
			average_loss = 0
			save_path = saver.save(session, 'Data/rnn/embeddings.ckpt')

		# Note that this is expensive (~20% slowdown if computed every 500 steps)
		# if step % 10000 == 0:
		#     sim = similarity.eval()
		#     for i in range(valid_size):
		#         valid_word = reverse_dictionary[valid_examples[i]]
		#         top_k = 8 # number of nearest neighbors
		#         nearest = (-sim[i, :]).argsort()[1:top_k+1]
		#         print("nearest: ", nearest)
		#         log_str = "Nearest to %s:" % valid_word
		#         for k in range(top_k):
		#             cw = nearest[k]
		#             print(cw, [reverse_dictionary[cw]])
		#             close_word = [reverse_dictionary[cw]]
		#             log_str = "%s %s," % (log_str, close_word)
		#         print(log_str)

	final_embeddings = normalized_embeddings.eval()

	np.save("Data/rnn/my_final_embeddings.npy", final_embeddings)
	save_path = saver.save(session, 'Data/rnn/final_embeddings.ckpt')

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	plot_only = 4000
	low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
	labels = [reverse_dictionary[i] for i in range(plot_only)]
	print("Plotting...")
	plot_with_labels(low_dim_embs, labels)

