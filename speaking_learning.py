import math
import os
import random
import string
import collections
import tensorflow as tf

# import text
text = open('data/_Portugue_Literature_Dataset.txt', encoding='utf8').read()

def get_vocab(text):
	# se der erro de charmap no windows tem que user o comando chcp 65001 pra deixar o console em utf8
	words = text.split()
	no_hiphen = [i.replace('-','') for i in words]
	return no_hiphen

words = get_vocab(text)

##### embeddings
vocabulary_size = len(words)
embedding_size = 150
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

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

del words  # Hint to reduce memory.

print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


####################################
### Approach 1 - treino uma lista de palavras como input e a próxima como output
SEQLEN = 5
STEP = 1
input_words = []
label_words = []

for i in range(0, vocabulary_size - SEQLEN, STEP):
    input_words.append(data[i:i + SEQLEN])
    label_words.append(data[i + SEQLEN])

### Test
n_steps = 50
n_neurons = 200
n_layers = 3
num_encoder_symbols = 20000
num_decoder_symbols = 20000
embedding_size = 150
learning_rate = 0.01