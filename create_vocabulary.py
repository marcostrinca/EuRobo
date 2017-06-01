import numpy as np
import random
import tensorflow as tf
import datetime
import pickle

from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation

vocab_size = 10000
seed = 42
embedding_dim = 100

# import text
text = open('data/_Portugue_Literature_Dataset.txt', encoding='utf8').read()
# print(text[:500])

from collections import Counter
from itertools import chain
def get_vocab(lst):

	words = lst.split()
	vocab_count = Counter(words)
	vocab = list(map(lambda x: x[0], sorted(vocab_count.items(), key=lambda x: -x[1])))

	# this way return the characters
	# vocab_count = Counter(w for txt in lst for w in txt.split())
	# vocab = list(map(lambda x: x[0], sorted(vocab_count.items(), key=lambda x: -x[1])))
	return vocab, vocab_count

vocab, vocab_count = get_vocab(text)
# print(vocab[:50])
print ('qdd de palavras únicas:', len(vocab))


# some statistics on data
# import matplotlib.pyplot as plt
# plt.plot([vocab_count[w] for w in vocab]);
# plt.gca().set_xscale("log", nonposx='clip')
# plt.gca().set_yscale("log", nonposy='clip')
# plt.title('word distribution in headlines and discription')
# plt.xlabel('rank')
# plt.ylabel('total appearances');

######## INDEX WORDS
empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word

def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    
    idx2word = dict((idx,word) for word,idx in word2idx.items())

    return word2idx, idx2word

word2idx, idx2word = get_idx(vocab, vocab_count)
#print(word2idx)


###### INPUTS
total_words = text.split()
nb_words = len(total_words)
print('qdd total palavras:', nb_words)
# print(total_words[:50])

SEQLEN = 5
STEP = 1
input_words = []
label_words = []

for i in range(0, nb_words - SEQLEN, STEP):
    input_words.append(total_words[i:i + SEQLEN])
    label_words.append(total_words[i + SEQLEN])

for i in range(11, 20):
    print(input_words[i], "->", label_words[i])


######### VECTORIZE INPUTS
X = np.zeros((len(input_words), SEQLEN, nb_words), dtype=np.bool)
y = np.zeros((len(input_words), nb_words), dtype=np.bool)
for i, input_word in enumerate(input_words):
    for j, wd in enumerate(input_word):
        X[i, j, word2idx[wd]] = 1
    y[i, word2idx[label_words[i]]] = 1


####### BUILD THE MODEL
model = Sequential()
model.add(SimpleRNN(512, return_sequences=False, input_shape=(SEQLEN, nb_words)))
model.add(Dense(nb_words))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

batch_size = 128
for i in range(50):
    print("=" * 50)
    print("Iteration #: %d" % (i))
    model.fit(X, y, batch_size=batch_size, epoch=1)
    
    # testing, pick a sequence randomly as seed and use it to generate text from
    # model for the next 100 steps
    test_idx = np.random.randint(len(input_words))
    test_words = input_words[test_idx]
    print("Seed: %s" % (test_words))
    print(test_words, end="")
    for j in range(100):
        Xtest = np.zeros((1, SEQLEN, nb_words))
        for k, wd in enumerate(test_words):
            Xtest[0, k, word2idx[wd]] = 1
        pred = model.predict(Xtest, verbose=0)[0]
        ypred = idx2word[np.argmax(pred)]
        print(ypred, end="")
        # move forward with test_chars + ypred
        test_words = test_words[1:] + ypred
    print()





