import numpy as np
import random
import tensorflow as tf
import datetime
import pickle

vocab_size = 10000
seed = 42
embedding_dim = 100

# import text
text = open('data/_Portugue_Literature_Dataset.txt', encoding='utf8').read()
print(text[:500])

from collections import Counter
from itertools import chain
def get_vocab(lst):
	vocab_count = Counter(w for txt in lst for w in txt.split())
	vocab = list(map(lambda x: x[0], sorted(vocab_count.items(), key=lambda x: -x[1])))
	return vocab, vocab_count

vocab, vocab_count = get_vocab(text)

for l in vocab[:50]:
	print(l.decode('utf8'))

print ('...', len(vocab))