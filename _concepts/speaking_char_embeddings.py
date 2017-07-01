import os, sys
import numpy as np
import random
import string, collections
import tensorflow as tf
import zipfile

# import text
text = open('Data/_Portugue_Literature_Dataset.txt', encoding='utf8').read()
print('Data size %d' % len(text))

# create a small validation and train sets
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_counter = collections.Counter(text).most_common()
# print(vocabulary_counter)

vocabulary_chars = sorted(list(set(text)))
vocabulary_size = len(vocabulary_chars)
print(vocabulary_size)
print(vocabulary_chars)

def char2id(char):
  if char in sorted(vocabulary_chars):
    return ord(char)
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid)
  else:
    return ' '

print(char2id("a")) # 97
print(id2char(97))

batch_size=64
num_unrollings=10

# class to generate batches
class BatchGenerator(object):
  def __init__(self, mytext, batch_size, num_unrollings):
    self._mytext = mytext
    self._mytext_size = len(mytext)
    print(self._mytext_size)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._mytext_size // batch_size
    print("segment:")
    print(segment)
    self._cursor = [ offset * segment for offset in range(batch_size)]
    print("cursor:")
    print(self._cursor)
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      print("self.cursor b")
      print(self._cursor[b])
      batch[b, char2id(self._mytext[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._mytext_size
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))