import numpy as np
from gensim.models import Word2Vec, word2vec
import re

from keras.layers import Embedding, LSTM, Dense, TimeDistributed
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import time
import sys

from keras.regularizers import l2
from keras.utils import np_utils

import BIOF1Validation
import GermEvalReader


def read_wordvecs(filename):
    fin = open(filename)

    word2index = {}

    # Masking
    word2index['MASK'] = 0
    # Out of vocabulary words
    word2index['UNK'] = 1
    # Padding
    word2index['EOS'] = 2

    word_vecs = []
    fin = fin.readlines()[1:]
    for line in fin:
        splited_line = line.strip().split()
        word = splited_line[0]
        word_vecs.append(splited_line[1:])

        word2index[word] = len(word2index)

    word_vecs_np = np.zeros(shape=(len(word2index), len(word_vecs[1])), dtype='float32')
    word_vecs_np[3:] = word_vecs

    return word_vecs_np, word2index

word_vecs, word2index = read_wordvecs('/home/dominik/json/GoogleTest.txt')

def returnIndex(word):
    if word in word2index:
        return word2index[word]
    else:
        return word2index['UNK']

#step8
with open('out.txt') as fin:
    raw_text = fin.read()

# Get sentences
sentences = re.split('\?+!+|!+\?+|\.+|!+|\?+', raw_text)

# Get rid of empty sentences
sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

# Tokenize sentences (simple space tokenizer) and lower case them
sentences = [[w.lower() for w in s.split()] for s in sentences]

sent = []
for sen in sentences:
    part =[]
    for pa in sen:
        part.append(returnIndex(pa))
    sent.append(part)
sentences = sent
sentences = pad_sequences(sentences)

training_set=sentences[:round(len(sentences) * 0.8)]
test_set = sentences[round(len(sentences) * 0.8):]
windowSize = 2 # 2 to the left, 2 to the right
numHiddenUnits = 100

# Create a mapping for our labels
label2index = {'O': 0}
idx = 1

for bioTag in ['B-', 'I-']:
    for nerClass in ['PER', 'LOC', 'ORG', 'OTH']:
        for subtype in ['', 'deriv', 'part']:
            label2index[bioTag + nerClass + subtype] = idx
            idx += 1

# Inverse label mapping
index2label = {v: k for k, v in label2index.items()}






n_in = 2*windowSize+1
n_hidden = numHiddenUnits
n_out = len(label2index)

train_x, train_y = GermEvalReader.createNumpyArray(training_set, windowSize, word2index, label2index)
test_x, test_y = GermEvalReader.createNumpyArray(test_set, windowSize, word2index, label2index)
train_y_cat = np_utils.to_categorical(train_y, n_out)


number_of_epochs = 10
batch_size = 35

model = Sequential()

model.add(Embedding(output_dim=word_vecs.shape[1], input_dim=word_vecs.shape[0],
                    input_length=n_in,  weights=[word_vecs], mask_zero=False))
model.add(LSTM(n_hidden, W_regularizer=l2(0.0001), U_regularizer=l2(0.0001), return_sequences=True))
model.add(TimeDistributed(Dense(n_out, activation='softmax', W_regularizer=l2(0.0001))))
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')



print(str(training_set.shape[0]) + ' train samples')
print(str(training_set.shape[1]) + ' train dimension')
print(str(test_set.shape[0]) + ' test samples')

print("\n%d epochs" % number_of_epochs)
print("%d mini batches" % (len(training_set)/batch_size))

sys.stdout.flush()

for epoch in range(number_of_epochs):
    start_time = time.time()

    # Train for 1 epoch
    model.fit(training_set, train_y_cat, nb_epoch=1, batch_size=batch_size, verbose=False, shuffle=True)
    print("%.2f sec for training" % (time.time() - start_time))
    sys.stdout.flush()

    model.pre
    # Compute precision, recall, F1 on dev & test data
    pre_test, rec_test, f1_test = BIOF1Validation.compute_f1(model.predict_classes(test_x, verbose=0), test_y,
                                                             index2label)

    print("%d epoch: F1 on dev: %f, F1 on test: %f" % (epoch + 1, f1_test))
    sys.stdout.flush()