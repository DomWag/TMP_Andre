import re
import sys
import time

import numpy as np
from keras.layers import Embedding, LSTM, Dense, TimeDistributed
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.utils import np_utils
from __future__ import division

import BIOF1Validation


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


with open('out.txt') as fin:
    raw_text = fin.read()

# Get sentences
sentences = re.split('\?+!+|!+\?+|\.+|!+|\?+', raw_text)

# Get rid of empty sentences
sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

# Tokenize sentences (simple space tokenizer) and lower case them
sentences = [[w.lower() for w in s.split()] for s in sentences]

#step 8, get the index of each token in the dataset
sent = []
for sen in sentences:
    part =[]
    for pa in sen:
        part.append(returnIndex(pa))
    sent.append(part)
sentences = sent
sentences = pad_sequences(sentences)

#split the dataset according to step9
training_set=sentences[:int(len(sentences) * 0.8)]
test_set = sentences[int(len(sentences) * 0.8):]


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





#the n in is the length of one sentence.
n_in = len(sentences[0])

n_hidden = numHiddenUnits
n_out = len(label2index)

train_x =  training_set
train_y = training_set[1:]
train_y = np.append(train_y, word2index['EOS'])
train_y_cat = np_utils.to_categorical(train_y, n_out)

test_x =  test_set
test_y = test_set[1:]
test_y = np.append(test_y, word2index['EOS'])

number_of_epochs = 10
batch_size = 35
model = Sequential()

model.add(Embedding(output_dim=word_vecs.shape[1], input_dim=word_vecs.shape[0],
                    input_length=n_in,  input_shape=(114549,25),weights=[word_vecs], mask_zero=False))
model.add(LSTM(n_hidden, W_regularizer=l2(0.0001), U_regularizer=l2(0.0001), return_sequences=True))
model.add(TimeDistributed(Dense(n_out, activation='softmax', W_regularizer=l2(0.0001))))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



print(str(training_set.shape[0]) + ' train samples')
print(str(training_set.shape[1]) + ' train dimension')
print(str(test_set.shape[0]) + ' test samples')

print("\n%d epochs" % number_of_epochs)
print("%d mini batches" % (len(training_set)/batch_size))

sys.stdout.flush()

for epoch in range(number_of_epochs):
    start_time = time.time()

    # Train for 1 epoch

    model.fit(train_x, train_y_cat, nb_epoch=1, batch_size=batch_size, verbose=False, shuffle=True)
    print("%.2f sec for training" % (time.time() - start_time))
    sys.stdout.flush()

   #get the perplexity
    probs = model.predict_proba(test_x)
    prob_prod = np.prod(probs)
    size_prob = probs.size
    invert_size = 1/size_prob
    print("perplexity is"+str(pow(prob_prod, -invert_size)))

    sys.stdout.flush()