"""

This code is modified from

https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py
and ttps://github.com/fchollet/keras/blob/master/examples/
for our own purposes

"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np



import os

import sys

import numpy as np

import random

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Flatten, Dropout, Lambda

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model, model_from_json, Sequential

from keras.utils import to_categorical

from keras.optimizers import RMSprop

from keras import backend as K



BASE_DIR = './'

# directory containing glove encodings from Wikipedia (we can swap this out for another encoding later)

# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/

GLOVE_DIR = BASE_DIR + 'glove/'

TEXT_DATA_DIR = BASE_DIR + 'nerData'

# number of words an entity is allowed to have

# distribution of number of words in peoples names can be found in peopleNamesDisbn

# distribution of number of words in company names can be found in companyNamesDisbn

# Note most of the names above that are fairly esoteric or just plain noise.  Included is

# python code to remove them

MAX_SEQUENCE_LENGTH = 10






# Total number of unique tokens in peoples names is 90K, including a lot of non-English names.  To remove those

# we use an egregious hack to check if its UTF-8

# Total number of unique tokens in company names is 37K

# Assuming no overlap between the two we get about 127K.  We may need to tweak this parameter as we go

# but according to the Keras documentation, this can even be left unset

MAX_NB_WORDS = 40000

# Size of embeddings from Glove (we will try the 100 dimension encoding to start with)

EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.2



# first, build index mapping words in the glove embeddings set

# to their embedding vector.  This is a straightforward lookup of

# words in Glove and then their embeddings which should be a 100 sized array of floats

print('Reading word embeddings: Indexing word vectors.')



embeddings_index = {}

f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))



# second, prepare text samples and their labels

print('Processing text dataset')



texts1 = []  # list of text samples in part 1
texts2 = [] # list of text samples in part 2

labels_index = {}  # dictionary mapping label name to numeric id - here the label name is just the name of the file in the data dir



name_list = sorted(os.listdir(TEXT_DATA_DIR))
name = name_list[0]
label_id = len(labels_index)

labels_index[name] = label_id

fpath = os.path.join(TEXT_DATA_DIR, name)

if os.path.isdir(fpath):

    raise ValueError('bad data directory')

if sys.version_info < (3,):

    f = open(fpath)

else:

    f = open(fpath, encoding='latin-1')



for t in f.readlines():

    num_tokens = len(t.strip().split(' '))

    if 0 < num_tokens < MAX_SEQUENCE_LENGTH:
        split_text = t.split('|')

        texts1.append(split_text[0])

        texts2.append(split_text[1])

f.close()



print('Found %s texts.' % len(texts1))



# finally, vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(texts1 + texts2)

# this step creates a sequence of words ids for each word in each label

sequences1 = tokenizer.texts_to_sequences(texts1)
sequences2 = tokenizer.texts_to_sequences(texts2)


word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)


#labels = to_categorical(np.asarray(labels))

print('Shape of data1 tensor:', data1.shape)

#print('Shape of label tensor:', labels.shape)

print('Shape of data2 tensor:', data2.shape)

# split the data into a training set and a validation set

indices = np.arange(data1.shape[0])

np.random.shuffle(indices)

data1 = data1[indices]

data2 = data2[indices]

num_validation_samples = int(VALIDATION_SPLIT * data1.shape[0])



x_train = data1[:-num_validation_samples]

y_train = data2[:-num_validation_samples]

x_test = data1[-num_validation_samples:]

y_test = data2[-num_validation_samples:]



print('Preparing embedding matrix.')



# prepare embedding matrix

num_words = min(MAX_NB_WORDS, len(word_index) - 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
print (embedding_matrix)
for word, i in word_index.items():

    if i >= MAX_NB_WORDS:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector



# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(num_words,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=False)



print('Training model.')



# train a 1D convnet with global maxpooling

#sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#need two of these
#embedded_sequences = embedding_layer(sequence_input)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

#for now will take any bad pairs, will take only relivent ones later
def create_pairs(x, y):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    for index in range(len(x)):
        pairs += [[x[index], y[index]]]
        pairs += [[x[index], y[(index + 1) % len(x)]]]
        labels += [1, 0]
    # n = min([len(digit_indices[d]) for d in range(10)]) - 1
    # for d in range(10):
    #     for i in range(n):
    #         z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
    #         pairs += [[x[z1], x[z2]]]
    #         inc = random.randrange(1, 10)
    #         dn = (d + inc) % 10
    #         z1, z2 = digit_indices[d][i], digit_indices[dn][i]
    #         pairs += [[x[z1], x[z2]]]
    #         labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim, embedding_layer):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()

    seq.add(embedding_layer)
    seq.add(Flatten())
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# the data, shuffled and split between train and test sets
#need to change this not sure how

input_dim = MAX_SEQUENCE_LENGTH
epochs = 20

# create training+test positive and negative pairs
# these next lines also need to change
#digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(x_train, y_train)

#digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(x_test, y_test)
print (len(tr_y))
# network definition
base_network = create_base_network(input_dim, embedding_layer)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
print (len(pred))
print (pred[0])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


