"""

This code is modified from

https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py
and ttps://github.com/fchollet/keras/blob/master/examples/
for our own purposes

"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from matcher_functions import connect, get_aliases, load_good_buckets, create_double_num_dicts 
from matcher_class import matcher
import os
from sqlalchemy import Table, Column, Integer, String, ForeignKey, Float
import sys
from sys import argv
import numpy as np
from embeddings import KazumaCharEmbedding
import random

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Flatten, Dropout, Lambda

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model, model_from_json, Sequential

from keras.utils import to_categorical

from keras.optimizers import RMSprop

from keras import backend as K

from keras import regularizers
from keras.regularizers import L1L2



BASE_DIR = './Machine_Learning/'

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

VALIDATION_SPLIT = 0.05



# first, build index mapping words in the glove embeddings set

# to their embedding vector.  This is a straightforward lookup of

# words in Glove and then their embeddings which should be a 100 sized array of floats

# print('Reading word embeddings: Indexing word vectors.')



# embeddings_index = {}

# f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))

# for line in f:

#     values = line.split()

#     word = values[0]

#     coefs = np.asarray(values[1:], dtype='float32')

#     embeddings_index[word] = coefs

# f.close()



# print('Found %s word vectors.' % len(embeddings_index))



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


#change to get from sql and not read from file
con, meta = connect(argv[1], argv[2], argv[3])
#load pairs from database
aliases = get_aliases(con, meta)
for pair in aliases:

    num_tokens = len(pair[0].strip().split(' ')) + len(pair[1].strip().split(' '))

    if 0 < num_tokens < MAX_SEQUENCE_LENGTH:

        texts1.append(pair[0])

        texts2.append(pair[1])

f.close()
#this returns a new set of texts to use as similar non-matches for texts1
def get_no_match_texts(argv, texts1):
    def get_non_match(name1, bucket_words, matching_set):
        for word in name1.split(" "):
            if word in bucket_words:
                bucket = bucket_words[word]
            else:
                return None
            if len(bucket[1]) > 1:
                for name2 in bucket[1]:
                    if (name1, name2[1]) not in matching_set:
                        return name2[1]
        return None
    no_match_texts = []
    #this should not be done here and needs to be fixed up before more work is done
    #it should instead be done by a singel function in matcher_functions
    #establish connection to database
    con, meta = connect(argv[1], argv[2], argv[3])
    #load pairs from database
    aliases = get_aliases(con, meta)
    #create dictionaries assingning serial numbers to names and names from serial numbers
    num_to_word, word_to_num = create_double_num_dicts(aliases)
    #load the buckets from the database bucket_list is aranges as follows:
    #bucket_list[pair_of_buckets][bucket(this must be 0 or 1)][name (this represents a single name)][0 for number and 1 for pre-procced name]
    bucket_list, bucket_words = load_good_buckets('wordtable1', 'wordtable2', word_to_num, con, meta)
    for index in range(len(texts1)):
        new_text = get_non_match(texts1[index], bucket_words, aliases)
        if new_text == None:
            new_text = texts1[(index + 1) % len(texts1)]
        no_match_texts.append(new_text)
    return no_match_texts



print('Found %s texts.' % len(texts1))



# finally, vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#gets the special no match charectars
texts3 = get_no_match_texts(argv, texts1)

#this removes all non-ascii charectars from the 3 sets of strings
texts1 = [str(item) for item in texts1]
texts2 = [str(item) for item in texts2]
texts3 = [str(item) for item in texts3]
print ("1 is {}".format(texts1[1]))
print ("2 is {}".format(texts2[1]))
print ("3 is {}".format(texts3[1]))
tokenizer.fit_on_texts(texts1 + texts2 + texts3)
#this step should get similar but non-matching items to keep for later matching
# this step creates a sequence of words ids for each word in each label
sequences1 = tokenizer.texts_to_sequences(texts1)
sequences2 = tokenizer.texts_to_sequences(texts2)
no_match_sequences = tokenizer.texts_to_sequences(texts3)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)
no_match_data = pad_sequences(no_match_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# print (data1[0])
# print (data2[0])
# print (texts1[0])
# print (texts2[0])
#labels = to_categorical(np.asarray(labels))

print('Shape of data1 tensor:', data1.shape)

#print('Shape of label tensor:', labels.shape)

print('Shape of data2 tensor:', data2.shape)

# split the data into a training set and a validation set

indices = np.arange(data1.shape[0])

np.random.shuffle(indices)

data1 = data1[indices]

data2 = data2[indices]

no_match_data = no_match_data[indices]

num_validation_samples = int(VALIDATION_SPLIT * data1.shape[0])



x_train = data1[:-num_validation_samples]

y_train = data2[:-num_validation_samples]

z_train = no_match_data[:-num_validation_samples]

x_test = data1[-num_validation_samples:]

y_test = data2[-num_validation_samples:]

z_test = no_match_data[-num_validation_samples:]

print('Preparing embedding matrix.')



# prepare embedding matrix

num_words = min(MAX_NB_WORDS, len(word_index))
num_words = MAX_NB_WORDS
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
k = KazumaCharEmbedding()

for word, i in word_index.items():

    if i >= MAX_NB_WORDS:

        continue
    #print(word)                                                 
    embedding_vector = k.emb(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
    #else:
     #   print(word + )



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
def create_pairs(x, y, z):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    for index in range(len(x)):
        if sum(x[index]) == 0:
            print(sequence_to_word(y[index], reverse_word_index))
            print("X::{}:: y::{}:: z::{}::".format(x[index], y[index], z[index]))
        pairs += [[x[index], y[index]]]
        pairs += [[x[index], z[index]]]
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


def create_base_network(input_dim, embedding_layer, reg):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(embedding_layer)
    seq.add(Flatten())
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
#                    kernel_regularizer=reg))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
 #                   kernel_regularizer=reg))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

def f1score(predictions, labels):
    #labels[predictions.ravel() < 0.5].sum()
    predictions = predictions.ravel()
    fsocre = 0.0
    true_positive = 0.0
    false_positive = 0
    false_negitive = 0
    for i in range(len(labels)):
        if predictions[i] < 0.5:
            if labels[i] == 1:
                true_positive += 1
            else:
                false_positive += 1
        elif labels[i] == 1:
            false_negitive += 1
    print('tp' + str(true_positive))
    print('fp' + str(false_positive))
    print('fn' + str(false_negitive))
    fscore = (2 * true_positive) / ((2 * true_positive) + false_negitive + false_positive)
    print (fscore)
    return fscore 
# the data, shuffled and split between train and test sets
#need to change this not sure how

input_dim = MAX_SEQUENCE_LENGTH
epochs = 1

# create training+test positive and negative pairs
# these next lines also need to change
#digit_indices = [np.where(y_train == i)[0] for i in range(10)]
print("x_train {} , y_train {} , z_train {} ".format(x_train, y_train, z_train))
tr_pairs, tr_y = create_pairs(x_train, y_train, z_train)

#digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(x_test, y_test, z_test)
print (len(tr_y))
# network definition
base_network = create_base_network(input_dim, embedding_layer, L1L2(0.0,0.0))

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
#change the optimizer (adam)
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
# compute final accuracy on training and test sets
#add an LSTM layer (later)
# test_pairs = [[lambda x : x.replace(" ", ""), lambda name1, name2 : name1 in name2 or name2 in name1],
#  [lambda x : set(x.split()), lambda name1, name2 : name1.issubset(name2) or name2.issubset(name1)]]
# matcher = matcher(argv[1], argv[2], argv[3], test_pairs, 1)
pred_learning = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred_learning, tr_y)
tr_f1 = f1score(pred_learning, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
pred_learning = np.append(pred_learning, pred, axis=0)
te_acc = compute_accuracy(pred, te_y)
te_f1 = f1score(pred, te_y)
print("Machine Learning Accuracy")
print(tr_acc)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
print('* f1score on the training set: %0.4f' % (tr_f1))
print('* f1socre on test set: %0.4f' % (te_f1))
#compute accuracy using a rule based matcher
def sequence_to_word(sequence, reverse_word_index):
    return " ".join([reverse_word_index[x] for x in sequence if x in reverse_word_index])
def sequence_pair_to_word_pair(sequence_pair, reverse_word_index):
    return [sequence_to_word(sequence_pair[0], reverse_word_index), sequence_to_word(sequence_pair[1], reverse_word_index)]
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
print(tr_pairs)
print(sequence_to_word(tr_pairs[0][0], reverse_word_index))
print(sequence_to_word(tr_pairs[0][1], reverse_word_index))
print(tr_y[0])

test_pairs = [[lambda x : x.replace(" ", ""), lambda name1, name2 : name1 in name2 or name2 in name1],
 [lambda x : set(x.split()), lambda name1, name2 : name1.issubset(name2) or name2.issubset(name1)]]
matcher = matcher(argv[1], argv[2], argv[3], test_pairs, 1)

pred_rules = np.asarray([int(not matcher.match(*sequence_pair_to_word_pair(name_pair, reverse_word_index))) for name_pair in tr_pairs])
tr_acc = compute_accuracy(pred_rules, tr_y)
tr_f1 = f1score(pred_rules, tr_y)
pred = np.asarray([int(not matcher.match(*sequence_pair_to_word_pair(name_pair, reverse_word_index))) for name_pair in te_pairs])
pred_rules = np.append(pred_rules, pred, axis=0)
te_acc = compute_accuracy(pred, te_y)
te_f1 = f1score(pred, te_y)
print("Rule-Based Accuracy")
print('* Accuracy on training set (rules): %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set (rules): %0.2f%%' % (100 * te_acc))
print('* f1score on the training set: %0.4f' % (tr_f1))
print('* f1socre on test set: %0.4f' % (te_f1))
con, meta = connect(argv[1], argv[2], argv[3])
execute_pairs = []
if 'predictions' in meta.tables:
    meta.tables['predictions'].drop(con)
predictions = Table('predictions', meta, Column('name1', String), Column('name2', String), Column('rule_predict', Integer), Column('learning_predict', Float), Column('true_pair', Integer), Column('te_or_tr', String), extend_existing=True)
zipping_string = ('name1', 'name2', 'true_pair', 'rule_predict', 'learning_predict', 'te_or_tr')
print(len(tr_y))
print(len(tr_pairs))
print(len(pred_rules))
print(len(pred_learning))
print(len(te_y))
print(len(te_pairs))
print(tr_pairs[1][0])
for i in range(len(tr_y)):
    execute_pairs.append(dict(zip(zipping_string, (sequence_to_word(tr_pairs[i][0], reverse_word_index), sequence_to_word(tr_pairs[i][1], reverse_word_index), int(tr_y[i]), int(pred_rules[i]), float(pred_learning[i][0].item()), 'tr'))))
offset = len(tr_y)
for i in range(len(te_y)):
    execute_pairs.append(dict(zip(zipping_string, (sequence_to_word(te_pairs[i][0], reverse_word_index), sequence_to_word(te_pairs[i][1], reverse_word_index), int(te_y[i]), int(pred_rules[offset + i]), float(pred_learning[offset + i][0].item()), 'te'))))
meta.create_all(con)
con.execute(predictions.insert(), execute_pairs)

