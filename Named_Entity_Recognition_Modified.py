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
from annoy import AnnoyIndex
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
from sqlalchemy.sql import select
import random
import argparse

BASE_DIR = './Machine_Learning/'

GLOVE_DIR = BASE_DIR + 'glove/'

TEXT_DATA_DIR = BASE_DIR + 'nerData'

DO_ANN_ON_EMBEDDINGS = False

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

MAX_NB_WORDS = 150000

# Size of embeddings from Glove (we will try the 100 dimension encoding to start with)

EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.05


def check_for_zeroes(to_check, intro_string):
        found = 0
        for vector in to_check:
            if sum(vector) == 0:
                print(intro_string + str(vector))
                found += 1
        if not found:
            print(intro_string + " no problems found")
        else:
                print(intro_string + ' found this many: '+ str(found))

def get_diff_names_with_overlap(con, aliases):
    no_match_texts = []
    texts = []
    # load a mapping from entity id to names
    entityid2names = {}
    for row in aliases:
        if row[2] in entityid2names:
            names = entityid2names[row[2]];
        else:
            names = []
            entityid2names[row[2]] = names
        names.append(row[0])
        names.append(row[1])
        texts.append(row[0])

    print(len(aliases))

    print("getting word2entities")
    # load a mapping from words to entities
    word2entities = {}
    rows = con.execute("select word, entities from word2entities;")
    for row in rows:
        word2entities[row[0]] = row[1]

   
    for index in range(len(texts)):
        name_arr = texts[index].split()
        new_text = ''
        for n in name_arr:
            # a name part may have been filtered out of word2entities
            if n not in word2entities:
                continue
            if new_text:
                break
            for e in word2entities[n]:
                if e == texts[index] or e not in entityid2names:   # if the entity is the same as this anchor's text skip it
                    continue
                names = entityid2names[e]
                for x in names:
                    if n in x:
                        new_text = x
                        break
        if new_text:
            no_match_texts.append(new_text)
        else:
            no_match_texts.append(texts[index + 1])
    print("done processing matches with overlap")

    return no_match_texts

def get_diff_names_with_no_overlap(aliases):
    entitylist = []

    for row in aliases:
        entitylist.append(row[0])
 
    s = [i for i in range(len(entitylist))]
    random.shuffle(s)

    ret = []
    for i in range(len(entitylist)):
        if s[i] == i :
            s[i] = s[i+1]
            s[i+1] = i

        ret.append(entitylist[s[i]])
    return ret

  
#this returns a new set of texts to use as similar non-matches for texts1
def get_no_match_texts(user, password, db, texts1):
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
    con, meta = connect(user, password, db)
    #load pairs from database
    aliases = get_aliases(con, meta)
    #create dictionaries assigning serial numbers to names and names from serial numbers
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

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Equation 4
    '''
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) +
                  y_true * K.square(K.maximum(margin - y_pred, 0)))

#for now will take any bad pairs, will take only relivent ones later
def create_pairs(x, y, z):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    for index in range(len(x)):
        pairs += [[x[index], y[index]]]
        pairs += [[x[index], z[index]]]
        labels += [0, 1]
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
    final_layer = Dense(128, activation='relu')
    seq.add(final_layer)
 #                   kernel_regularizer=reg))
    return seq, final_layer

def embedded_representation(embedding_layer):
    seq = Sequential()
    seq.add(embedding_layer)
    seq.add(Flatten())
    return seq
 

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    same_correct = 0
    diff_correct = 0
    for i in range(len(labels)):
        if predictions[i] < 0.5 and labels[i] == 0:
            same_correct += 1
        elif predictions[i] >= 0.5 and labels[i] == 1:
            diff_correct += 1

    print("Precision computation: same - " + str(same_correct) + " different: " + str(diff_correct) + " from total: " + str(len(labels)))
    return (same_correct + diff_correct) / len(labels)

def get_aliases_with_ids(con, meta):
    #load pairs from database
    aliases = con.execute("select alias1, alias2, entityid from aliases order by entityid;")
    entities = []
    for row in aliases:
        entities.append((row[0], row[1], row[2]))
    return entities

def f1score(predictions, labels):
    #labels[predictions.ravel() < 0.5].sum()
    predictions = predictions.ravel()
    fsocre = 0.0
    true_positive = 0.0
    false_positive = 0
    false_negitive = 0
    for i in range(len(labels)):
        if predictions[i] < 0.5:
            if labels[i] == 0:
                true_positive += 1
            else:
                false_positive += 1
        elif labels[i] == 0:
            false_negitive += 1
    print('tp' + str(true_positive))
    print('fp' + str(false_positive))
    print('fn' + str(false_negitive))
    fscore = (2 * true_positive) / ((2 * true_positive) + false_negitive + false_positive)
    print (fscore)
    return fscore 

#compute accuracy using a rule based matcher
def sequence_to_word(sequence, reverse_word_index):
    return " ".join([reverse_word_index[x] for x in sequence if x in reverse_word_index])
    
def sequence_pair_to_word_pair(sequence_pair, reverse_word_index):
    return [sequence_to_word(sequence_pair[0], reverse_word_index), sequence_to_word(sequence_pair[1], reverse_word_index)]

if __name__ == '__main__':
    print('Processing text dataset')

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-u', dest="user", help="username")
    parser.add_argument('-p', dest="password", help="password")
    parser.add_argument('-d', dest="db", help="dbname")
    parser.add_argument('-a', dest="num_pairs", help="number of same pairs in db", nargs='?', default=2, type=int)

    args = parser.parse_args()

    texts1 = []  # list of text samples in part 1
    texts2 = [] # list of text samples in part 2

    #change to get from sql and not read from file
    con, meta = connect(args.user, args.password, args.db)
    aliases= get_aliases_with_ids(con, meta)

    unique_aliases = []

    # collect up all the anchors that are unique (anchors will get repeated if num_pairs > 2)
    prev = int(aliases[0][2])
    unique_aliases.append(aliases[0])
    for tuple in aliases:
        texts1.append(tuple[0])
        texts2.append(tuple[1])
        if int(tuple[2]) != prev:
            unique_aliases.append(tuple)
            prev = int(tuple[2])

    print('Found %s texts.' % len(texts1))

    texts3 = []
    print(len(unique_aliases))
    print(len(texts1))
    print(len(texts2))
 
    # get the different pairs
    if args.num_pairs == 2:
        print("args num pairs is 2")
        texts3.extend(get_diff_names_with_overlap(connect(args.user, args.password, args.db)[0], unique_aliases))
    elif args.num_pairs == 3:
        print("args num pairs is 3")
        texts3.extend(get_diff_names_with_overlap(connect(args.user, args.password, args.db)[0], unique_aliases))
        texts3.extend(get_diff_names_with_no_overlap(unique_aliases))
    elif args.num_pairs == 4:
        print("args num pairs is 4")
        texts3.extend(get_diff_names_with_overlap(connect(args.user, args.password, args.db)[0], unique_aliases))
        texts3.extend(get_diff_names_with_no_overlap(unique_aliases))
        texts3.extend(get_diff_names_with_no_overlap(unique_aliases))
 
    print(len(texts3))
 
    assert len(texts1) == len(texts2)
    assert len(texts2) == len(texts3), str(len(texts3))

    # vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

    tokenizer.fit_on_texts(texts1 + texts2 + texts3)
    # this step should get similar but non-matching items to keep for later matching
    # this step creates a sequence of words ids for each word in each label
    sequences1 = tokenizer.texts_to_sequences(texts1)
    for sc in range(len(texts1)):
            if sum(sequences1[sc]) == 0:
                    print('here is a problem word :' + texts1[sc] + '::')
    sequences2 = tokenizer.texts_to_sequences(texts2)
    no_match_sequences = tokenizer.texts_to_sequences(texts3)
    word_index = tokenizer.word_index
    check_for_zeroes(sequences1, " sequences")
    print('Found %s unique tokens.' % len(word_index))


    annoy_data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
    annoy_data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)
    no_match_data = pad_sequences(no_match_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of data1 tensor:', annoy_data1.shape)
    print('Shape of data2 tensor:', annoy_data2.shape)

    # split the data into a training set and a validation set, shuffling items
    indices = np.arange(annoy_data1.shape[0])
    np.random.shuffle(indices)

    texts1 = np.array(texts1)
    texts2 = np.array(texts2)
    texts3 = np.array(texts3)

    texts1 = texts1[indices]
    texts2 = texts2[indices]
    texts3 = texts3[indices]

    for i in range(len(texts1)):
        print(texts1[i] + " paired with: " + texts2[i])
        print(texts1[i] + " paired with: " + texts3[i])


    data1 = annoy_data1[indices]
    data2 = annoy_data2[indices]
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
    num_words = len(word_index) + 1                     # word_index is indexed from 1-N

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    kz = KazumaCharEmbedding()

    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = kz.emb(word)

        if embedding_vector is not None:
            if sum(embedding_vector) == 0:
                print("failed to find embedding for:" + word)
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # note that we set trainable = False so as to keep the embeddings fixed
    check_for_zeroes(embedding_matrix, "here is the first pass")
    embedding_layer = Embedding(num_words,

                                EMBEDDING_DIM,

                                weights=[embedding_matrix],

                                input_length=MAX_SEQUENCE_LENGTH,

                                trainable=False)


    print('Training model.')


    # the data, shuffled and split between train and test sets
    #need to change this not sure how

    input_dim = MAX_SEQUENCE_LENGTH
    epochs = 10

    # create training+test positive and negative pairs
    # these next lines also need to change
    #digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    print("x_train {} , y_train {} , z_train {} ".format(x_train, y_train, z_train))
    tr_pairs, tr_y = create_pairs(x_train, y_train, z_train)
    #digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(x_test, y_test, z_test)
    print (len(tr_y))
    # network definition
    base_network, final_layer = create_base_network(input_dim, embedding_layer, L1L2(0.0,0.0))

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
    print(base_network.summary())
    # train
    rms = RMSprop()
    #change the optimizer (adam)
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epochs)
    # compute final accuracy on training and test sets
    #add an LSTM layer (later)
    # testpairs = [[lambda x : x.replace(" ", ""), lambda name1, name2 : name1 in name2 or name2 in name1],
    #  [lambda x : set(x.split()), lambda name1, name2 : name1.issubset(name2) or name2.issubset(name1)]]
    # matcher = matcher(argv[1], argv[2], argv[3], test_pairs, 1)
    pred_learning = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    # out = model.layers[2].get_output_at(0)
    # inp = model.input
    # func = K.function([inp], [out])   # evaluation functions
    # print("here should be a vector")
    # print(func([tr_pairs[:, 0][0], tr_pairs[:, 1][0]]))
    # Testing
    # print (layer_outs)
    tr_acc = compute_accuracy(pred_learning, tr_y)
    tr_f1 = f1score(pred_learning, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    pred_learning = np.append(pred_learning, pred, axis=0)
    te_acc = compute_accuracy(pred, te_y)

    te_f1 = f1score(pred, te_y)

    x_test_text = texts1[-num_validation_samples:]
    y_test_text = texts2[-num_validation_samples:]
    z_test_text = texts3[-num_validation_samples:]

    text_pairs, text_y = create_pairs(x_test_text, y_test_text, z_test_text)

    for i in range(len(text_pairs)):
        print(str(text_pairs[i]))
        print(pred[i])
        print(model.predict([np.array([te_pairs[i, 0]]), np.array([te_pairs[i, 1]])]))

    # from https://github.com/spotify/annoy
    f = 128

    if DO_ANN_ON_EMBEDDINGS:
        inter_model = embedded_representation(embedding_layer)
    else:
        inter_model = Model(input_a, processed_a) 

    intermediate_output1 = inter_model.predict(x_test)
    intermediate_output2 = inter_model.predict(y_test)
    intermediate_output3 = inter_model.predict(z_test)

    mid_predictions = np.concatenate((intermediate_output1, intermediate_output2, intermediate_output3))


    # print(mid_predictions[0])
    # print (len(mid_predictions[0]))

    if DO_ANN_ON_EMBEDDINGS:
        t = AnnoyIndex(len(mid_predictions[0]), metric='euclidean')  # Length of item vector that will be indexed
    else:
        t = AnnoyIndex(f, metric='euclidean')  # Length of item vector that will be indexed

    for i in range(len(mid_predictions)):
        v = mid_predictions[i]
        t.add_item(i, v)

    t.build(100) # 100 trees
    t.save('test.ann')

    # ...

    all_texts = np.concatenate((x_test_text, y_test_text, z_test_text))
    match = 0
    no_match = 0

    for index in range(len(x_test_text)):
        nearest = t.get_nns_by_vector(mid_predictions[index], 5)
        print(nearest)
        nearest_text = [all_texts[i] for i in nearest]
        print("query={} names = {} true_match = {} reject= {}".format(x_test_text[index], nearest_text, y_test_text[index], z_test_text[index]))

        for i in nearest:
            print(all_texts[i])
            if i >= len(x_test_text) and (i < len(x_test_text) + len(y_test_text)):
                arr = np.array([y_test[i - len(x_test_text)]])
            elif i >= len(x_test_text) + len(y_test_text):
                arr = np.array([z_test[i - len(x_test_text) - len(y_test_text)]])
            else:
                arr = np.array([x_test[i]])
            print(model.predict([np.array([x_test[index]]), arr]))
            print(t.get_distance(index, i))

        print("true match prediction:")
        print(model.predict([np.array([x_test[index]]), np.array([y_test[index]])]))
        print("true match distance:")
        print(t.get_distance(index, index + len(x_test_text)))

        print("true reject prediction:")
        print(model.predict([np.array([x_test[index]]), np.array([z_test[index]])]))
        print("true reject distance:")
        print(t.get_distance(index, index + len(x_test_text) + len(y_test_text)))

        if y_test_text[index] in nearest_text:
            match += 1
            print("MATCH FOUND")
        else:
            no_match += 1

    print("match: {} no_match: {}".format(match, no_match))

    print("Machine Learning Accuracy")
    print(tr_acc)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('* f1score on the training set: %0.4f' % (tr_f1))
    print('* f1socre on test set: %0.4f' % (te_f1))

    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    print(tr_pairs)
    print(sequence_to_word(tr_pairs[0][0], reverse_word_index))
    print(sequence_to_word(tr_pairs[1][1], reverse_word_index))
    print(tr_y[0])

    test_pairs = [[lambda x : x.replace(" ", ""), lambda name1, name2 : name1 in name2 or name2 in name1],
     [lambda x : set(x.split()), lambda name1, name2 : name1.issubset(name2) or name2.issubset(name1)]]
    matcher = matcher(args.user, args.password, args.db, test_pairs, 1)

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
    print('* f1score on test set: %0.4f' % (te_f1))
    con, meta = connect(args.user, args.password, args.db)
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
    for i in range(len(tr_y)):
        execute_pairs.append(dict(zip(zipping_string, (sequence_to_word(tr_pairs[i][0], reverse_word_index), sequence_to_word(tr_pairs[i][1], reverse_word_index), int(tr_y[i]), int(pred_rules[i]), float(pred_learning[i][0].item()), 'tr'))))
    offset = len(tr_y)
    for i in range(len(te_y)):
        execute_pairs.append(dict(zip(zipping_string, (sequence_to_word(tr_pairs[i][0], reverse_word_index), sequence_to_word(te_pairs[i][1], reverse_word_index), int(te_y[i]), int(pred_rules[offset + i]), float(pred_learning[offset + i][0].item()), 'te'))))
    meta.create_all(con)
    con.execute(predictions.insert(), execute_pairs)

