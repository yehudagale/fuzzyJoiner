from sys import argv

from keras import backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Flatten, Dropout, Lambda, GRU
from keras.layers.wrappers import Bidirectional

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model, model_from_json, Sequential

import numpy as np

from embeddings import KazumaCharEmbedding

from annoy import AnnoyIndex

from keras.callbacks import ModelCheckpoint, EarlyStopping

from names_cleanser import NameDataCleanser

import random

#must fix
MAX_NB_WORDS = 140000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 10
MARGIN=1

DEBUG = False
DEBUG_DATA_LENGTH = 10000


def f1score(positive, negative):
    #labels[predictions.ravel() < 0.5].sum()
    fsocre = 0.0
    true_positive = 0.0
    false_positive = 0
    false_negitive = 0
    for i in range(len(positive)):
        if positive[i] <= negative[i]:
            true_positive += 1
        else:
            false_negitive += 1
            false_positive += 1
    print('tp' + str(true_positive))
    print('fp' + str(false_positive))
    print('fn' + str(false_negitive))
    fscore = (2 * true_positive) / ((2 * true_positive) + false_negitive + false_positive)
    return fscore 


def get_embedding_layer(tokenizer):
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    print('about to get kz')
    kz = KazumaCharEmbedding()
    print('got kz')
    for word, i in word_index.items():

        if i >= MAX_NB_WORDS:

            continue
        embedding_vector = kz.emb(word)

        # i = 0
        # while sum(embedding_vector) == 0 and i <= 1000:
        #     embedding_vector = k.emb(word)
        #     i++;
        #     if i == 1000:
        #         print("fail")
        if embedding_vector is not None:
            if sum(embedding_vector) == 0:
                print("failed to find embedding for:" + word)
            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(num_words,

                                EMBEDDING_DIM,

                                weights=[embedding_matrix],

                                input_length=MAX_SEQUENCE_LENGTH,

                                trainable=False)
    return embedding_layer

def get_sequences(texts, tokenizer):
    sequences = {}
    sequences['anchor'] = tokenizer.texts_to_sequences(texts['anchor'])
    sequences['anchor'] = pad_sequences(sequences['anchor'], maxlen=MAX_SEQUENCE_LENGTH)
    sequences['negative'] = tokenizer.texts_to_sequences(texts['negative'])
    sequences['negative'] = pad_sequences(sequences['negative'], maxlen=MAX_SEQUENCE_LENGTH)
    sequences['positive'] = tokenizer.texts_to_sequences(texts['positive'])
    sequences['positive'] = pad_sequences(sequences['positive'], maxlen=MAX_SEQUENCE_LENGTH)

    return sequences

def read_entities(filepath):
    entities = []
    i = 0
    with open(filepath) as fl:
        for line in fl:
            entities.append(line)
            i += 1
            if i > DEBUG_DATA_LENGTH and DEBUG:
                break
    return entities

def read_file(file_path):
    texts = {'anchor':[], 'negative':[], 'positive':[]}
    fl = open(file_path, 'r')
    i = 0
    for line in fl:
        line_array = line.split("|")
        texts['anchor'].append(line_array[0])
        texts['positive'].append(line_array[1])
        texts['negative'].append(line_array[2])
        i += 1
        if i > DEBUG_DATA_LENGTH and DEBUG:
            break
    return texts

def split(entities, test_split = 0.2):
    random.shuffle(entities)
    num_validation_samples = int(test_split * len(entities))
    return entities[:-num_validation_samples], entities[-num_validation_samples:]

def triplet_loss(y_true, y_pred):
    margin = K.constant(MARGIN)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0]  < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def build_unique_entities(entity2same):
    unique_text = []
    entity2index = {}

    for key in entity2same:
        entity2index[key] = len(unique_text)
        unique_text.append(key)
        vals = entity2same[key]
        for v in vals:
            entity2index[v] = len(unique_text)
            unique_text.append(v)

    return unique_text, entity2index


def generate_triplets_from_ANN(model, sequences, entity2unique, entity2same, unique_text, generate_triplets=True):
    predictions = model.predict(sequences)
    t = AnnoyIndex(len(predictions[0]), metric='euclidean')  # Length of item vector that will be indexed
    for i in range(len(predictions)):
        # print(predictions[i])
        v = predictions[i]
        t.add_item(i, v)

    t.build(100) # 100 trees

    match = 0
    no_match = 0
    triplets = {}


    triplets['anchor'] = []
    triplets['positive'] = []
    triplets['negative'] = []
    NNlen = 20
    for key in entity2same:
        index = entity2unique[key]
        nearest = t.get_nns_by_vector(predictions[index], NNlen)
        nearest_text = set([unique_text[i] for i in nearest])
        expected_text = set(entity2same[key])
        # annoy has this annoying habit of returning the queried item back as a nearest neighbor.  Remove it.
        if key in nearest_text:
            nearest_text.remove(key)
        # print("query={} names = {} true_match = {}".format(unique_text[index], nearest_text, expected_text))
        overlap = expected_text.intersection(nearest_text)
        # collect up some statistics on how well we did on the match
        m = len(overlap)
        match += m
        # since we asked for only 10 nearest neighbors, and we get at most 9 neighbors that are not the same as key (!)
        # make sure we adjust our estimate of no match appropriately
        no_match += min(len(expected_text), NNlen - 1) - m

        # sample only the negatives that are true negatives
        # that is, they are not in the expected set - sampling only 'semi-hard negatives is not defined here'
        # positives = expected_text - nearest_text
        positives = expected_text
        negatives = nearest_text - expected_text

        print(key + str(expected_text) + str(nearest_text))
        for i in negatives:
            for j in positives:
                dist_pos = t.get_distance(index, entity2unique[j])
                dist_neg = t.get_distance(index, entity2unique[i])
                print(key + "|" +  j + "|" + i)
                print(dist_pos)
                print(dist_neg)               
                triplets['anchor'].append(key)
                triplets['positive'].append(j)
                triplets['negative'].append(i)
    if generate_triplets:
        return triplets, match/(match + no_match)
    return match/(match + no_match)

def generate_names(entities, limit_pairs=False):
    num_names = 4
    names_generator = NameDataCleanser(0, num_names, limit_pairs=limit_pairs)
    entity2same = {}
    for entity in entities:
        ret = names_generator.cleanse_data(entity)
        if ret and len(ret) >= num_names:
            entity2same[ret[0]] = ret[1:]
    return entity2same

def embedded_representation_model(embedding_layer):
    seq = Sequential()
    seq.add(embedding_layer)
    seq.add(Flatten())
    return seq

def build_model(embedder):
    main_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    net = embedder(main_input)
    net = GRU(128, return_sequences=True, activation='relu', name='embed')(net)
    net = GRU(128, return_sequences=True, activation='relu', name='embed2')(net)
    net = GRU(128, return_sequences=True, activation='relu', name='embed2a')(net)
    net = GRU(128, activation='relu', name='embed3')(net)
    net = Lambda(l2Norm, output_shape=[128])(net)

    base_model = Model(embedder.input, net, name='triplet_model')

    input_shape=(MAX_SEQUENCE_LENGTH,)
    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_pos')
    input_negative = Input(shape=input_shape, name='input_neg')
    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)
    positive_dist = Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])
    stacked_dists = Lambda( 
                lambda vects: K.stack(vects, axis=1),
                name='stacked_dists'
    )([positive_dist, negative_dist])
    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')
    model.compile(optimizer="rmsprop", loss=triplet_loss, metrics=[accuracy])
    test_positive_model = Model([input_anchor, input_positive, input_negative], positive_dist)
    test_negative_model = Model([input_anchor, input_positive, input_negative], negative_dist)
    inter_model = Model(input_anchor, net_anchor)

    return model, test_positive_model, test_negative_model, inter_model

# read all entities and create positive parts of a triplet
entities = read_entities(argv[1])
train, test = split(entities, test_split = .20)

entity2same_train = generate_names(train)
entity2same_test = generate_names(test, limit_pairs=True)
print(entity2same_train)
print(entity2same_test)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, split=" ")   

# build a set of data structures useful for annoy, the set of unique entities (unique_text), 
# a mapping of entities in texts to an index in unique_text, a mapping of entities to other same entities, and the actual
# vectorized representation of the text.  These structures will be used iteratively as we build up the model
# so we need to create them once for re-use
unique_text, entity2unique = build_unique_entities(entity2same_train)
unique_text_test, entity2unique_test = build_unique_entities(entity2same_test)

tokenizer.fit_on_texts(unique_text + unique_text_test)

sequences = tokenizer.texts_to_sequences(unique_text)
sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
sequences_test = tokenizer.texts_to_sequences(unique_text_test)
sequences_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

# build models
embedder = get_embedding_layer(tokenizer)
model, test_positive_model, test_negative_model, inter_model = build_model(embedder)
embedder_model = embedded_representation_model(embedder)


test_data, test_match_stats = generate_triplets_from_ANN(embedder_model, sequences_test, entity2unique_test, entity2same_test, unique_text_test)
test_seq = get_sequences(test_data, tokenizer)
print("Test stats:" + str(test_match_stats))

match_stats = 0
num_iter = 100
counter = 0
current_model = embedder_model
prev_match_stats = 0

while test_match_stats < .9 and counter < num_iter:
    counter += 1
    train_data, match_stats = generate_triplets_from_ANN(current_model, sequences, entity2unique, entity2same_train, unique_text)
    print("Match stats:" + str(match_stats))
 
    number_of_names = len(train_data['anchor'])
    # print(train_data['anchor'])
    print("number of names" + str(number_of_names))
    Y_train = np.random.randint(2, size=(1,2,number_of_names)).T

    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    early_stop = EarlyStopping(monitor='val_accuracy', patience=1, mode='max') 

    callbacks_list = [checkpoint, early_stop]

    train_seq = get_sequences(train_data, tokenizer)

    # check just for 5 epochs because this gets called many times
    model.fit([train_seq['anchor'], train_seq['positive'], train_seq['negative']], Y_train, epochs=3,  batch_size=40, callbacks=callbacks_list, validation_split=0.2)
    current_model = inter_model
    # print some statistics on this epoch
    print("training data predictions")
    positives = test_positive_model.predict([train_seq['anchor'], train_seq['positive'], train_seq['negative']])
    negatives = test_negative_model.predict([train_seq['anchor'], train_seq['positive'], train_seq['negative']])
    print("f1score for train is: {}".format(f1score(positives, negatives)))
    print("test data predictions")
    positives = test_positive_model.predict([test_seq['anchor'], test_seq['positive'], test_seq['negative']])
    negatives = test_negative_model.predict([test_seq['anchor'], test_seq['positive'], test_seq['negative']])
    print("f1score for test is: {}".format(f1score(positives, negatives)))

    test_match_stats = generate_triplets_from_ANN(current_model, sequences_test, entity2unique_test, entity2same_test, unique_text_test, False)
    print("Test stats:" + str(test_match_stats))



