from sys import argv

from keras import backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Flatten, Dropout, Lambda

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model, model_from_json, Sequential

import numpy as np

from embeddings import KazumaCharEmbedding

from annoy import AnnoyIndex

import Named_Entity_Recognition_Modified

#must fix
MAX_NB_WORDS = 140000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 10

DEBUG = False
DEBUG_DATA_LENGTH = 100

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

def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return gname + pname + ".jpg"
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
def get_tokenizer(texts):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts['anchor'] + texts['negative'] + texts['positive'])
    return tokenizer

def get_sequences(texts, tokenizer):
    sequences = {}
    sequences['anchor'] = tokenizer.texts_to_sequences(texts['anchor'])
    sequences['anchor'] = pad_sequences(sequences['anchor'], maxlen=MAX_SEQUENCE_LENGTH)
    sequences['negative'] = tokenizer.texts_to_sequences(texts['negative'])
    sequences['negative'] = pad_sequences(sequences['negative'], maxlen=MAX_SEQUENCE_LENGTH)
    sequences['positive'] = tokenizer.texts_to_sequences(texts['positive'])
    sequences['positive'] = pad_sequences(sequences['positive'], maxlen=MAX_SEQUENCE_LENGTH)
    return sequences
def read_file(file_path):
    texts = {'anchor':[], 'negative':[], 'positive':[]}
    fl = open(file_path, 'r')
    i = 0
    for line in fl:
        line_array = line.split("|")
        texts['anchor'].append(line_array[0])
        texts['positive'].append(line_array[1])
        #removes the new line charecter at the end
        texts['negative'].append(line_array[2][:-1])
        i += 1
        if i > DEBUG_DATA_LENGTH and DEBUG:
            break
    return texts
def get_test(texts, sequences, percent):
    indices = np.arange(sequences['anchor'].shape[0])
    np.random.shuffle(indices)
    ret_sequence = {}
    ret_sequence['anchor'] = sequences['anchor'][indices]
    ret_sequence['positive'] = sequences['positive'][indices]
    ret_sequence['negative'] = sequences['negative'][indices]
    num_validation_samples = int(percent * sequences['anchor'].shape[0])


    ret_train = {}
    ret_train['anchor'] = ret_sequence['anchor'][:-num_validation_samples]
    ret_train['positive'] = ret_sequence['positive'][:-num_validation_samples]
    ret_train['negative'] = ret_sequence['negative'][:-num_validation_samples]

    ret_test = {}
    ret_test['anchor']= ret_sequence['anchor'][-num_validation_samples:]

    ret_test['positive']= ret_sequence['positive'][-num_validation_samples:]

    ret_test['negative'] = ret_sequence['negative'][-num_validation_samples:]

    ret_texts = {}
    texts['anchor'] = np.array(texts['anchor'])
    texts['positive'] = np.array(texts['positive'])
    texts['negative'] = np.array(texts['negative'])

    ret_texts['anchor'] = texts['anchor'][indices]
    ret_texts['positive'] = texts['positive'][indices]
    ret_texts['negative'] = texts['negative'][indices]
    return ret_train, ret_test, ret_texts

def triplet_loss(y_true, y_pred):
        margin = K.constant(1)
        return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
def assign_triplets(data, model):
    sequences = numpy.concatenate(data['positive'], data['negative'])
    unique_sequence = []
    anchor_place = {}
    unique_set = set([])
    for item in data['anchor']:
        if item not in unique_set:
            anchor_place[item] = len(unique_sequence)
            unique_sequence.append(item)
            unique_set.add(item)
    for item in sequences:
        if item not in unique_set:
            unique_sequence.append(item)
            unique_set.add(item)

    #make annoy index
    unique_sequence = np.array(unique_sequence)
    predictions = model.predict(unique_sequence)
    

    t = AnnoyIndex(len(predictions[0]), metric='euclidean')  # Length of item vector that will be indexed
    for i in range(len(predictions)):
        v = predictions[i]
        t.add_item(i, v)

    t.build(100) # 100 trees
    #create nearest neighbors list
    anchor_to_nearest = {}
    new_data = {}
    new_data['anchor'] = []
    new_data['positive'] = []
    new_data['negative'] = []
    index = 0
    while index < len(data['anchor']):
        name = data['anchor'][index]
        if name in anchor_to_nearest:
            if anchor_to_nearest[name]:
                new_data['anchor'].append(name)
                new_data['negative'].append(anchor_to_nearest[name].pop())
                new_data['positive'].append(data['positive'][index])
            index += 1
        else:
            anchor_to_nearest[name] = t.get_nns_by_item(anchor_place[name], 5)
    new_data['anchor'] = np.array(new_data['anchor'])
    new_data['positive'] = np.array(new_data['positive'])
    new_data['negative'] = np.array(new_data['negative'])

    return new_data

def do_annoy(model, texts, tokenizer, verbose):
    unique_text = []
    entity_idx = []
    entity2same = {}

    for i in range(len(texts['anchor'])):
        if not texts['anchor'][i] in entity2same:
            entity2same[texts['anchor'][i]] = []
            entity_idx.append(len(unique_text))
            unique_text.append(texts['anchor'][i])
        l = entity2same[texts['anchor'][i]]
        if texts['positive'][i] not in l:
            entity2same[texts['anchor'][i]].append(texts['positive'][i])
            unique_text.append(texts['positive'][i])

    print(entity2same)
    print(unique_text)

    sequences = tokenizer.texts_to_sequences(unique_text)
    sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(sequences)
 

    t = AnnoyIndex(len(predictions[0]), metric='euclidean')  # Length of item vector that will be indexed
    for i in range(len(predictions)):
        v = predictions[i]
        t.add_item(i, v)

    t.build(100) # 100 trees

    match = 0
    no_match = 0

    for index in entity_idx:
        nearest = t.get_nns_by_vector(predictions[index], 5)
        print(nearest)
        nearest_text = set([unique_text[i] for i in nearest])
        expected_text = set(entity2same[unique_text[index]])
        nearest_text.remove(unique_text[index])
        print("query={} names = {} true_match = {}".format(unique_text[index], nearest_text, expected_text))
        if verbose:
            print([t.get_distance(index, i) for i in nearest])
        overlap = expected_text.intersection(nearest_text)
        print(overlap)
        m = len(overlap)
        match += m
        no_match += len(expected_text) - m

    print("match: {} no_match: {}".format(match, no_match))
def print_deb_data(debbuging_data):
    for i in range(debbuging_data['number']):
        print('anch: --{}-- pos:--{}-- neg:--{}--'.format(debbuging_data['texts']['anchor'][i], debbuging_data['texts']['positive'][i], debbuging_data['texts']['negative'][i]))
        print('sequences: anch: --{}-- pos:--{}-- neg:--{}--'.format(debbuging_data['sequences']['anchor'][i], debbuging_data['sequences']['positive'][i], debbuging_data['sequences']['negative'][i]))

def debugging_text_and_sequences(reordered_text, training_data, number):
    debbuging_data = {}
    debbuging_data['number'] = number
    debbuging_data['sequences'] = {}
    debbuging_data['texts'] = {}
    debbuging_data['sequences']['anchor'] = []
    debbuging_data['sequences']['positive'] = []
    debbuging_data['sequences']['negative'] = []
    debbuging_data['texts']['anchor'] = []
    debbuging_data['texts']['positive'] = []
    debbuging_data['texts']['negative'] = []

    for i in range(number):
        debbuging_data['texts']['anchor'].append(reordered_text['anchor'][i])
        debbuging_data['texts']['positive'].append(reordered_text['positive'][i])
        debbuging_data['texts']['negative'].append(reordered_text['negative'][i])
        debbuging_data['sequences']['anchor'].append(training_data['anchor'][i])
        debbuging_data['sequences']['positive'].append(training_data['positive'][i])
        debbuging_data['sequences']['negative'].append(training_data['negative'][i])
    return debbuging_data
# triples_data = create_triples(IMAGE_DIR)
texts = read_file(argv[1])
print("anchor: {} positive: {} negative: {}".format(texts['anchor'][0], texts['positive'][0], texts['negative'][0]))
tokenizer = get_tokenizer(texts)
print('got tokenizer')
sequences = get_sequences(texts, tokenizer)
train_data, test_data, reordered_text = get_test(texts, sequences, 0.05)

debbuging_data = debugging_text_and_sequences(reordered_text, train_data, 20)


number_of_names = len(train_data['anchor'])
print('sequenced words')
Y_train = np.random.randint(2, size=(1,2,number_of_names)).T


embedder = get_embedding_layer(tokenizer)
print('got embeddings')
main_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
net = embedder(main_input)
net = Flatten(name='flatten')(net) 
net = Dense(128, activation='relu', name='embed')(net)
net = Dense(128, activation='relu', name='embed2')(net)
net = Dense(128, activation='relu', name='embed3')(net)
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

model.fit([train_data['anchor'], train_data['positive'], train_data['negative']], Y_train, epochs=5,  batch_size=15, validation_split=0.2)
test_positive_model = Model([input_anchor, input_positive, input_negative], positive_dist)
test_negative_model = Model([input_anchor, input_positive, input_negative], negative_dist)

print("training data predictions")
positives = test_positive_model.predict([train_data['anchor'], train_data['positive'], train_data['negative']])
negatives = test_negative_model.predict([train_data['anchor'], train_data['positive'], train_data['negative']])
print("f1score is: {}".format(f1score(positives, negatives)))
print("test data predictions")
positives = test_positive_model.predict([test_data['anchor'], test_data['positive'], test_data['negative']])
negatives = test_negative_model.predict([test_data['anchor'], test_data['positive'], test_data['negative']])
print("f1score is: {}".format(f1score(positives, negatives)))

# model.save('triplet_loss_resnet50.h5')

inter_model = Model(input_anchor, net_anchor)
do_annoy(inter_model, texts, tokenizer, False)
print('annoy on embeddings for debbuging_data')
do_annoy(Named_Entity_Recognition_Modified.embedded_representation(embedder), debbuging_data['texts'], tokenizer, True)
print('annoy on full model for debbuging_data')
do_annoy(inter_model, debbuging_data['texts'], tokenizer, True)
print_deb_data(debbuging_data)