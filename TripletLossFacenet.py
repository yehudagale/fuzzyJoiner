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

#must fix
MAX_NB_WORDS = 140000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 10

DEBUG = False
DEBUG_DATA_LENGTH = 100

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
        texts['negative'].append(line_array[2])
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
    ret_test['anchor']= ret_test['anchor'][-num_validation_samples:]

    ret_test['positive']= ret_test['positive'][-num_validation_samples:]

    ret_test['negative'] = ret_test['negative'][-num_validation_samples:]

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

def do_annoy(model, texts, tokenizer):
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
    predictions = inter_model.predict(sequences)
 

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
        overlap = expected_text.intersection(nearest_text)
        print(overlap)
        m = len(overlap)
        match += m
        no_match += len(expected_text) - m

    print("match: {} no_match: {}".format(match, no_match))

# triples_data = create_triples(IMAGE_DIR)
texts = read_file(argv[1])
print("anchor: {} positive: {} negative: {}".format(texts['anchor'][0], texts['positive'][0], texts['negative'][0]))
tokenizer = get_tokenizer(texts)
print('got tokenizer')
sequences = get_sequences(texts, tokenizer)
test_data, train_data, reordered_text = get_test(texts, sequences, 0.05)
number_of_names = len(texts['anchor'])
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

model.fit([test_data['anchor'], test_data['positive'], test_data['negative']], Y_train, epochs=5,  batch_size=15, validation_split=0.2)

# model.save('triplet_loss_resnet50.h5')

inter_model = Model(input_anchor, net_anchor)
do_annoy(inter_model, texts, tokenizer)




