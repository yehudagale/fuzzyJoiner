from sys import argv

from keras import backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Flatten, Dropout, Lambda

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model, model_from_json, Sequential

import numpy as np

from embeddings import KazumaCharEmbedding
#must fix
MAX_NB_WORDS = 140000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 10

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
    for line in fl:
        line_array = line.split("|")
        texts['anchor'].append(line_array[0])
        texts['positive'].append(line_array[1])
        texts['negative'].append(line_array[2])
    return texts
def triplet_loss(y_true, y_pred):
        margin = K.constant(0.2)
        return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# triples_data = create_triples(IMAGE_DIR)
texts = read_file(argv[1])
print("anchor: {} positive: {} negative: {}".format(texts['anchor'][0], texts['positive'][0], texts['negative'][0]))
tokenizer = get_tokenizer(texts)
print('got tokenizer')
sequences = get_sequences(texts, tokenizer)
number_of_names = len(texts['anchor'])
print('sequenced words')
# dim = 1500
# h = 299
# w= 299
# anchor =np.zeros((dim,h,w,3))
# positive =np.zeros((dim,h,w,3))
# negative =np.zeros((dim,h,w,3))


# for n,val in enumerate(triples_data[0:1500]):
#     image_anchor = plt.imread(os.path.join(IMAGE_DIR, val[0]))
#     image_anchor = imresize(image_anchor, (h, w))    
#     image_anchor = image_anchor.astype("float32")
#     #image_anchor = image_anchor/255.
#     image_anchor = keras.applications.resnet50.preprocess_input(image_anchor, data_format='channels_last')
#     anchor[n] = image_anchor

#     image_positive = plt.imread(os.path.join(IMAGE_DIR, val[1]))
#     image_positive = imresize(image_positive, (h, w))
#     image_positive = image_positive.astype("float32")
#     #image_positive = image_positive/255.
#     image_positive = keras.applications.resnet50.preprocess_input(image_positive, data_format='channels_last')
#     positive[n] = image_positive

#     image_negative = plt.imread(os.path.join(IMAGE_DIR, val[2]))
#     image_negative = imresize(image_negative, (h, w))
#     image_negative = image_negative.astype("float32")
#     #image_negative = image_negative/255.
#     image_negative = keras.applications.resnet50.preprocess_input(image_negative, data_format='channels_last')
#     negative[n] = image_negative

Y_train = np.random.randint(2, size=(1,2,number_of_names)).T


# resnet_input = Input(shape=(h,w,3))
# resnet_model = ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)


# for layer in resnet_model.layers:
#     layer.trainable = False  

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

model.fit([sequences['anchor'], sequences['positive'], sequences['negative']], Y_train, epochs=50,  batch_size=15, validation_split=0.2)

model.save('triplet_loss_resnet50.h5')
