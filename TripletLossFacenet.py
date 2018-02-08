def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return gname + pname + ".jpg"

def create_triples(image_dir):
    img_groups = {}
    for img_file in os.listdir(image_dir):
        prefix, suffix = img_file.split(".")
        gid, pid = prefix[0:4], prefix[4:]

        if gid in img_groups.keys():
            img_groups[gid].append(pid)
        else:
            img_groups[gid] = [pid]
    pos_triples, neg_triples = [], []

    for key in img_groups.keys():
        triples = [(key + x[0] + ".jpg", key + x[1] + ".jpg", str(int(key)+3 if int(key)<1495 else int(key)-3)+'01'+'.jpg')
                 for x in itertools.combinations(img_groups[key], 2)]
        pos_triples.extend(triples)

    return pos_triples

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

triples_data = create_triples(IMAGE_DIR)


dim = 1500
h = 299
w= 299
anchor =np.zeros((dim,h,w,3))
positive =np.zeros((dim,h,w,3))
negative =np.zeros((dim,h,w,3))


for n,val in enumerate(triples_data[0:1500]):
    image_anchor = plt.imread(os.path.join(IMAGE_DIR, val[0]))
    image_anchor = imresize(image_anchor, (h, w))    
    image_anchor = image_anchor.astype("float32")
    #image_anchor = image_anchor/255.
    image_anchor = keras.applications.resnet50.preprocess_input(image_anchor, data_format='channels_last')
    anchor[n] = image_anchor

    image_positive = plt.imread(os.path.join(IMAGE_DIR, val[1]))
    image_positive = imresize(image_positive, (h, w))
    image_positive = image_positive.astype("float32")
    #image_positive = image_positive/255.
    image_positive = keras.applications.resnet50.preprocess_input(image_positive, data_format='channels_last')
    positive[n] = image_positive

    image_negative = plt.imread(os.path.join(IMAGE_DIR, val[2]))
    image_negative = imresize(image_negative, (h, w))
    image_negative = image_negative.astype("float32")
    #image_negative = image_negative/255.
    image_negative = keras.applications.resnet50.preprocess_input(image_negative, data_format='channels_last')
    negative[n] = image_negative

Y_train = np.random.randint(2, size=(1,2,dim)).T


resnet_input = Input(shape=(h,w,3))
resnet_model = ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)


for layer in resnet_model.layers:
    layer.trainable = False  


net = resnet_model.output
net = Flatten(name='flatten')(net) 
net = Dense(128, activation='relu', name='embed')(net)
net = Dense(128, activation='relu', name='embed2')(net)
net = Dense(128, activation='relu', name='embed3')(net)
net = Lambda(l2Norm, output_shape=[128])(net)

base_model = Model(resnet_model.input, net, name='resnet_model')

input_shape=(h,w,3)
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

model.fit([anchor, positive, negative], Y_train, epochs=50,  batch_size=15, validation_split=0.2)

model.save('triplet_loss_resnet50.h5')