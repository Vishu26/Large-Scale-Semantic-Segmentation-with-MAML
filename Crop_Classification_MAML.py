import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Activation, BatchNormalization, Reshape, Dropout
import tensorflow.python.keras.backend as K
sess = K.get_session()
from tensorflow.compat.v1.keras.backend import set_session
import imp, h5py
import pickle
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.compat.v1.Session(config=config))

def build_FCN(nrows, ncols, nbands, NUMBER_CLASSES):
    """Function to create Keras model of sample network."""
    model = keras.models.Sequential()
    model.add(ZeroPadding2D((3, 3), input_shape=(nrows, ncols, nbands)))
    model.add(Convolution2D(
              filters=32,
              kernel_size=(7, 7),
              dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(
              pool_size=(3, 3),
              strides=(1, 1)
    ))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(
              filters=32,
              kernel_size=(5, 5),
              dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l1(0.01)
    ))
    model.add(BatchNormalization(axis=3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1)
    ))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(
              filters=32,
              kernel_size=(5, 5),
              dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l1(0.01)
    ))
    model.add(BatchNormalization(axis=3))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1)
    ))
    model.add(Dropout(0.25))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(
              filters=32,
              kernel_size=(5, 5),
              dilation_rate=(1, 1),kernel_regularizer=tf.keras.regularizers.l1(0.01)
    ))
    model.add(BatchNormalization(axis=3))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1)
    ))
    model.add(keras.layers.Conv2D(
              filters=NUMBER_CLASSES,
              kernel_size=(1, 1),kernel_regularizer=tf.keras.regularizers.l1(0.01)
    ))
    model.add(keras.layers.Activation(
              activation="softmax"
    ))
    
    return model

patchXT = np.load('Xtrain.npz')['arr_0']
patchYT = np.load('Ytrain.npz')['arr_0']
patchtx = np.load('Xvalid.npz')['arr_0']
patchty = np.load('Yvalid.npz')['arr_0']
Ytone = np.load('ytone.npz')['arr_0']
Yvone = np.load('yvone.npz')['arr_0']


def mean_iou(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
        jac = (intersection + 1e-7) / (sum_ - intersection + 1e-7)
        return jac

    
def jaccard_distance(smooth=100):
    def jd(y_true, y_pred):
        """ Calculates mean of Jaccard distance as a loss function """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd =  (1 - jac) * smooth
        return tf.reduce_mean(jd)
    return jd


model = build_FCN(128, 128, 8, 5)
model.compile(loss=jaccard_distance(), metrics=['acc', mean_iou])
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.models import clone_model

inner_optimizer = tf.keras.optimizers.SGD(learning_rate=0.007)
outer_optimizer = tf.keras.optimizers.SGD(learning_rate=0.007)
cce = jaccard_distance()

trained_idx = defaultdict(list)

train_acc, test_acc = [], []

for i in tqdm(range(20)):

    model_weights = model.get_weights()
    task_weights = []
    query_sets_X, query_sets_Y = [], []
    batch_loss, batch_acc = [], []
    total_loss = 0

    for t in range(20):

        idx = np.random.choice(len(patchXT), size=5)
        patchX, patchY = patchXT[idx], Ytone[idx]

        # Generate Support Set X
        #support_idx = np.random.choice(range(patchX.shape[0]), size=3*patchX.shape[0]//4)
        support_set_X = patchX[:1]

        # Generate Query Set X
        #query_set_X = patchX[np.delete(range(patchX.shape[0]), support_idx)]
        query_set_X = patchX[1:]
        query_sets_X.append(query_set_X)

        # Generate Support Set Y
        #support_set_Y = patchY[support_idx]
        support_set_Y = patchY[:1]
        #support_set_Y = to_categorical_4d(support_set_Y.astype(np.uint8), 5)

        # Generate Query Set Y
        #query_set_Y = patchY[np.delete(range(patchX.shape[0]), support_idx)]
        query_set_Y = patchY[1:]
        #query_set_Y = to_categorical_4d(query_set_Y.astype(np.uint8), 5)
        query_sets_Y.append(query_set_Y)

        # Meta Train
        copy_model = clone_model(model)
        for _ in range(5):
            #for batch in range(0, support_set_X.shape[0], 16):
            with tf.GradientTape() as train_tape:
                train_loss = cce(support_set_Y, copy_model(support_set_X))
                #train_loss = tf.reduce_mean(train_loss)
            # Step 6
            gradients = train_tape.gradient(train_loss, copy_model.trainable_variables)
            inner_optimizer.apply_gradients(zip(gradients, copy_model.trainable_variables))

        task_weights.append(copy_model.get_weights())

    # Meta Update
    with tf.GradientTape() as test_tape:
        for m in range(20):
            model.set_weights(task_weights[m])
        #with tf.GradientTape() as test_tape:
        #weights = model.get_weights()
            if m==0:
                test_loss = cce(query_sets_Y[m], model(query_sets_X[m]))
            else:
                test_loss += cce(query_sets_Y[m], model(query_sets_X[m]))

    model.set_weights(model_weights)
    gradients = test_tape.gradient(test_loss, model.trainable_variables)
    outer_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    test_acc.append(model.evaluate(patchtx, Yvone, batch_size=32, verbose=1)[0])
    train_acc.append(model.evaluate(patchXT, Ytone, batch_size=32, verbose=1)[0])
    print(F"Train-acc: {train_acc[-1]}")
    print(F"Test-acc: {test_acc[-1]}")


idd = np.random.choice(len(patchtx), size=10)

s_test_X = patchtx[idd[:1]]
s_test_Y = Yvone[idd[:1]]
q_test_X = patchtx[idd[1:]]
q_test_Y = Yvone[idd[1:]]

weight = model.get_weights()
for _ in range(5):
            #for batch in range(0, support_set_X.shape[0], 16):
    with tf.GradientTape() as train_tape:
        train_loss = cce(s_test_Y, model(s_test_X))
                #train_loss = tf.reduce_mean(train_loss)
            # Step 6
    gradients = train_tape.gradient(train_loss, model.trainable_variables)
    inner_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

with tf.GradientTape() as test_tape:
    test_loss = cce(q_test_Y, model(q_test_X))
model.set_weights(weight)
gradients = test_tape.gradient(test_loss, model.trainable_variables)
outer_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
print(model.evaluate(patchtx, Yvone, batch_size=32, verbose=1))
