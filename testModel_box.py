import tensorflow as tf
import glob
import cv2
import random
import numpy as np
import os
import ctypes
import time

def new_weights_conv(name,shape):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
           initializer=tf.contrib.layers.xavier_initializer_conv2d())

def new_weights_fc(name,shape):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
           initializer=tf.contrib.layers.xavier_initializer())
       
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length], dtype=tf.float32), dtype=tf.float32)

def new_conv_layer(name,input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights_conv(name,shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights
  
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(name,input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs, use_nonlinear):
    weights = new_weights_fc(name,[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_nonlinear:
      layer = tf.nn.sigmoid(layer)

    return layer, weights


# Convolutional Layer 1.
filter_size1 = 5
num_filters1 = 32
num_filters2 = 64
num_filters3 = 128


n_classes = 4
batch_size = 256
imgSize = 64

x = tf.placeholder(tf.float32, [None, imgSize, imgSize])
x_image = tf.reshape(x, [-1, imgSize, imgSize, 1])
y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

layer_conv1a, weights_conv1a = \
    new_conv_layer("conv1a",input=x_image,
                   num_input_channels=1,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=False)

layer_conv1a1, weights_conv1a1 = \
    new_conv_layer("conv1a1",input=layer_conv1a,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv1b, weights_conv1b = \
    new_conv_layer("conv1b",input=layer_conv1a1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=False)

layer_conv1b1, weights_conv1b1 = \
    new_conv_layer("conv1b1",input=layer_conv1b,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv1c, weights_conv1c = \
    new_conv_layer("conv1c",input=layer_conv1b1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=False)

layer_conv1c1, weights_conv1c1 = \
    new_conv_layer("conv1c1",input=layer_conv1c,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv1c1)

layer_f, weights_f = new_fc_layer("fc",input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=n_classes,
                         use_nonlinear=True)

y_pred = layer_f

print(layer_conv1a)
print(layer_flat)
print(layer_f)

import pickle
def recup(folder):
  X_train = pickle.load(open('./'+ folder + '/Xtrain.dump', 'rb'))
  X_test = pickle.load(open('./'+ folder + '/Xtest.dump', 'rb'))
  y_test = pickle.load(open('./'+ folder + '/Ytest.dump', 'rb'))
  y_train = pickle.load(open('./'+ folder + '/Ytrain.dump', 'rb'))
  return X_train, y_train, X_test, y_test


saver = tf.train.Saver()
save_dir = 'final_model_box/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_model')

X_train, y_train, X_test, y_test = recup('dataTrain_box')
images = glob.glob('final_model16/original_images/*.jpg')

cap = cv2.VideoCapture(0)
t = time.time()
cpt = 0
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess=sess, save_path=save_path)
  for h, elm in enumerate(X_train):
    #ret, image_np = cap.read()
    image_np = elm
    gray_image = elm
    #gray_image = cv2.cvtColor(cv2.resize(image_np, (imgSize,imgSize)), cv2.COLOR_BGR2GRAY)
    t2 = time.time()
    gray_image = cv2.equalizeHist(gray_image)
    xa, ya, xb, yb = y_pred.eval({x:[gray_image]})[0]

    print(1/(time.time() - t), 1/(time.time() - t2))
    image = cv2.resize(image_np, (400,300))
    cv2.line(image, (int(xa*400),int(ya*300)), (int(xa*400),int(yb*300)), (255,0,0), 2)
    cv2.line(image, (int(xa*400),int(yb*300)), (int(xb*400),int(yb*300)), (255,0,0), 2)
    cv2.line(image, (int(xb*400),int(ya*300)), (int(xb*400),int(yb*300)), (255,0,0), 2)
    cv2.line(image, (int(xa*400),int(ya*300)), (int(xb*400),int(ya*300)), (255,0,0), 2)
    xa1, ya1, xb1, yb1 = xa, ya, xb, yb
    xa, ya, xb, yb = y_train[h]
    cv2.line(image, (int(xa*400),int(ya*300)), (int(xa*400),int(yb*300)), (0,0,0), 2)
    cv2.line(image, (int(xa*400),int(yb*300)), (int(xb*400),int(yb*300)), (0,0,0), 2)
    cv2.line(image, (int(xb*400),int(ya*300)), (int(xb*400),int(yb*300)), (0,0,0), 2)
    cv2.line(image, (int(xa*400),int(ya*300)), (int(xb*400),int(ya*300)), (0,0,0), 2)
    
    val = ((xa1-xa)**2+(ya1-ya)**2+(xb1-xb)**2+(yb1-yb)**2)/4
    print(val)
    cpt += val
    cv2.imshow('object detection', image)
    t = time.time()
    if cv2.waitKey(500) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

cpt /= len(X_train)
print(cpt)
print(cpt**0.5)