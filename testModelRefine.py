import tensorflow as tf
import glob
import cv2
import random
import numpy as np
import os
import ctypes
import time
from tqdm import tqdm

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
                   dropout,            # Dropout rate
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
      layer = tf.nn.relu(layer)

    return layer, weights


# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 32
num_filters2 = 64
num_filters3 = 128


n_classes = 15
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
                   dropout=keep_prob,
                   use_pooling=False)

layer_conv1a1, weights_conv1a1 = \
    new_conv_layer("conv1a1",input=layer_conv1a,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   dropout=keep_prob,
                   use_pooling=True)

layer_conv1b, weights_conv1b = \
    new_conv_layer("conv1b",input=layer_conv1a1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters2,
                   dropout=keep_prob,
                   use_pooling=False)

layer_conv1b1, weights_conv1b1 = \
    new_conv_layer("conv1b1",input=layer_conv1b,
                   num_input_channels=num_filters2,
                   filter_size=filter_size1,
                   num_filters=num_filters2,
                   dropout=keep_prob,
                   use_pooling=True)

layer_conv1c, weights_conv1c = \
    new_conv_layer("conv1c",input=layer_conv1b1,
                   num_input_channels=num_filters2,
                   filter_size=filter_size1,
                   num_filters=num_filters2,
                   dropout=keep_prob,
                   use_pooling=False)

layer_conv1c1, weights_conv1c1 = \
    new_conv_layer("conv1c1",input=layer_conv1c,
                   num_input_channels=num_filters2,
                   filter_size=filter_size1,
                   num_filters=num_filters2,
                   dropout=keep_prob,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv1c1)

layer_f, weights_f = new_fc_layer("fc",input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=n_classes,
                         use_nonlinear=False)

y_pred = tf.nn.softmax(layer_f)
y_pred_cls = tf.argmax(y_pred, dimension=1)

print(layer_conv1a)
print(layer_flat)
print(layer_f)

correct = tf.equal(tf.argmax(layer_f, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

saver = tf.train.Saver()
save_dir = 'final_modelRefine/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_model')


gestures = ['None', 'fist', 'thumb up', 'thumb down', 'stop', \
            'catch', 'swing', 'phone', 'victory', \
            'C', 'okay', '2 fingers', '2 fingers horiz', \
            'rock&roll', 'rock&roll horiz']

cptGest = [[0,0] for f in range(len(gestures))]
t = time.time()

liste = glob.glob('imageNew/*.png')
#liste = glob.glob('image/*.png')
#liste = glob.glob('image/laouen/*.png')
random.shuffle(liste)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess=sess, save_path=save_path)
  for elm in tqdm(liste[:5000]):
    image_np = np.array(cv2.imread(elm,1))
    value = int(elm.split('\\')[1].split('_')[0])
    #cv2.imshow('object detection', cv2.resize(image_np, (400,300)))
    gray_image = cv2.cvtColor(cv2.resize(image_np, (imgSize,imgSize)), cv2.COLOR_BGR2GRAY)
    #t2 = time.time()
    #gray_image = cv2.equalizeHist(gray_image)
    result = y_pred.eval({x:[gray_image], keep_prob: 1})

    if np.argmax(result) == value:
      cptGest[value][1] += 1
      #print(gestures[np.argmax(result)], 1/(time.time() - t), 1/(time.time() - t2))
    else:
      cptGest[value][0] += 1
      cptGest[value][1] += 1
      #print(gestures[np.argmax(result)], gestures[value], 1/(time.time() - t), 1/(time.time() - t2))
      #input()

    
    t = time.time()
    if cv2.waitKey(5) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

print(cptGest)
for x,elm in enumerate(cptGest):
  print(gestures[x],elm[0]/elm[1]*100)