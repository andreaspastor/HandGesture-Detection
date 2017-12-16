import tensorflow as tf
import glob
import cv2
import random
import numpy as np
import os
import ctypes
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print(self.video.isOpened())

    def __del__(self):
        self.video.release()
    
    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

#info sur le geste à faire
#recup auto du numero de l'image à ajouter 0_{nb}
#save in color or in gray

# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()
cap = VideoCamera()
cameraSize = (800, 600)
saveSize = 256

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def recup(folder):
  X_test = np.load('./'+ folder + '/Xtest.npy')
  y_test = np.load('./'+ folder + '/Ytest.npy')
  return X_test, y_test

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


"""X_test, y_test = recup('dataTrain')
print(len(X_test))


print(X_test[0])
print('')
print(y_test[0])

input("recuperation done")"""
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
save_dir = 'final_model/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_model')


gestures = ['None', 'fist', 'thumb up', 'thumb down', 'stop', 'catch', \
            'swing', 'phone', 'victory','C', 'okay', '2 fingers', \
            '2 fingers Horiz', 'rock&roll', 'rock&roll Horiz']



images, labels = [], []
def main(g):
  global images, labels, imgSize, saveSize
  t = time() + 1
  cpt = 0
  pauseState = True
  print('Pause :', pauseState, 'Press SPACE to start')

  while cpt <= 25:
    image_np = cap.get_frame()

    cv2.imshow('object detection', cv2.resize(image_np, cameraSize))
    if time() - t > 0.1 and not(pauseState):
      print('shoot', cpt)
      color_image = cv2.cv2.resize(image_np, (saveSize,saveSize))
      name = './image/' + str(g) + '_' + str(cpt) +'.png'
      cv2.imwrite(name, color_image)
      images.append(np.array(cv2.resize(cv2.imread(name, 0), (imgSize,imgSize))))
      classes = np.zeros(n_classes)
      classes[g] = 1
      labels.append(classes)
      t = time()
      cpt += 1

    key = cv2.waitKey(25) & 0xFF 
    if key == ord(' '):
      pauseState = not(pauseState)
      print('Pause :', pauseState, 'Press SPACE to change state')
    elif key == ord('q'):
      cv2.destroyAllWindows()
      break

save_dir = 'image/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



for g in range(n_classes):
  print('Lancement main :', gestures[g])
  main(g)

X_test = np.array(images)
y_test = np.array(labels)

batch_size = int(X_test.shape[0]/4)
X_test = X_test[:4*batch_size]
y_test = y_test[:4*batch_size]

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess=sess, save_path=save_path)
  size_sample = int(len(X_test)/4)
  y_ = y_pred.eval({x:X_test[:batch_size], keep_prob: 1})
  for g in range(batch_size,size_sample,batch_size):
    print(g, ' / ', size_sample)
    y_ = np.vstack((y_,y_pred.eval({x:X_test[g:g+batch_size], keep_prob: 1})))


cnf_matrix = confusion_matrix(np.argmax(y_test[:g+batch_size],1), np.argmax(y_,1))
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True, classes=gestures,
                      title='Confusion matrix, without normalization')
plt.show()


