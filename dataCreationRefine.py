
""" Script to save examples of gestures in a folder to train a model after.
	This script open a thread on a webcam to take picture of gestures made by a person.
	All this pictures are resize to be store and use after in the training part. """

import cv2
import numpy as np
import os
from time import time, sleep
import glob
import tensorflow as tf
import sys

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
save_dir = 'final_modelRetrain/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_model')

if len(sys.argv) < 2:
	print("You forget to give a number of pictures to take (25 by default)")
	sys.argv.append("25")


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

# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()
cap = VideoCamera()

#Size of saved images, camera, and numbers of gestures
imgSize = 256
cameraSize = (800, 600)

gestures = ['None', 'fist', 'thumb up', 'thumb down', \
            'stop', 'catch', 'swing', 'phone', 'victory', \
            'C', 'okay', '2 fingers', '2 fingers horiz', \
            'rock&roll', 'rock&roll horiz']
nbClass = len(gestures)

# List of all gestures
# 0 => None
# 1 => Fist
# 2 => Thumb Up
# 3 => Thumb Down
# 4 => Stop
# 5 => Catch
# 6 => Swing
# 7 => Phone
# 8 => Victory
# 9 => C
# 10 => Okay
# 11 => 2 fingers
# 12 => 2 fingers horizontal
# 13 => rock and roll
# 14 => rock and roll horizontal


#Main function where pictures are taken and saved
def captureGesture(numGestures):
	global maxValue
	t = time()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess=sess, save_path=save_path)
		cpt = maxValue
		pauseState = True
		print('Pause :', pauseState, 'Press SPACE to start')

		while cpt <= maxValue + int(sys.argv[1]):
			image_np = cap.get_frame()

			cv2.imshow('object detection', cv2.resize(image_np, cameraSize))
			if time() - t > 0.1 and not(pauseState):
				save_image = cv2.resize(image_np, (imgSize,imgSize))
				gray_image = cv2.cvtColor(cv2.resize(image_np, (64,64)), cv2.COLOR_BGR2GRAY)
				gray_image = cv2.equalizeHist(gray_image)
				#print(gray_image)
				result = y_pred.eval({x:[gray_image], keep_prob: 1})

				print(gestures[np.argmax(result)])
				if np.argmax(result) != numGestures: 
					print('New picture', cpt)
					cpt += 1
					t = time()
					cv2.imwrite('./imageNew/' + str(numGestures) + '_' + str(cpt) +'.png', save_image)

			key = cv2.waitKey(25) & 0xFF 
			if key == ord(' '):
				pauseState = not(pauseState)
				print('Pause :', pauseState, 'Press SPACE to change state')	
			elif key == ord('q'):
				cv2.destroyAllWindows()
				break

#Create a save folder if it doesn't exist
save_dir = 'imageNew/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


#Automatically get the last index of saved images
liste = glob.glob(save_dir + '*.png')
maxValue = -1
for elm in liste:
	value = int(elm.split('_')[1].split('.')[0])
	if value > maxValue:
		maxValue = value

#Now we take images for each gesture
for g in range(nbClass):
	print('Run capture :', gestures[g])
	captureGesture(g)
