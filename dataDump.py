
""" Script to transform images saved into training examples.
	All images are open and resize to a same size.
	After some copy are made to create even more examples from differentes rotations of the original image

"""

import glob
import cv2
import random
import numpy as np
import os
import pickle
from PIL import Image
import sys
from threading import Thread, RLock
from time import time


#Creation of a lock
rlock = RLock()

#Storage of training examples and size of these images
data = []
imgSize = 64

#Class to open in thread all the images store on the disk
class OpenImage(Thread):
    """ Thread for open images. """
    def __init__(self, listA):
        global data, imgSize
        Thread.__init__(self)
        self.listA = listA
        self.img, self.value = None, None

    def run(self):
        """ Code to execute to open. """
        i = 0
        for elm in self.listA:
            self.value = int(elm.split('\\')[1].split('_')[0])
            self.img = np.array(cv2.resize(cv2.imread(elm, 0), (imgSize,imgSize)))
            with rlock:
                data.append([self.img, self.value])


def recup(liste):
	global data
	#Load in RAM of all images on the disk
	# Threads creation
	threads = []

	nbThread = 20
	size = int(len(liste)/nbThread)
	for x in range(nbThread):
	    threads.append(OpenImage(liste[x*size:(x+1)*size]))

	# Run of all threads
	for thread in threads:
	    thread.start()


	# Waiting for all threads
	for thread in threads:
	    thread.join()

	return None


def dataTraitement():
	global data
	rotationStep = 10 #rotation step in degree
	rotation = 30 #max rotation
	split = 0.90 #porcentage of split set
	nbClass = 15 #14 gestures
	print('Load in memory done ...')
	#Traitement of images flip, rotation, ... for training
	X_train = []
	y_train = []
	data_train = []
	for elm in data[:int(len(data)*split)]:
	  classe = np.zeros(nbClass)
	  classe[elm[1]] = 1
	  img1 = Image.fromarray(elm[0])
	  img2 = Image.fromarray(np.flip(elm[0],1))
	  data_train.append([np.flip(elm[0],1), classe])
	  data_train.append([elm[0], classe])
	  for x in range(-rotation, rotation, rotationStep):
	    img1a = img1.rotate(x)
	    img2a = img2.rotate(x)
	    data_train.append([np.array(img1a), classe])
	    data_train.append([np.array(img2a), classe])

	print('Traitement data_train done ...')
	#Traitement of images flip, rotation, ... for testing
	X_test = []
	y_test = []
	data_test = []
	for elm in data[int(len(data)*split):]:
	  classe = np.zeros(nbClass)
	  classe[elm[1]] = 1
	  img1 = Image.fromarray(elm[0])
	  img2 = Image.fromarray(np.flip(elm[0],1))
	  data_test.append([np.flip(elm[0],1), classe])
	  data_test.append([elm[0], classe])
	  for x in range(-rotation, rotation, rotationStep):
	    img1a = img1.rotate(x)
	    img2a = img2.rotate(x)
	    data_test.append([np.array(img1a), classe])
	    data_test.append([np.array(img2a), classe])

	print('Traitement data_test done ...')
	data = []
	random.shuffle(data_test)
	random.shuffle(data_train)

	XClassTest = [[] for x in range(nbClass)]
	YClassTest = [[] for y in range(nbClass)]
	for elm in data_test:
	  x = np.argmax(elm[1])
	  YClassTest[x].append(elm[1])
	  XClassTest[x].append(elm[0])


	for elm in data_train:
	  X_train.append(elm[0])
	  y_train.append(elm[1])
	data_train = 0

	for elm in data_test:
	  X_test.append(elm[0])
	  y_test.append(elm[1])
	data_test = 0

	X_train, y_train, X_test, y_test = np.array(X_train, dtype=np.uint8), np.array(y_train, dtype=np.uint8), np.array(X_test, dtype=np.uint8), np.array(y_test, dtype=np.uint8)
	XClassTest, YClassTest = np.array(XClassTest), np.array(YClassTest)
	print('Ready to dump')
	return X_train, X_test, XClassTest, y_train, y_test, YClassTest




liste = glob.glob('../../image/*.png')
listeLaouen = glob.glob('../../image/laouen/*.png')
liste = liste + listeLaouen

random.shuffle(liste)

print(len(liste), 'images to load !')

""" Prevent memory error images a load in batches of 20000 images and after save
	At the end there is 3-4 files of 1.5 Go each
"""

batch_size = 20000
for x in range(0,len(liste),batch_size):
	recup(liste[x:x+batch_size])
	print(x,len(data))

	random.shuffle(data)

	X_train, X_test, XClassTest, y_train, y_test, YClassTest = dataTraitement()

	print('Ready to dump')

	save_dir = './dataTrain/'
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)


	print("Number of images for training part :", len(X_train))
	print("Number of images for testing part :", len(X_test))

	np.save('./dataTrain/Ytest_'+str(x), y_test)
	y_test = 0
	np.save('./dataTrain/Ytrain_'+str(x), y_train)
	y_train = 0
	np.save('./dataTrain/YtestClass_'+str(x), YClassTest)
	YClassTest = 0

	np.save('./dataTrain/XtestClass_'+str(x), XClassTest)
	XClassTest = 0
	np.save('./dataTrain/Xtest_'+str(x), X_test)
	X_test = 0
	np.save('./dataTrain/Xtrain_'+str(x), X_train)
	X_train = 0


