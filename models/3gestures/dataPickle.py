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

rlock = RLock()
class OpenImage(Thread):
    """ Thread for open images. """
    def __init__(self, listA):
        global data
        Thread.__init__(self)
        self.listA = listA
        self.img, self.value = None, None

    def run(self):
        """ Code to execute to open. """
        i = 0
        for elm in self.listA:
            self.value = int(elm.split('\\')[1].split('_')[0])
            if self.value in [0,1,2,3]:
              self.img = np.array(cv2.resize(cv2.imread(elm, 0), (imgSize,imgSize)))
              with rlock:
                  data.append([self.img, self.value])

liste = glob.glob('../../image/*.png')
listeLaouen = glob.glob('../../image/laouen/*.png')

random.shuffle(liste)
random.shuffle(listeLaouen)
#pourcentage d'exemples pour train le modèle
#pourcentage pour le test 1 - split
split = 0.90
nbClass = 4
pasRotation = 10 #pas de la rotation de l'image en degrée
rotation = 30
imgSize = 64

data = []
#Chargement en RAM des images trouvées
# Threads Creation
t1 = time()
threads = []

nbThread = 20
size = int(len(liste)/nbThread)
for x in range(nbThread):
    threads.append(OpenImage(liste[x*size:(x+1)*size]))

# Lancement des threads
for thread in threads:
    thread.start()


# Attend que les threads se terminent
for thread in threads:
    thread.join()

size = int(len(listeLaouen)/nbThread)
threads2 = []
for x in range(nbThread):
    threads2.append(OpenImage(listeLaouen[x*size:(x+1)*size]))

# Lancement des threads
for thread in threads2:
    thread.start()

# Attend que les threads se terminent
for thread in threads2:
    thread.join()

print('len de data', len(data), time() - t1)

print('Chargement en RAM des images done ...')
#Traitement des images pour l'entrainement du modèle
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
  for x in range(-rotation, rotation, pasRotation):
    img1a = img1.rotate(x)
    img2a = img2.rotate(x)
    data_train.append([np.array(img1a), classe])
    data_train.append([np.array(img2a), classe])

print('Traitement data_train done ...')
#Traitement des images pour le test du modèle
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
  for x in range(-rotation, rotation, pasRotation):
    img1a = img1.rotate(x)
    img2a = img2.rotate(x)
    data_test.append([np.array(img1a), classe])
    data_test.append([np.array(img2a), classe])

print('Traitement data_test done ...')
data = 0
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

save_dir = './dataTrain/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



np.save('./dataTrain/Xtest', X_test)
print("Nombres exemples de test", len(X_test))
X_test = 0
np.save('./dataTrain/Ytest', y_test)
y_test = 0

np.save('./dataTrain/XtestClass', XClassTest)
XClassTest = 0
np.save('./dataTrain/YtestClass', YClassTest)
YClassTest = 0

np.save('./dataTrain/Ytrain', y_train)
y_train = 0
np.save('./dataTrain/Xtrain', X_train)
print("Nombres exemples d'entrainement", len(X_train))
X_train = 0





