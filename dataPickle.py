import glob
import cv2
import random
import numpy as np
import os
import pickle
from PIL import Image


liste = glob.glob('./image/**')
data = []
for elm in liste:
  img = np.array(cv2.imread(elm, 0))
  value = int(elm.split('\\')[1][:1])
  data.append([img,value])
random.shuffle(data)

X_train = []
y_train = []
data_train = []
for elm in data[:int(len(data)*0.90)]:
  classe = np.zeros(6)
  classe[elm[1]] = 1
  img1 = Image.fromarray(elm[0])
  img2 = Image.fromarray(np.flip(elm[0],1))
  data_train.append([np.flip(elm[0],1), classe])
  data_train.append([elm[0], classe])
  for x in range(20, 360, 20):
    img1a = img1.rotate(x)
    img2a = img2.rotate(x)
    data_train.append([np.array(img1a), classe])
    data_train.append([np.array(img2a), classe])

X_test = []
y_test = []
data_test = []
for elm in data[int(len(data)*0.90):]:
  classe = np.zeros(6)
  classe[elm[1]] = 1
  img1 = Image.fromarray(elm[0])
  img2 = Image.fromarray(np.flip(elm[0],1))
  data_test.append([np.flip(elm[0],1), classe])
  data_test.append([elm[0], classe])
  for x in range(20, 360, 20):
    img1a = img1.rotate(x)
    img2a = img2.rotate(x)
    data_test.append([np.array(img1a), classe])
    data_test.append([np.array(img2a), classe]) 

data = 0
random.shuffle(data_test)
random.shuffle(data_train)

for elm in data_train:
  X_train.append(elm[0])
  y_train.append(elm[1])
data_train = 0

for elm in data_test:
  X_test.append(elm[0])
  y_test.append(elm[1])
data_test = 0

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

save_dir = './dataTrain/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


pickle.dump(X_train, open('./dataTrain/Xtrain.dump', 'wb'))
pickle.dump(y_train, open('./dataTrain/Ytrain.dump', 'wb'))
pickle.dump(X_test, open('./dataTrain/Xtest.dump', 'wb'))
pickle.dump(y_test, open('./dataTrain/Ytest.dump', 'wb'))




print("Nombres exemples d'entrainement", len(X_train))
print("Nombres exemples de test", len(X_test))

