import glob
import cv2
import random
import numpy as np
import os
import pickle
from PIL import Image


#pourcentage d'exemples pour train le modèle
#pourcentage pour le test 1 - split
split = 0.9
nbClass = 3
pasRotation = 4 #pas de la rotation de l'image en degrée
rotation = 45
imgSize = 64

#Afin de récupérer l'ensemble des noms des images stockées 
liste = glob.glob('./image/*.png')
listeFermee = glob.glob('./image/1/**')
listeOuvert = glob.glob('./image/2/**')

listeFermee2 = glob.glob('./image/Triesch_Dataset/1/**')
listeOuvert2 = glob.glob('./image/Triesch_Dataset/2/**')


#Chargement en RAM des images trouvées
data = []
for elm in liste:
  #imread avec 0 pour ouvrir en gray scale et 1 pour ouvrir en couleur
  img = np.array(cv2.resize(cv2.imread(elm, 0), (imgSize,imgSize)))
  img = cv2.equalizeHist(img)
  """cv2.imshow('object detection', img)
  if cv2.waitKey(25) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break"""
  value = int(elm.split('\\')[1][:1])
  data.append([img,value])


for elm in listeFermee:
  img = np.array(cv2.resize(cv2.imread(elm, 0), (imgSize,imgSize)))
  value = 1
  img = cv2.equalizeHist(img)
  data.append([img,value])

for elm in listeOuvert:
  img = np.array(cv2.resize(cv2.imread(elm, 0), (imgSize,imgSize)))
  value = 2
  img = cv2.equalizeHist(img)
  data.append([img,value])

random.shuffle(data)


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

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
XClassTest, YClassTest = np.array(XClassTest), np.array(YClassTest)


save_dir = './dataTrain/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


pickle.dump(X_train, open('./dataTrain/Xtrain.dump', 'wb'))
pickle.dump(y_train, open('./dataTrain/Ytrain.dump', 'wb'))
pickle.dump(X_test, open('./dataTrain/Xtest.dump', 'wb'))
pickle.dump(y_test, open('./dataTrain/Ytest.dump', 'wb'))

pickle.dump(XClassTest, open('./dataTrain/XtestClass.dump', 'wb'))
pickle.dump(YClassTest, open('./dataTrain/YtestClass.dump', 'wb'))



print("Nombres exemples d'entrainement", len(X_train))
print("Nombres exemples de test", len(X_test))

