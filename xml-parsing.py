import glob
import bs4 as bs
import cv2
import pickle
import numpy as np
import random
from PIL import Image
import os

liste = glob.glob('final_model16/feature_points/*.xml')
images = glob.glob('final_model16/original_images/*.jpg')
imgSize = 64
split = 0.9

type = ['WristThumb', 'WristPinky', 'Knuckle1', 'Knuckle2', 'FingerTip']
data = []
for x, elm in enumerate(liste[:]):
	with open(elm, "r" ) as f:
		html_doc = f.read()
	soup = bs.BeautifulSoup(html_doc, 'html5lib')
	imgDim = soup.find_all('img')
	w, h = int(imgDim[0].get('width')), int(imgDim[0].get('height'))

	df = len(data)
	for p in soup.find_all('hand'):
		X, Y = [], []
		for child in p.findChildren():
			X.append(int(child.get('x')))
			Y.append(int(child.get('y')))
		if len(X) > 0:
			xa, xb = max(X)/w, min(X)/w
			ya, yb = max(Y)/h, min(Y)/h
	if len(data) - 1 > df:
		print('Noonnnnn')
	image = np.array(cv2.resize(cv2.imread(images[x], 0), (imgSize, imgSize)))

	data.append([image, xa,ya,xb,yb])
	"""cv2.line(image, (int(xa*imgSize),int(ya*imgSize)), (int(xa*imgSize),int(yb*imgSize)), (255,0,0), 2)
	cv2.line(image, (int(xa*imgSize),int(yb*imgSize)), (int(xb*imgSize),int(yb*imgSize)), (255,0,0), 2)
	cv2.line(image, (int(xb*imgSize),int(ya*imgSize)), (int(xb*imgSize),int(yb*imgSize)), (255,0,0), 2)
	cv2.line(image, (int(xa*imgSize),int(ya*imgSize)), (int(xb*imgSize),int(ya*imgSize)), (255,0,0), 2)
	cv2.imshow('images',image)
	key = cv2.waitKey(2) & 0x0F

cv2.destroyAllWindows()"""

random.shuffle(data)

#Traitement des images pour l'entrainement du modèle
X_train = []
y_train = []
data_train = []
for elm in data[:int(len(data)*split)]:
  img1 = Image.fromarray(elm[0])
  img2 = Image.fromarray(np.flip(elm[0],1))

  """cv2.imshow('elm0', elm[0])
  cv2.imshow('elm0 + 90', np.array(Image.fromarray(elm[0]).rotate(90)))
  cv2.imshow('elm0 flip1', np.flip(elm[0],1))
  cv2.imshow('elm0 flip1 + 90', np.array(Image.fromarray(np.flip(elm[0],1)).rotate(90)))
  cv2.imshow('elm0 flip0', np.flip(elm[0],0))
  cv2.imshow('elm0 flip0 + 90', np.array(Image.fromarray(np.flip(elm[0],0)).rotate(90)))
  cv2.imshow('elm0 flip', np.flip(np.flip(elm[0],0),1))
  cv2.imshow('elm0 flip + 90', np.array(Image.fromarray(np.flip(np.flip(elm[0],0),1)).rotate(90)))

  key = cv2.waitKey(100000) & 0x0F
  cv2.destroyAllWindows()
  df()"""
  """print(elm[1], elm[2], elm[3], elm[4])
  print(elm[2], 1-elm[3], elm[4], 1-elm[1])
  print(elm[1], 1-elm[4], elm[3], 1-elm[2])
  print(1-elm[4], 1-elm[3], 1-elm[2], 1-elm[1])
  print(1-elm[3], 1-elm[4], 1-elm[1], 1-elm[2])
  print(1-elm[4], elm[1], 1-elm[2], elm[3])
  print(1-elm[3], elm[2], 1-elm[1], elm[4])
  print(elm[2], elm[1], elm[4], elm[3])
  input()"""
  data_train.append([elm[0], [elm[1], elm[2], elm[3], elm[4]]])
  data_train.append([np.array(Image.fromarray(elm[0]).rotate(90)), [elm[2], 1-elm[3], elm[4], 1-elm[1]]])
  
  data_train.append([np.flip(elm[0],0), [elm[1], 1-elm[4], elm[3], 1-elm[2]]])
  data_train.append([np.array(Image.fromarray(np.flip(elm[0],0)).rotate(90)), [1-elm[4], 1-elm[3], 1-elm[2], 1-elm[1]]])
  
  data_train.append([np.flip(np.flip(elm[0],1),0), [1-elm[3], 1-elm[4], 1-elm[1], 1-elm[2]]])
  data_train.append([np.array(Image.fromarray(np.flip(np.flip(elm[0],1),0)).rotate(90)), [1-elm[4], elm[1], 1-elm[2], elm[3]]])
  
  data_train.append([np.flip(elm[0],1), [1-elm[3], elm[2], 1-elm[1], elm[4]]])
  data_train.append([np.array(Image.fromarray(np.flip(elm[0],1)).rotate(90)), [elm[2], elm[1], elm[4], elm[3]]])
  
  """for x in range(-rotation, rotation, pasRotation):
    img1a = img1.rotate(x)
    img2a = img2.rotate(x)
    data_train.append([np.array(img1a), classe])
    data_train.append([np.array(img2a), classe])"""


#Traitement des images pour le test du modèle
X_test = []
y_test = []
data_test = []
for elm in data[int(len(data)*split):]:
  img1 = Image.fromarray(elm[0])
  img2 = Image.fromarray(np.flip(elm[0],1))
  
  data_test.append([elm[0], [elm[1], elm[2], elm[3], elm[4]]])
  data_test.append([np.array(Image.fromarray(elm[0]).rotate(90)), [elm[2], 1-elm[3], elm[4], 1-elm[1]]])
  
  data_test.append([np.flip(elm[0],0), [elm[1], 1-elm[4], elm[3], 1-elm[2]]])
  data_test.append([np.array(Image.fromarray(np.flip(elm[0],0)).rotate(90)), [1-elm[4], 1-elm[3], 1-elm[2], 1-elm[1]]])
  
  data_test.append([np.flip(np.flip(elm[0],1),0), [1-elm[3], 1-elm[4], 1-elm[1], 1-elm[2]]])
  data_test.append([np.array(Image.fromarray(np.flip(np.flip(elm[0],1),0)).rotate(90)), [1-elm[4], elm[1], 1-elm[2], elm[3]]])
  
  data_test.append([np.flip(elm[0],1), [1-elm[3], elm[2], 1-elm[1], elm[4]]])
  data_test.append([np.array(Image.fromarray(np.flip(elm[0],1)).rotate(90)), [elm[2], elm[1], elm[4], elm[3]]])
  """for x in range(-rotation, rotation, pasRotation):
    img1a = img1.rotate(x)
    img2a = img2.rotate(x)
    data_test.append([np.array(img1a), classe])
    data_test.append([np.array(img2a), classe])"""

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


save_dir = './dataTrain_box/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


pickle.dump(X_train, open('./dataTrain_box/Xtrain.dump', 'wb'))
pickle.dump(y_train, open('./dataTrain_box/Ytrain.dump', 'wb'))
pickle.dump(X_test, open('./dataTrain_box/Xtest.dump', 'wb'))
pickle.dump(y_test, open('./dataTrain_box/Ytest.dump', 'wb'))


print("Nombres exemples d'entrainement", len(X_train))
print("Nombres exemples de test", len(X_test))