import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time, sleep


#info sur le geste à faire
#recup auto du numero de l'image à ajouter 0_{nb}
#save in color or in gray

cap = cv2.VideoCapture(0)
imgSize = 256
cameraSize = (800, 600)
nbClass = 6

def main(x):
	t = time() + 1
	cpt = 0
	while cpt < 1100:
		ret, image_np = cap.read()

		cv2.imshow('object detection', cv2.resize(image_np, cameraSize))
		if time() - t > 0.1:
			print('shoot', cpt)
			gray_image = cv2.cvtColor(cv2.resize(image_np, (imgSize,imgSize)), cv2.COLOR_BGR2GRAY)
			cv2.imwrite('./image/' + str(x) + '_' + str(cpt) +'.png', gray_image)
			t = time()
			cpt += 1

		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

ret, image_np = cap.read()
save_dir = 'image/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cv2.imshow('object detection', cv2.resize(image_np, cameraSize)
for x in range(0, nbClass):
	print('Lancement main :', x)
	for t in range(3,0,-1):
		print(t)
		sleep(1)
	main(x)