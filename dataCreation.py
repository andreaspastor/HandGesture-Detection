import cv2
import numpy as np
import os
from time import time, sleep


import sys
import select

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline()
            return True
    return False

#info sur le geste à faire
#recup auto du numero de l'image à ajouter 0_{nb}
#save in color or in gray

cap = cv2.VideoCapture(0)
imgSize = 256
cameraSize = (800, 600)
nbClass = 1

# 0 => rien
# 1 => Poing fermé
# 2 => Point ouvert

def main(x):
	t = time() + 1
	cpt = int(sys.argv[1])
	while cpt < int(sys.argv[1]) + int(sys.argv[2]):
		ret, image_np = cap.read()

		cv2.imshow('object detection', cv2.resize(image_np, cameraSize))
		if time() - t > 0.5:
			print('shoot', cpt)
			gray_image = cv2.resize(image_np, (imgSize,imgSize))
			cv2.imwrite('./image/0_falseMarcel/' + str(x) + '_' + str(cpt) +'.png', gray_image)
			t = time()
			cpt += 1

		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

ret, image_np = cap.read()
save_dir = 'image/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cv2.imshow('object detection', cv2.resize(image_np, cameraSize))
for x in range(0, nbClass):
	print('Lancement main :', x)
	main(x)
