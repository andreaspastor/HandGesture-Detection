import cv2
import numpy as np
import os
from time import time, sleep
import glob

import sys

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

imgSize = 256
cameraSize = (800, 600)
nbClass = 15

# 0 => rien
# 1 => Poing fermé
# 2 => pouce haut
# 3 => pouce bas
# 4 => main ouverte doigts serrés
# 5 => main ouverte doigts ecartés
# 6 => pouce + auriculaire horizontal
# 7 => pouce + auriculaire vertical
# 8 => victoire
# 9 => C
# 10 => ok
# 11 => victoire serré
# 12 => victoire serré horizontal
# 13 => rock and roll
# 14 => rock and roll horizontal

gestures = ['None', 'fist', 'thumb up', 'thumb down', \
            'stop', 'catch', 'swing', 'phone', 'victory', \
            'C', 'okay', '2 fingers', '2 fingers horiz', \
            'rock&roll', 'rock&roll horiz']

def main(x):
	t = time() + 1
	global maxValue
	cpt = maxValue
	pauseState = True
	print('Pause :', pauseState, 'Press SPACE to start')

	while cpt <= maxValue + int(sys.argv[1]):
		image_np = cap.get_frame()

		cv2.imshow('object detection', cv2.resize(image_np, cameraSize))
		if time() - t > 0.1 and not(pauseState):
			print('shoot', cpt)
			gray_image = cv2.resize(image_np, (imgSize,imgSize))
			cv2.imwrite('./image/' + str(x) + '_' + str(cpt) +'.png', gray_image)
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


""" Récupération du dernier index d'image dans le dossier des images """
liste = glob.glob(save_dir + '*.png')
maxValue = -1
for elm in liste:
	value = int(elm.split('_')[1].split('.')[0])
	if value > maxValue:
		maxValue = value

for x in range(nbClass):
	print('Lancement main :', gestures[x])
	main(x)
