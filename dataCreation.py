
""" Script to save examples of gestures in a folder to train a model after.
	This script open a thread on a webcam to take picture of gestures made by a person.
	All this pictures are resize to be store and use after in the training part. """

import cv2
import numpy as np
import os
from time import time, sleep
import glob

import sys

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
def captureGesture(x):
	global maxValue

	t = time() + 1
	cpt = maxValue
	pauseState = True
	print('Pause :', pauseState, 'Press SPACE to start')

	while cpt <= maxValue + int(sys.argv[1]):
		image_np = cap.get_frame()

		cv2.imshow('object detection', cv2.resize(image_np, cameraSize))
		if time() - t > 0.1 and not(pauseState):
			print('New picture', cpt)
			gray_image = cv2.resize(image_np, (imgSize,imgSize))
			#cv2.imwrite('./image/' + str(x) + '_' + str(cpt) +'.png', gray_image)
			t = time()
			cpt += 1

		key = cv2.waitKey(25) & 0xFF 
		if key == ord(' '):
			pauseState = not(pauseState)
			print('Pause :', pauseState, 'Press SPACE to change state')	
		elif key == ord('q'):
			cv2.destroyAllWindows()
			break

#Create a save folder if it doesn't exist
save_dir = 'image/'
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
for x in range(nbClass):
	print('Run capture :', gestures[x])
	captureGesture(x)
