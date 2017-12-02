import glob
import bs4 as bs
import cv2
import pickle

liste = glob.glob('final_model16/feature_points/*.xml')
images = glob.glob('final_model16/original_images/*.jpg')

type = ['WristThumb', 'WristPinky', 'Knuckle1', 'Knuckle2', 'FingerTip']
data = []
for x, elm in enumerate(liste):
	with open(elm, "r" ) as f:
		html_doc = f.read()
	soup = bs.BeautifulSoup(html_doc, 'html5lib')

	df = len(data)
	for p in soup.find_all('hand'):
		X, Y = [], []
		for child in p.findChildren():
			X.append(int(child.get('x')))
			Y.append(int(child.get('y')))
		if len(X) > 0:
			xa, xb = max(X), min(X)
			ya, yb = max(Y), min(Y)
			data.append([xa,ya,xb,yb])
	if len(data) - 1 > df:
		print('Noonnnnn')
	image = cv2.imread(images[x])
	cv2.line(image, (xa,ya), (xa,yb), (255,0,0), 2)
	cv2.line(image, (xa,yb), (xb,yb), (255,0,0), 2)
	cv2.line(image, (xb,ya), (xb,yb), (255,0,0), 2)
	cv2.line(image, (xa,ya), (xb,ya), (255,0,0), 2)
	cv2.imshow('images',image)
	key = cv2.waitKey(2) & 0x0F


#pickle.dump(data)

cv2.destroyAllWindows()