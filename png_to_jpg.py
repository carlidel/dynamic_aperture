import cv2
import os

def png_to_jpg(pathname):
	names = os.listdir(pathname)
	if not os.path.exists(pathname + "JPEG"):
		os.makedirs(pathname + "JPEG")
	for image in names:
		if ".png" in image:
			img = cv2.imread(pathname + image)
			cv2.imwrite(pathname + "JPEG/" + image[:-3] + 'jpg', img)
			print("Converted: " + image)
