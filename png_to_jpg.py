import cv2
import os

def png_to_jpg(pathname):
	for root, dirs, files in os.walk(pathname):
		for name in files:
			if ".png" in name:
				img = cv2.imread(os.path.join(root, name))
				#print(root, name)
				cv2.imwrite("JPEG/" + root[4:] + "/" + name[:-3] + 'jpg', img)
				print("Converted: " + name)
