import cv2
import glob
import os

knuckles_images_foldername =  '/Users/momo/Downloads/Assignment3/Q1/Knuckle/Data'
palms_images_foldername = '/Users/momo/Downloads/Assignment3/Q1/Palm/Data'
veins_images_foldername = '/Users/momo/Downloads/Assignment3/Q1/Vein/Data'

for i in range(1,4):
	if i == 1:
		src_dir = knuckles_images_foldername
	elif i == 2:
		src_dir = palms_images_foldername
	elif i == 3:
		src_dir = veins_images_foldername

	for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
		imgname = os.path.basename(os.path.normpath(jpgfile))
		imgpath = os.path.join(src_dir,imgname)
		img = cv2.imread(imgpath)
		newjpgfile = cv2.resize(img,(459,352))
		if i == 1:
			path = '/Users/momo/Downloads/Assignment3/Q1/Knuckle/resized'
		elif i == 2:
			path = '/Users/momo/Downloads/Assignment3/Q1/Palm/resized'
		elif i == 3:
			path = '/Users/momo/Downloads/Assignment3/Q1/Vein/resized'

		cv2.imwrite(os.path.join(path , imgname), newjpgfile)

