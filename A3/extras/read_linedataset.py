#reading in the line dataset

import numpy as np
import pandas as pd
import cv2
import glob
import os

#x for stacking images, y for one hot encoded true classes,y_int for classes in integer form
x=np.empty([1000,28,28,3])
y=np.empty([1000,96])
g=np.empty([1000,96])
y_int=np.empty([1000])
g_int=np.empty([1000])

first = 1
#check os.walk documentation
for root, dirs, files in os.walk("pathname to target folder"):
  #root has names of each subfolder,chdir is changing to those directories
	os.chdir(root)
  #read all images in this subfolder
	images = np.array([cv2.imread(file) for file in glob.glob(root+"/*.jpg")])
	num_of_images=images.shape[0]
  
  #safety precaution incase folder is empty
	if num_of_images == 0:
		continue
	else:
    #if first image, variables are created instead of concatenated
		if first == 1:
			x = images
      
      #name has the label of the image extracted from the last part of the folder's pathname
			name = os.path.basename(os.path.normpath(root))
			y.fill(0)
			num=0
      
      #one hot encode in y i.e. 1 in only the class' position
			for i in range(0,999):
				y[i,num]=1
        
      #filling y_int with integer value of true class
			y_int.fill(num)
      
      #store corresponding label names of form 0_1_0_0
			d={'strname':name,'intname':num}
			corr = pd.DataFrame(data = d,index=[num])
			first = 0
      
      #update next label
			num=num+1
      
		else:
      #concatenate new image to old
			x = np.concatenate((x,images),axis=0)
			name = os.path.basename(os.path.normpath(root))
      #g will be concatenated to y,similarly g_int to y_int
			g.fill(0)
			for i in range(0,999):
				g[i,num]=1
			y=np.concatenate((y,g),axis=0)
			g_int.fill(num)
			y_int=np.concatenate((y_int,g_int),axis=0)
			d={'strname' : name,'intname':num}
			newcorr = pd.DataFrame(data = d,index=[num])
			corr = corr.append(newcorr)
			num=num+1
#reshaping x, dont remeber why, probably to make it suitable for later parts. 96000 total images,and stacked 28x28x3=2352,each.
x=np.reshape(x,[96000,2352])

#splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test,y_int_train,y_int_test = train_test_split(x, y, y_int, test_size=0.25, random_state=42)
