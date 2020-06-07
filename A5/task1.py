from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, AveragePooling2D
from keras.models import Model
from keras.losses import binary_crossentropy,mean_squared_error
from keras.optimizers import RMSprop,adam
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
import glob
import os
import tensorflow as tf
import csv
from PIL import Image
from contextlib import redirect_stdout
from keras.optimizers import RMSprop,Adam
import IntersectionOverUnion

def compute_losses(predicted_coods, batch_val_Y):
	IOU_per_img = 0.0
	j = 0
	#for i in range(4):
	IOU_per_img = IntersectionOverUnion.bb_intersection_over_union(predicted_coods, batch_val_Y)
	#j += 4

	return IOU_per_img

def log(row_entry):
	with open('losses_list.csv', 'a') as csvFile:
	    writer = csv.writer(csvFile)
	    writer.writerow(row_entry)

	csvFile.close()
    
def build_unified_data(array1,array2,array3):
	newarray = array1
	newarray = np.concatenate((newarray,array2),axis = 0)
	newarray = np.concatenate((newarray,array3),axis = 0)
	return newarray

def load_data_from_paths(path_mat_1):
	data_mat_1 = []
	for f in path_mat_1:
		img = Image.open(f)
		img = np.array(img)
		#img = img[2:-2, :, :]
		img_min = np.min(img)
		img_max = np.max(img)
		img = (img - img_min) / (img_max - img_min)
		data_mat_1.append(img) 

	data_mat_1 = np.asarray(data_mat_1)
	return data_mat_1

opt = Adam(lr=0.00001)

def perform_training(model,image_path_train,image_cls_train,image_reg_train,image_path_test,image_cls_test,image_reg_test):
	numEpochs = 1
	batch_size = 1
	losses_title = ['Epoch Number', 'batch_loss']
	log(losses_title)
	for jj in range(numEpochs):
		print("Running epoch : %d" % jj)
		batch_loss_file = open('Results/batch_loss_file.txt', 'a')
		batch_loss_per_epoch = 0.0
		num_batches = int(len(image_path_train)/batch_size)
		
		for batch in range(num_batches):
			batch_train_X_paths = image_path_train[batch*batch_size:min((batch+1)*batch_size,len(image_path_train))]
			batch_train_Y = image_cls_train[batch*batch_size:min((batch+1)*batch_size,len(image_cls_train))]
			batch_train_Z = image_reg_train[batch*batch_size:min((batch+1)*batch_size,len(image_reg_train))]
			batch_train_X = load_data_from_paths(batch_train_X_paths)
			batch_train_Z = batch_train_Z / np.array([352,459]*2)
			print('while training')
			loss = model.train_on_batch(batch_train_X, [batch_train_Y, batch_train_Z])
			print(loss)
			print ('epoch_num: %d batch_num: %d loss: %f \n' % (jj,batch,loss[0]))
			batch_loss_file.write("%d %d %f %f %f\n" % (jj, batch, loss[0],loss[1],loss[2]))
			batch_loss_per_epoch += loss[0]
		
		batch_loss_per_epoch = batch_loss_per_epoch / num_batches

		model.save_weights("Model_weights/model_epoch_"+ str(jj % 10) +".h5")

		#testing
		IOU_for_all_val_imgs = []
		num_batches_val = int(len(image_path_test)/batch_size)
		for batch in range(num_batches_val):
			batch_val_X_paths = image_path_test[batch*batch_size:min((batch+1)*batch_size,len(image_path_test))]
			batch_val_Y = image_cls_test[batch*batch_size:min((batch+1)*batch_size,len(image_cls_test))]
			batch_val_Z = image_reg_test[batch*batch_size:min((batch+1)*batch_size,len(image_reg_test))]
			test_X = load_data_from_paths(batch_val_X_paths)
			[y_pred,z_pred] = model.predict(test_X)
			predicted_coods_unnorm = z_pred * np.array([352,459]*2)
			predicted_coods_unnorm = np.rint(predicted_coods_unnorm[0])
			batch_val_Z = batch_val_Z[0]
			print(predicted_coods_unnorm.shape)
			print(batch_val_Z.shape)
			IOU_for_all_val_imgs.append(compute_losses(predicted_coods_unnorm, batch_val_Z))
			#print(y_pred, z_pred * np.array([352,459]*2))
			
		IOU_for_all_val_imgs = np.asarray(IOU_for_all_val_imgs)
		avg_IOU_all_val_imgs = np.mean(IOU_for_all_val_imgs)
		losses_list = [jj, batch_loss_per_epoch, IOU_for_all_val_imgs, avg_IOU_all_val_imgs]
		losses_list = [jj, batch_loss_per_epoch,IOU_for_all_val_imgs, avg_IOU_all_val_imgs]
		log(losses_list)
		

	batch_loss_file.close()


def make_mats(path_1,path_2):
	gt_mat = pd.read_csv(path_1,sep=",", header = None)
	gt_mat.columns = ["Image Path","x1","y1","x2","y2","class"]
	#print(gt_mat)
	gt_mat=gt_mat.sort_values('Image Path')
	num_of_files=gt_mat.shape[0]
	file_path_mat = []
	not_exist_file =[]
	for i in range(0,num_of_files):
		path = gt_mat.iloc[i]['Image Path']
		path_list = path.split('/',2)
		new_path = path_2 + '/' + path_list[1]
		#print(new_path)
		if os.path.exists(new_path):
			file_path_mat.append(new_path)
		else:
			not_exist_file.append(i)
	#print('file path',file_path_mat)
	valid_num_of_files = num_of_files-len(not_exist_file)
	#print('valid num of files',valid_num_of_files) #checks for shitty data sets
	y_mat = np.empty([valid_num_of_files,3])
	z_mat = np.empty([valid_num_of_files,4])
	j=0
	for i in range(0,num_of_files):
		if i not in not_exist_file:
			if(gt_mat.iloc[1]['class']=='knuckle'):
				y_mat[j][0] = int(1)
				y_mat[j][1] = int(0)
				y_mat[j][2] = int(0)
			elif (gt_mat.iloc[1]['class']=='Palm'): 
				y_mat[j][0] = int(0)
				y_mat[j][1] = int(1)
				y_mat[j][2] = int(0)
			elif (gt_mat.iloc[1]['class']=='veins'): 
				y_mat[j][0] = int(0)
				y_mat[j][1] = int(0)
				y_mat[j][2] = int(1)
			else:
				print('ERROR')

			z_mat[j][0] = int(gt_mat.iloc[i]['x1'])
			z_mat[j][1] = int(gt_mat.iloc[i]['y1'])
			z_mat[j][2] = int(gt_mat.iloc[i]['x2'])
			z_mat[j][3] = int(gt_mat.iloc[i]['y2'])
			j+=1
   

	return(file_path_mat,y_mat,z_mat)


def makeModel():
	conv_base = VGG16(include_top = False,
					  weights = 'imagenet',
					  input_shape=[352,459,3])
	features = conv_base.output
	layer = Conv2D(512, (3, 3), activation='relu', padding='same')(features)
	layer = MaxPooling2D(pool_size=(2, 2))(layer)
	layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
	layer = MaxPooling2D(pool_size=(2, 2))(layer)

	with tf.variable_scope('classification_head', reuse=tf.AUTO_REUSE):
		layer1 = Flatten()(layer)
		layer1 = Dense(2096,activation='relu')(layer1)
		layer1 = Dropout(0.5)(layer1)
		layer1 = Dense(2096,activation='relu')(layer1)
		layer1 = Dropout(0.5)(layer1)
		layer1 = Dense(3,activation ='softmax',name='layer1')(layer1)

	with tf.variable_scope('regression_head',reuse=tf.AUTO_REUSE):
		layer2 = Flatten()(layer)
		#layer2 = Dense(4096,activation='relu')(layer2)
		#layer2 = Dropout(0.5)(layer2)
		#layer2 = Dense(4096,activation='relu')(layer2)
		#layer2 = Dropout(0.5)(layer2)
		layer2 = Dense(4,name='layer2')(layer2)

	op = [layer1,layer2]
	model = Model(inputs=conv_base.input, outputs=op)
	i=0
	for l in model.layers:
	    l.trainable = False
	    i+=1
	    if i==19:
	    	break
	losses ={"layer1" : "categorical_crossentropy","layer2" :"mean_absolute_error"}
	loss_weight = {"layer1": 1.0, "layer2": 1.0}
	model.compile(loss=losses, optimizer=opt)
	with open('model_summary_task_1.txt', 'w') as f:
		with redirect_stdout(f):
			model.summary()
	return (model)



def main():
	knuckle_data_path = 'path/to/knuckle/data'
	palms_data_path = 'path/to/palm/data'
	veins_data_path = 'path/to/vein/data'
	knuckle_gt_path = 'path/to/knuckle/data/groundtruth.txt'
	palms_gt_path = 'path/to/palm/data/groundtruth.txt'
	veins_gt_path = 'path/to/vein/data/groundtruth.txt'
	model = makeModel()
	
	#class = {100,010,001} for {knuckle,palm,veins} respectively
	knuckle_image_path_mat,knuckle_cls_mat,knuckle_reg_mat = make_mats(knuckle_gt_path,knuckle_data_path)
	palms_image_path_mat,palms_cls_mat,palms_reg_mat = make_mats(palms_gt_path,palms_data_path)
	veins_image_path_mat,veins_cls_mat,veins_reg_mat = make_mats(veins_gt_path,veins_data_path)

	#unified data
	Data_path_mat = build_unified_data(knuckle_image_path_mat,palms_image_path_mat,veins_image_path_mat)
	cls_labels = build_unified_data(knuckle_cls_mat,palms_cls_mat,veins_cls_mat)
	reg_labels = build_unified_data(knuckle_reg_mat,palms_reg_mat,veins_reg_mat)

	#train test split
	image_path_train,image_path_test,image_cls_train,image_cls_test,image_reg_train,image_reg_test = train_test_split(Data_path_mat,cls_labels,reg_labels)
	
	#perform training
	perform_training(model,image_path_train,image_cls_train,image_reg_train,image_path_test,image_cls_test,image_reg_test)
	

main()




