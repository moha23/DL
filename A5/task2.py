import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Model
from keras.losses import binary_crossentropy,mean_squared_error
from keras.optimizers import RMSprop,adam

def make_mat_of_paths(Data_path):
	Data_folder = sorted(os.listdir(Data_path))

	Data_path_mat = []
	for f in Data_folder:
		Data_path_mat.append(Data_path + '/' + f)

	return Data_path_mat

def make_mat_of_groundtruth(Data_path):
	#gt_mat = np.genfromtxt(Data_path, delimiter=',', dtype=None, names=('image path', 'x1', 'y1', 'x2', 'y2', 'class'))
	Data_folder = sorted(os.listdir(Data_path))
	first = 1
	for f in Data_folder:
		if first == 1:
			df = pd.read_csv(f,sep=",", header = None)
			s = pd.concat((df.loc[i] for i in df.index), ignore_index=True)
			gt_mat = pd.DataFrame([s])
			gt_mat.columns = ["x1_1","y1_1","x2_1","y2_1","x1_2","y1_2","x2_2","y2_2","x1_3","y1_3","x2_3","y2_3","x1_4","y1_4","x2_4","y2_4"]
			first = 0
		else:
			df = pd.read_csv(f,sep=",", header = None)
			s = pd.concat((df.loc[i] for i in df.index), ignore_index=True)
			temp = pd.DataFrame([s])
			temp.columns = ["x1_1","y1_1","x2_1","y2_1","x1_2","y1_2","x2_2","y2_2","x1_3","y1_3","x2_3","y2_3","x1_4","y1_4","x2_4","y2_4"]
			gt_mat = pd.concat(gt_mat,temp)
	return(gt_mat)

def makeModel(inputx):
	
	conv_base = VGG16(include_top = False,
					  weights = 'imagenet',
					  input_tensor=inputx,
					  input_shape=[352,459,3])
	conv_base.trainable = False
	for l in conv_base.layers:
	    l.trainable = False
	features = conv_base.output

	with tf.variable_scope('regression_head',reuse=tf.AUTO_REUSE):
		layer2 = Flatten()(features)
		layer2 = Dense(4096,activation='relu')(layer2)
		layer2 = Dropout(0.5)(layer2)
		layer2 = Dense(4096,activation='relu')(layer2)
		layer2 = Dropout(0.5)(layer2)
		layer2 = Dense(16,name='layer2')(layer2)

	model = Model(inputs=conv_base.input, outputs=layer2)
	i=0
	for l in model.layers:
	    l.trainable = False
	    i+=1
	    if i==19:
	    	break
	
	model.compile(loss='mean_squared_error', optimizer='adam')
	#model.summary()
	return (model)

def main():
	data_path = '/Users/momo/Downloads/Four_Slap_Fingerprint/Image'
	groundtruth_path = '/Users/momo/Downloads/Four_Slap_Fingerprint/Ground_truth'

	inputx = tf.placeholder(tf.float32, shape=[None, 352,459,3], name='inputx')
	model = makeModel(inputx)
	data_path_mat = make_mat_of_paths(knuckle_data_path)
	groundtruth_path_mat = make_mat_of_groundtruth(knuckle_gt_path)
	

	


main()
