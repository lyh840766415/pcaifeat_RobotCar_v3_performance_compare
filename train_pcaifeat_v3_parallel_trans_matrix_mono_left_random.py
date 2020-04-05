import tensorflow as tf
import numpy as np
from loading_input_v3 import *
from pointnetvlad_v3.pointnetvlad_trans import *
import pointnetvlad_v3.loupe as lp
import nets_v3.resnet_v1_trans as resnet
import shutil
from multiprocessing.dummy import Pool as ThreadPool
import threading
import time
import cv2
sys.path.append('/data/lyh/lab/robotcar-dataset-sdk/python')
from camera_model import CameraModel
from transform import build_se3_transform
import numpy as np
from image import load_image
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

#thread pool
pool = ThreadPool(10)

# is rand init 
RAND_INIT = False
# model path
MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3_1/log/train_save_trans_fusion/model_00180060.ckpt"
PC_MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3_1/log/train_save_trans_pc/pc_model_00174058.ckpt"
IMG_MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3_1/log/train_save_trans_fusion_1/img_model_00348116.ckpt"
# log path
LOG_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3_1/log/train_save_trans_random"
# 1 for point cloud only, 2 for image only, 3 for pc&img&fc
TRAINING_MODE = 3
#TRAIN_ALL = True
ONLY_TRAIN_FUSION = False

#Loss
quadruplet = True


# Epoch & Batch size &FINAL EMBBED SIZE & learning rate
EPOCH = 20
LOAD_BATCH_SIZE = 100
FEAT_BARCH_SIZE = 1
LOAD_FEAT_RATIO = LOAD_BATCH_SIZE//FEAT_BARCH_SIZE
EMBBED_SIZE = 1000
BASE_LEARNING_RATE = 5e-5

#pos num,neg num,other neg num,all_num
POS_NUM = 2
NEG_NUM = 5
OTH_NUM = 1
BATCH_DATA_SIZE = 1 + POS_NUM + NEG_NUM + OTH_NUM

# Hard example mining start
HARD_MINING_START = 5

# Margin
MARGIN1 = 0.5
MARGIN2 = 0.2

#Train file index & pc img matching
TRAIN_FILE = 'generate_queries_v3/training_queries_RobotCar_trans_matrix.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)

#cur_load for get_batch_keys
CUR_LOAD = 0

#multi threading share global variable
TRAINING_DATA = []
TRAINING_DATA_LOCK = threading.Lock()
#for each load batch
BATCH_REACH_END = False
cnt = 0
LOAD_QUENE_SIZE = 10
EP = 0

#camera model and posture
CAMERA_MODEL = None
G_CAMERA_POSESOURCE = None



def init_camera_model_posture():
	global CAMERA_MODEL
	global G_CAMERA_POSESOURCE
	models_dir = "/data/lyh/lab/robotcar-dataset-sdk/models/"
	CAMERA_MODEL = CameraModel(models_dir, "mono_left")
	#read the camera and ins extrinsics
	extrinsics_path = "/data/lyh/lab/robotcar-dataset-sdk/extrinsics/mono_left.txt"
	print(extrinsics_path)
	with open(extrinsics_path) as extrinsics_file:
		extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
	G_camera_vehicle = build_se3_transform(extrinsics)
	print(G_camera_vehicle)
	
	extrinsics_path = "/data/lyh/lab/robotcar-dataset-sdk/extrinsics/ins.txt"
	print(extrinsics_path)
	with open(extrinsics_path) as extrinsics_file:
		extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
	G_ins_vehicle = build_se3_transform(extrinsics)
	print(G_ins_vehicle)
	G_CAMERA_POSESOURCE = G_camera_vehicle*G_ins_vehicle
	
	
def get_learning_rate(epoch):
	learning_rate = BASE_LEARNING_RATE*((0.9)**(epoch//5))
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate

def get_bn_decay(step):
	#batch norm parameter
	DECAY_STEP = 200000
	BN_INIT_DECAY = 0.5
	BN_DECAY_DECAY_RATE = 0.5
	BN_DECAY_DECAY_STEP = float(DECAY_STEP)
	BN_DECAY_CLIP = 0.99
	bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,step*FEAT_BARCH_SIZE,BN_DECAY_DECAY_STEP,BN_DECAY_DECAY_RATE,staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay


def init_imgnetwork(pc_trans_feat):
	with tf.variable_scope("img_var"):
		img_placeholder = tf.placeholder(tf.float32,shape=[FEAT_BARCH_SIZE*BATCH_DATA_SIZE,256,256,3])
		img_feat, img_pc_feat = resnet.endpoints(img_placeholder,pc_trans_feat,is_training=True)
	return img_placeholder, img_feat, img_pc_feat


'''
def init_imgnetwork(pc_trans_feat):	
	with tf.variable_scope("img_var"):
		img_placeholder = tf.placeholder(tf.float32,shape=[FEAT_BARCH_SIZE*BATCH_DATA_SIZE,256,256,3])
		_,img_feat,_= resnet.endpoints(img_placeholder,is_training=True)
		img_feat = tf.layers.dense(img_feat,1000,activation=tf.nn.relu)
		img_feat = tf.nn.l2_normalize(img_feat,1)
	return img_placeholder, img_feat, None
'''
		
def init_pcnetwork(step):
	with tf.variable_scope("pc_var"):
		pc_placeholder = tf.placeholder(tf.float32,shape=[FEAT_BARCH_SIZE*BATCH_DATA_SIZE,4096,3])
		trans_mat_placeholder = tf.placeholder(tf.float32,shape=[FEAT_BARCH_SIZE*BATCH_DATA_SIZE,64,4096])
		is_training_pl = tf.placeholder(tf.bool, shape=())
		bn_decay = get_bn_decay(step)
		pc_feat,pc_trans_feat = pointnetvlad(pc_placeholder,trans_mat_placeholder,is_training_pl,bn_decay)	
	return pc_placeholder,is_training_pl,trans_mat_placeholder,pc_feat,pc_trans_feat
	
def init_fusion_network(pc_feat,img_feat):
	with tf.variable_scope("fusion_var"):
		concat_feat = tf.concat((pc_feat,img_feat),axis=1)
		pcai_feat = tf.layers.dense(concat_feat,EMBBED_SIZE,activation=tf.nn.relu)
	return pcai_feat

def init_pcainetwork():
	#training step
	step = tf.Variable(0)
	pc_trans_feat = None	
	#init sub-network
	if TRAINING_MODE != 2:
		pc_placeholder, is_training_pl, trans_mat_placeholder, pc_feat,pc_trans_feat = init_pcnetwork(step)
	if TRAINING_MODE != 1:
		img_placeholder, img_feat, img_pc_feat = init_imgnetwork(pc_trans_feat)
		img_feat = img_pc_feat
	if TRAINING_MODE == 3:
		pcai_feat = init_fusion_network(pc_feat,img_pc_feat)
	
	'''
	variables = tf.trainable_variables()
	pc_variable = [v for v in variables if v.name.split('/')[0] =='pc_var']
	img_variable = [v for v in variables if v.name.split('/')[0] =='img_var']
	fusion_variable = [v for v in variables if v.name.split('/')[0] =='fusion_var']
	
	for var in pc_variable:
		print(var)
	for var in img_variable:
		print(var)
	for var in fusion_variable:
		print(var)
	exit()
	
	pc_variable_cnt = 0
	for variable in pc_variable:
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
			pc_variable_cnt += variable_parameters
	print("pc_parameter size = ", pc_variable_cnt)
	
	img_variable_cnt = 0
	for variable in img_variable:
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
			img_variable_cnt += variable_parameters
	print("img_parameter size = ", img_variable_cnt)
	
	fusion_variable_cnt = 0
	for variable in fusion_variable:
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
			fusion_variable_cnt += variable_parameters
	print("fusion_parameter size = ", fusion_variable_cnt)
		
	exit()
	'''
	
	#prepare data and loss
	if TRAINING_MODE != 2:
		pc_feat = tf.reshape(pc_feat,[FEAT_BARCH_SIZE,BATCH_DATA_SIZE,pc_feat.shape[1]])
		q_pc_vec, pos_pc_vec, neg_pc_vec, oth_pc_vec = tf.split(pc_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
		pc_loss = lazy_quadruplet_loss(q_pc_vec, pos_pc_vec, neg_pc_vec, oth_pc_vec, MARGIN1, MARGIN2)
		tf.summary.scalar('pc_loss', pc_loss)

		
	if TRAINING_MODE != 1:
		img_feat = tf.reshape(img_feat,[FEAT_BARCH_SIZE,BATCH_DATA_SIZE,img_feat.shape[1]])
		q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec = tf.split(img_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
		img_loss = lazy_quadruplet_loss(q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec, MARGIN1, MARGIN2)
		tf.summary.scalar('img_loss', img_loss)
	
	
	if TRAINING_MODE == 3:
		pcai_feat = tf.reshape(pcai_feat,[FEAT_BARCH_SIZE,BATCH_DATA_SIZE,pcai_feat.shape[1]])
		q_vec, pos_vec, neg_vec, oth_vec = tf.split(pcai_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
		finally_loss = lazy_quadruplet_loss(q_vec, pos_vec, neg_vec, oth_vec, MARGIN1, MARGIN2)
		all_loss = finally_loss + 2*pc_loss + img_loss
		tf.summary.scalar('finally_loss', finally_loss)
		tf.summary.scalar('all_loss', all_loss)
		
	#learning rate strategy, all in one?
	epoch_num_placeholder = tf.placeholder(tf.float32, shape=())
	learning_rate = get_learning_rate(epoch_num_placeholder)
	tf.summary.scalar('learning_rate', learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	
	#variable update
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	#only the fusion_variable
	#TODO
	fusion_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	if ONLY_TRAIN_FUSION and TRAINING_MODE == 3:
		with tf.control_dependencies(fusion_ops):
			fusion_train_op = optimizer.minimize(all_loss, global_step=step)
	
	#training operation
	with tf.control_dependencies(update_ops):
		if TRAINING_MODE != 2:
			pc_train_op = optimizer.minimize(pc_loss, global_step=step)
		if TRAINING_MODE != 1:
			img_train_op = optimizer.minimize(img_loss, global_step=step)
		if TRAINING_MODE == 3:
			all_train_op = optimizer.minimize(all_loss, global_step=step)
	
	
	
	#merged all log variable
	merged = tf.summary.merge_all()
	
	#output of pcainetwork init
	if TRAINING_MODE == 1:
		ops = {
			"is_training_pl":is_training_pl,
			"pc_placeholder":pc_placeholder,
			"epoch_num_placeholder":epoch_num_placeholder,
			"pc_loss":pc_loss,
			"pc_train_op":pc_train_op,
			"merged":merged,
			"step":step}
		return ops
		
	if TRAINING_MODE == 2:
		ops = {
			"img_placeholder":img_placeholder,
			"epoch_num_placeholder":epoch_num_placeholder,
			"img_loss":img_loss,
			"img_train_op":img_train_op,
			"merged":merged,
			"step":step}
		return ops
		
	if TRAINING_MODE == 3 and ONLY_TRAIN_FUSION:
		ops = {
			"is_training_pl":is_training_pl,
			"pc_placeholder":pc_placeholder,
			"img_placeholder":img_placeholder,
			"trans_mat_placeholder":trans_mat_placeholder,
			"epoch_num_placeholder":epoch_num_placeholder,
			"pc_loss":pc_loss,
			"img_loss":img_loss,
			"all_loss":all_loss,
			"pc_train_op":pc_train_op,
			"img_train_op":img_train_op,
			"all_train_op":all_train_op,
			"fusion_train_op":fusion_train_op,
			"merged":merged,
			"step":step}
		return ops
		
	if TRAINING_MODE == 3:
		ops = {
			"is_training_pl":is_training_pl,
			"pc_placeholder":pc_placeholder,
			"img_placeholder":img_placeholder,
			"trans_mat_placeholder":trans_mat_placeholder,
			"epoch_num_placeholder":epoch_num_placeholder,
			"pc_loss":pc_loss,
			"img_loss":img_loss,
			"all_loss":all_loss,
			"pc_train_op":pc_train_op,
			"img_train_op":img_train_op,
			"all_train_op":all_train_op,
			"merged":merged,
			"step":step}
		return ops
		

def init_network_variable(sess,train_saver):
	sess.run(tf.global_variables_initializer())
	print("random init")
	
	if RAND_INIT:
		return
		
	if TRAINING_MODE == 1:
		train_saver['pc_saver'].restore(sess,PC_MODEL_PATH)
		print("pc_model restored")
		return
		
	if TRAINING_MODE == 2:
		train_saver['img_saver'].restore(sess,IMG_MODEL_PATH)
		print("img_model restored")
		return
	
	if TRAINING_MODE == 3 and ONLY_TRAIN_FUSION:
		train_saver['pc_saver'].restore(sess,PC_MODEL_PATH)
		print("pc_model restored")
		train_saver['img_saver'].restore(sess,IMG_MODEL_PATH)
		print("img_model restored")
		return
	
	if TRAINING_MODE == 3:
		#train_saver['all_saver'].restore(sess,MODEL_PATH)
		#print("all_model restored")
		#train_saver['pc_saver'].restore(sess,PC_MODEL_PATH)
		#print("pc_model restored")
		train_saver['img_saver'].restore(sess,IMG_MODEL_PATH)
		print("img_model restored")
		
		return

def init_train_saver():
	all_saver = tf.train.Saver()
	variables = tf.contrib.framework.get_variables_to_restore()
	pc_variable = [v for v in variables if v.name.split('/')[0] =='pc_var']
	img_variable = [v for v in variables if v.name.split('/')[0] =='img_var' and v.name.split('/')[1] == 'resnet_v1_50']
	
	pc_saver = None
	img_saver = None
	if TRAINING_MODE != 2:
		pc_saver = tf.train.Saver(pc_variable)
	if TRAINING_MODE != 1:
		img_saver = tf.train.Saver(img_variable)
		
	
	train_saver = {
		'all_saver':all_saver,
		'pc_saver':pc_saver,
		'img_saver':img_saver}
	
	return train_saver
	
def prepare_batch_data(pc_data,img_data,trans_data,feat_batch,ops,ep):
	is_training = True
	if TRAINING_MODE != 2:
		feat_batch_pc = pc_data[feat_batch*BATCH_DATA_SIZE*FEAT_BARCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BARCH_SIZE]
	if TRAINING_MODE != 1:
		feat_batch_img = img_data[feat_batch*BATCH_DATA_SIZE*FEAT_BARCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BARCH_SIZE]
	if TRAINING_MODE == 3:
		feat_batch_trans = trans_data[feat_batch*BATCH_DATA_SIZE*FEAT_BARCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BARCH_SIZE]
		
		
	if TRAINING_MODE == 1:
		train_feed_dict = {
		  ops["is_training_pl"]:is_training,
			ops["pc_placeholder"]:feat_batch_pc,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	if TRAINING_MODE == 2:
		train_feed_dict = {
			ops["img_placeholder"]:feat_batch_img,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	if TRAINING_MODE == 3:
		train_feed_dict = {
			ops["is_training_pl"]:is_training,
			ops["img_placeholder"]:feat_batch_img,
			ops["pc_placeholder"]:feat_batch_pc,
			ops["trans_mat_placeholder"]:	feat_batch_trans,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	print("prepare_batch_data_error,no_such train mode.")
	exit()

def train_one_step(sess,ops,train_feed_dict,train_writer):
	if TRAINING_MODE == 1:
		summary,step,pc_loss,_,= sess.run([ops["merged"],ops["step"],ops["pc_loss"],ops["pc_train_op"]],feed_dict = train_feed_dict)
		print("batch num = %d , pc_loss = %f"%(step, pc_loss))

	if TRAINING_MODE == 2:
		summary,step,img_loss,_,= sess.run([ops["merged"],ops["step"],ops["img_loss"],ops["img_train_op"]],feed_dict = train_feed_dict)
		print("batch num = %d , img_loss = %f"%(step, img_loss))
	
	if TRAINING_MODE == 3:
		if ONLY_TRAIN_FUSION:
			summary,step,all_loss,_,= sess.run([ops["merged"],ops["step"],ops["all_loss"],ops["fusion_train_op"]],feed_dict = train_feed_dict)
			print("batch num = %d , all_loss = %f"%(step, all_loss))
		
		else:
			summary,step,all_loss,_,= sess.run([ops["merged"],ops["step"],ops["all_loss"],ops["all_train_op"]],feed_dict = train_feed_dict)
			print("batch num = %d , all_loss = %f"%(step, all_loss))
			
	#other training strategy
	train_writer.add_summary(summary, step)
	return step
	
def evaluate():
	return
	
def model_save(sess,step,train_saver):
	if TRAINING_MODE == 1:
		save_path = train_saver['pc_saver'].save(sess,os.path.join(LOG_PATH, "pc_model_%08d.ckpt"%(step)))
		print("PC Model saved in file: %s" % save_path)
		return
		
	if TRAINING_MODE == 2:
		save_path = train_saver['img_saver'].save(sess,os.path.join(LOG_PATH, "img_model_%08d.ckpt"%(step)))
		print("IMG Model saved in file: %s" % save_path)
		return
	
	if TRAINING_MODE == 3:
		save_path = train_saver['pc_saver'].save(sess,os.path.join(LOG_PATH, "pc_model_%08d.ckpt"%(step)))
		print("PC Model saved in file: %s" % save_path)
		save_path = train_saver['img_saver'].save(sess,os.path.join(LOG_PATH, "img_model_%08d.ckpt"%(step)))
		print("IMG Model saved in file: %s" % save_path)
		save_path = train_saver['all_saver'].save(sess,os.path.join(LOG_PATH, "model_%08d.ckpt"%(step)))
		print("Model saved in file: %s" % save_path)
		return

def is_negative(query,not_negative):
	return not query in not_negative
	
def get_load_batch_filename(load_batch_keys,quadruplet):		
	pc_files = []
	img_files = []
	for key_cnt ,key in enumerate(load_batch_keys):
		pc_files.append(TRAINING_QUERIES[key]["query"])
		img_files.append("%s_mono_left.png"%(TRAINING_QUERIES[key]["query"][:-4]))
		random.shuffle(TRAINING_QUERIES[key]["positives"])
		
		#print(TRAINING_QUERIES[key])
		cur_pos = 0;
		for i in range(POS_NUM):
			while True:
				filename = "%s_mono_left.png"%(TRAINING_QUERIES[TRAINING_QUERIES[key]["positives"][cur_pos]]["query"][:-4])

				if filename in img_files[BATCH_DATA_SIZE*(key_cnt)+1:BATCH_DATA_SIZE*(key_cnt)+1+i]:
					cur_pos = cur_pos+1
					continue
				if os.path.exists(filename):
					break
				cur_pos = cur_pos+1
				if cur_pos>len(TRAINING_QUERIES[key]["positives"]):
					print("line 259, error in positive number")
					exit()
			
			pc_files.append(TRAINING_QUERIES[TRAINING_QUERIES[key]["positives"][cur_pos]]["query"])
			img_files.append(filename)		
		
		neg_indices = []
		for i in range(NEG_NUM):
			while True:
				while True:
					neg_ind = random.randint(0,len(TRAINING_QUERIES.keys())-1)
					if is_negative(neg_ind,TRAINING_QUERIES[key]["not_negative"]):
						break
				

				filename = "%s_mono_left.png"%(TRAINING_QUERIES[neg_ind]["query"][:-4])
				if filename in img_files[BATCH_DATA_SIZE*(key_cnt)+1+POS_NUM:BATCH_DATA_SIZE*(key_cnt)+1+POS_NUM+i]:
					continue
				if os.path.exists(filename):
					break
					
			neg_indices.append(neg_ind)
			pc_files.append(TRAINING_QUERIES[neg_ind]["query"])
			img_files.append(filename)
		
		'''
		tmp_list = img_files[9*(key_cnt)+1+POS_NUM:9*(key_cnt)+1+POS_NUM+NEG_NUM]
		if len(tmp_list)!=len(set(tmp_list)):
			print("neg_duplicate")
			input()
		'''
		
		if quadruplet:
			neighbors=[]
			for pos in TRAINING_QUERIES[key]["positives"]:
				neighbors.append(pos)
			for neg in neg_indices:
				for pos in TRAINING_QUERIES[neg]["positives"]:
					neighbors.append(pos)
					
			#print("neighbors size = ",len(neighbors))
			while True:
				neg_ind = random.randint(0,len(TRAINING_QUERIES.keys())-1)
				if is_negative(neg_ind,neighbors):
					filename = "%s_mono_left.png"%(TRAINING_QUERIES[neg_ind]["query"][:-4])
					if os.path.exists(filename):
						break						
									
			pc_files.append(TRAINING_QUERIES[neg_ind]["query"])
			img_files.append(filename)
	
	if TRAINING_MODE == 1:
		return pc_files,None
	
	if TRAINING_MODE == 2:
		return None,img_files
		
	if TRAINING_MODE == 3:
		return pc_files,img_files
	

def get_batch_keys(train_file_idxs,train_file_num):
	global CUR_LOAD
	load_batch_keys = []
	
	while len(load_batch_keys) < LOAD_BATCH_SIZE:
		skip_num = 0
		#make sure cur_load is valid
		if CUR_LOAD >= train_file_num:
			return True,None
			
		cur_key = train_file_idxs[CUR_LOAD]
		if len(TRAINING_QUERIES[cur_key]["positives"]) < POS_NUM:
			CUR_LOAD = CUR_LOAD + 1
			skip_num = skip_num + 1
			continue
		
		filename = "%s_mono_left.png"%(TRAINING_QUERIES[cur_key]["query"][:-4])
		if not os.path.exists(filename):
			#print(TRAINING_QUERIES[cur_key]["query"])
			CUR_LOAD = CUR_LOAD + 1
			skip_num = skip_num + 1
			continue
		
		valid_pos = 0
		for i in range(len(TRAINING_QUERIES[cur_key]["positives"])):
			filename = "%s_mono_left.png"%(TRAINING_QUERIES[TRAINING_QUERIES[cur_key]["positives"][i]]["query"][:-4])
			if os.path.exists(filename):
					valid_pos = valid_pos + 1
				
		if valid_pos < POS_NUM:
			skip_num = skip_num + 1
			CUR_LOAD = CUR_LOAD + 1
			continue
						
		load_batch_keys.append(train_file_idxs[CUR_LOAD])
		CUR_LOAD = CUR_LOAD + 1
		
	return False,load_batch_keys

def cal_trans_data(pc_dict):
	posfile = pc_dict[0]
	pointcloud = pc_dict[1]	
	rand_ind = np.random.randint(64,size=4096)
	arange_ind = np.arange(4096)
	row = rand_ind*4096 + arange_ind
	
	transform_matrix = np.zeros([64*4096,1])
	transform_matrix[row] = 1
	transform_matrix = transform_matrix.reshape([64,4096])
	
	return transform_matrix
	
	
	
def get_trans_datas(load_pc_filenames,pc_data,pool):
	dict_list = []
	for i in range(pc_data.shape[0]):
		para = ("%s_imgpos.txt"%(load_pc_filenames[i][:-4]),pc_data[i,:,:])
		#print(para_dict)
		dict_list.append(para)
	
	'''
	for i in range(len(dict_list)):
		cal_trans_data(dict_list[i])
	return
	'''
	trans_data = pool.map(cal_trans_data,dict_list)
	return np.array(trans_data)

def load_data(train_file_idxs):
	global BATCH_REACH_END
	global TRAINING_DATA
	global cnt
	
	while True:
		TRAINING_DATA_LOCK.acquire()
		list_len = len(TRAINING_DATA)
		TRAINING_DATA_LOCK.release()
		if list_len > LOAD_QUENE_SIZE:
			print("reach maximum")
			time.sleep(1)
			continue
		
		BATCH_REACH_END,load_batch_keys = get_batch_keys(train_file_idxs,train_file_idxs.shape[0])
		if BATCH_REACH_END:
			print("load thread ended---------------------------------------------------------------------------------------------------")
			break
		
		
		#select load_batch tuple
		load_pc_filenames,load_img_filenames = get_load_batch_filename(load_batch_keys,quadruplet)
		
		#load pc&img data from file
		pc_data,img_data = load_img_pc(load_pc_filenames,load_img_filenames,pool)		
		
		trans_data = None
		if TRAINING_MODE != 2:
			trans_data = get_trans_datas(load_pc_filenames,pc_data,pool)
			for i in range(len(pc_data)):
				cur_pc = pc_data[i]
				mean_cur_pc = cur_pc.mean(axis = 0)
				cur_pc = cur_pc - mean_cur_pc
				mean_cur_pc = cur_pc.mean(axis = 0)
				cur_pc = cur_pc - mean_cur_pc
				pc_data[i] = cur_pc
		
		'''
		for i in range(len(pc_data)):
			cur_pc = pc_data[i]
			print(cur_pc.mean(axis = 0))
		exit()
		#np.savetxt("hehe.txt",pc_data[0],delimiter=",",fmt="%.3f")
		trans_data = np.array(trans_data)
		print(trans_data.shape)
		print(pc_data.shape)
		print(img_data.shape)
		
		aa = np.sum(trans_data[0],1).reshape([8,8])
		print(np.sum(aa))
		plt.figure(1)
		plt.imshow(aa.T)
		plt.show()
		'''
		'''
		for i in range(900):
			np.savetxt("pc_%d.txt"%(i),pc_data[i,:,:],delimiter=" ",fmt = "%.3f")
			cv2.imwrite("img_%d.png"%(i),np.squeeze(img_data[i,:,:,:]))
			
		exit()
		'''
		TRAINING_DATA_LOCK.acquire()
		TRAINING_DATA.append([pc_data,img_data,trans_data])
		TRAINING_DATA_LOCK.release()
		
		cnt = cnt + 1
		print("load one batch",cnt)
		
	return

def training(sess,train_saver,train_writer,ops):
	global BATCH_REACH_END
	
	first_loop = True
	consume_all = False
	while True:			
		TRAINING_DATA_LOCK.acquire()
		list_len = len(TRAINING_DATA)
		TRAINING_DATA_LOCK.release()
		#determine whether the first loop
		if not first_loop:
			#determine whether the consume all
			if not consume_all:
				if list_len <= 0:
					consume_all = True
					#end training
					if BATCH_REACH_END:
						print("training thread ended")
						break
					continue
			else:
				if BATCH_REACH_END:
					print("training thread ended")
					break
				if list_len < LOAD_QUENE_SIZE:
					print("list_len = %d, wait for list_len >= %d"%(list_len,LOAD_QUENE_SIZE))
					time.sleep(20)
					continue
				else:
					consume_all = False
					continue					
		else:
			if list_len <= 0:
				time.sleep(1)
				continue
			first_loop = False
		
		TRAINING_DATA_LOCK.acquire()
		cur_batch_data = TRAINING_DATA[0]
		del(TRAINING_DATA[0])
		TRAINING_DATA_LOCK.release()
		pc_data = cur_batch_data[0]
		img_data = cur_batch_data[1]
		trans_data = cur_batch_data[2]
		
		print("consume one batch")
	
		for feat_batch in range(LOAD_FEAT_RATIO):
			#prepare this batch data
			train_feed_dict = prepare_batch_data(pc_data,img_data,trans_data,feat_batch,ops,EP)
											
			#training
			step = train_one_step(sess,ops,train_feed_dict,train_writer)
						
			#evaluate TODO
			if step%201 == 0:
				evaluate()
						
			if step%3001 == 0:
				model_save(sess,step,train_saver)
	
	return
	
	
def main():
	global CUR_LOAD
	global BATCH_REACH_END
	global EP
	
	#init_camera_model_posture
	init_camera_model_posture()
	
	#init network pipeline
	ops = init_pcainetwork()
	
	#init train saver
	train_saver = init_train_saver()

	#init GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	
	
	#init tensorflow Session
	with tf.Session(config=config) as sess:
		#init all the variable
		init_network_variable(sess,train_saver)
		train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
		
		#init_training thread
		training_thread = threading.Thread(target=training, args=(sess,train_saver,train_writer,ops,))
		training_thread.start()

		#start training
		for ep in range(EPOCH):
			train_file_num = len(TRAINING_QUERIES.keys())
			train_file_idxs = np.arange(0,train_file_num)
			np.random.shuffle(train_file_idxs)
			print('Eppch = %d, train_file_num = %f , FEAT_BATCH_SIZE = %f , iteration per batch = %f' %(ep,len(train_file_idxs), FEAT_BARCH_SIZE,len(train_file_idxs)//FEAT_BARCH_SIZE))
			EP = ep
			BATCH_REACH_END = False
			CUR_LOAD = 0
			#load data thread
			load_data_thread = threading.Thread(target=load_data, args=(train_file_idxs,))
			load_data_thread.start()
				
			load_data_thread.join()
		
		training_thread.join()
						
					
if __name__ == "__main__":
	main()