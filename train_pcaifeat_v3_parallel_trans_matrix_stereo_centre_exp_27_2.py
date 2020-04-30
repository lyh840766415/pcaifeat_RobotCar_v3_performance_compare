#dataset baseline select
#with ground 32bit
#no scale normlization
#no rotation normlization

import tensorflow as tf
import numpy as np
from loading_input_v3 import *
from pointnetvlad_v3.pointnetvlad_fusion_non_local import *
import pointnetvlad_v3.loupe as lp
import nets_v3.resnet_v1_fusion_non_local as resnet
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
pool = ThreadPool(5)

# is rand init 
RAND_INIT = False
# model path
MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3_performance_compare/log/train_save_trans_exp_27_1/model_00882294.ckpt"
PC_MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3_performance_compare/log/train_save_trans_exp_6_4/pc_model_00441147.ckpt"
IMG_MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3_performance_compare/log/train_save_trans_exp_6_5/img_model_00441147.ckpt"
# log path
LOG_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3_performance_compare/log/train_save_trans_exp_27_2"
# 1 for point cloud only, 2 for image only, 3 for pc&img&fc
TRAINING_MODE = 3
#TRAIN_ALL = True
ONLY_TRAIN_FUSION = False

#Loss
quadruplet = True


# Epoch & Batch size &FINAL EMBBED SIZE & learning rate
EPOCH = 20
LOAD_BATCH_SIZE = 100
FEAT_BATCH_SIZE = 1
LOAD_FEAT_RATIO = LOAD_BATCH_SIZE//FEAT_BATCH_SIZE
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
TRAIN_FILE = 'generate_queries_v3/training_queries_RobotCar_trans_ground.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
TEST_FILE = 'generate_queries_v3/test_queries_RobotCar_trans_ground.pickle'
TEST_QUERIES = get_queries_dict(TEST_FILE)


#cur_load for get_batch_keys
CUR_LOAD = 0

#multi threading share global variable
TRAINING_DATA = []
TRAINING_DATA_LOCK = threading.Lock()
#for each load batch
BATCH_REACH_END = False
cnt = 0
LOAD_QUENE_SIZE = 4
EP = 0

#camera model and posture
CAMERA_MODEL = None
G_CAMERA_POSESOURCE = None



def init_camera_model_posture():
	global CAMERA_MODEL
	global G_CAMERA_POSESOURCE
	models_dir = "/data/lyh/lab/robotcar-dataset-sdk/models/"
	CAMERA_MODEL = CameraModel(models_dir, "stereo_centre")
	#read the camera and ins extrinsics
	extrinsics_path = "/data/lyh/lab/robotcar-dataset-sdk/extrinsics/stereo.txt"
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
	learning_rate = BASE_LEARNING_RATE*((0.9)**((epoch)//5))
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate

def get_bn_decay(step):
	step = tf.div(step,2)
		
	#batch norm parameter
	DECAY_STEP = 230000
	BN_INIT_DECAY = 0.5
	BN_DECAY_DECAY_RATE = 0.5
	BN_DECAY_DECAY_STEP = float(DECAY_STEP)
	BN_DECAY_CLIP = 0.99
	bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,step*FEAT_BATCH_SIZE,BN_DECAY_DECAY_STEP,BN_DECAY_DECAY_RATE,staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

#resnet_v1_50 need to be modified
#output: feature map before global pooling batch_size * 8 * 10 * 2048
def init_imgnetwork(is_training = True):
	with tf.variable_scope("img_var"):
		img_placeholder = tf.placeholder(tf.float32,shape=[FEAT_BATCH_SIZE*BATCH_DATA_SIZE,240,320,3])
		img_feat = resnet.endpoints(img_placeholder,is_training=is_training)
	return img_placeholder, img_feat

#pointnetvlad_cls need to be modified
#output: pointnet batch_size * 4096 * 1024
#modify:remove vlad layer		
def init_pcnetwork(step):
	with tf.variable_scope("pc_var"):
		pc_placeholder = tf.placeholder(tf.float32,shape=[FEAT_BATCH_SIZE*BATCH_DATA_SIZE,4096,3])
		is_training_pl = tf.placeholder(tf.bool, shape=())
		bn_decay = get_bn_decay(step)
		tf.summary.scalar('bn_decay', bn_decay)
		pc_feat = pointnetvlad(pc_placeholder,is_training_pl,bn_decay)	
	return pc_placeholder,is_training_pl,pc_feat
	
#input point featmap and img featmap
#b * 4096 * 1024, b * 8 * 10 * 2048
#output: non_local_feat: b * 4096 * 1024, b * 8 * 10 * 2048
def init_fusion_network(pc_feat,img_feat,is_training = True):
	print(pc_feat)
	print(img_feat)
	batch_size = pc_feat.get_shape()[0].value
	point_num = pc_feat.get_shape()[1].value
	img_h = img_feat.get_shape()[1].value
	img_w = img_feat.get_shape()[2].value
	bottle_neck = 512
	pc_feat_size = pc_feat.get_shape()[3].value
	img_feat_size = img_feat.get_shape()[3].value
	cluster_size=64
	
	with tf.variable_scope("fusion_var"):
		t_net = tf.layers.conv2d(pc_feat,bottle_neck,1,activation=None,name="non_local_t")
		t_net = tf.reshape(t_net,[batch_size,-1,bottle_neck])
		p_net = tf.layers.conv2d(img_feat,bottle_neck,1,activation=None,name="non_local_p")
		p_net = tf.reshape(p_net,[batch_size,-1,bottle_neck])
		p_net= tf.transpose(p_net,[0,2,1])
		
		g_pc_net = tf.layers.conv2d(pc_feat,bottle_neck,1,activation=None,name="non_local_g_pc")
		g_pc_net = tf.reshape(g_pc_net,[batch_size,-1,bottle_neck])
		g_img_net = tf.layers.conv2d(img_feat,bottle_neck,1,activation=None,name="non_local_g_img")
		g_img_net = tf.reshape(g_img_net,[batch_size,-1,bottle_neck])
		
		t_p_relation = tf.matmul(t_net,p_net)
		t_p_relation_transpose = tf.transpose(t_p_relation,[0,2,1])
		t_p_g_pc = tf.matmul(t_p_relation/80,g_img_net)
		t_p_g_img = tf.matmul(t_p_relation_transpose/4096,g_pc_net)
		
		t_p_g_pc = tf.reshape(t_p_g_pc,[batch_size,point_num,1,bottle_neck])
		t_p_g_img = tf.reshape(t_p_g_img,[batch_size,img_h,img_w,bottle_neck])
		
		non_local_pc = tf.layers.conv2d(t_p_g_pc,pc_feat_size,1,activation=None,name="non_local_out_pc")
		non_local_img = tf.layers.conv2d(t_p_g_img,img_feat_size,1,activation=None,name="non_local_out_img")
		
		'''
		print(t_net)
		print(p_net)
		print(g_pc_net)
		print(g_img_net)
		print(t_p_relation)
		print(t_p_relation_transpose)
		print(t_p_g_pc)
		print(t_p_g_img)
		print(non_local_pc)
		print(non_local_img)
		'''
	
	pc_feat = pc_feat + non_local_pc
	img_feat = img_feat + non_local_img
	
	with tf.variable_scope("vlad_var"):
		NetVLAD = lp.NetVLAD(feature_size=pc_feat_size, max_samples=point_num,cluster_size=cluster_size,output_dim=EMBBED_SIZE, gating=True, add_batch_norm=True,is_training=is_training)
		pc_feat= tf.reshape(pc_feat,[-1,pc_feat_size])
		pc_feat = tf.nn.l2_normalize(pc_feat,1)
		pc_feat = NetVLAD.forward(pc_feat)
		#normalize to have norm 1
		pc_feat = tf.nn.l2_normalize(pc_feat,1)
		pc_feat =  tf.reshape(pc_feat,[-1,EMBBED_SIZE])
	
	with tf.variable_scope("img_var"):
		img_feat = tf.reduce_mean(img_feat, [1, 2], name='pool5')
		img_feat = tf.nn.l2_normalize(img_feat,1)
	
	pcai_feat = tf.concat((pc_feat,img_feat),axis=1)
	return pcai_feat,pc_feat,img_feat

def init_pcainetwork():
	#training step
	with tf.variable_scope("bn_decay"):
		step = tf.Variable(0)
	pc_trans_feat = None
	trans_mat_placeholder = None
	#init sub-network
	if TRAINING_MODE != 2:
		pc_placeholder, is_training_pl, pc_feat = init_pcnetwork(step)
	if TRAINING_MODE != 1:
		img_placeholder, img_feat = init_imgnetwork()
	if TRAINING_MODE == 3:
		pcai_feat,pc_feat,img_feat = init_fusion_network(pc_feat,img_feat)
	
	#prepare data and loss
	if TRAINING_MODE != 2:
		pc_feat = tf.reshape(pc_feat,[FEAT_BATCH_SIZE,BATCH_DATA_SIZE,pc_feat.shape[1]])
		q_pc_vec, pos_pc_vec, neg_pc_vec, oth_pc_vec = tf.split(pc_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
		pc_loss = lazy_quadruplet_loss(q_pc_vec, pos_pc_vec, neg_pc_vec, oth_pc_vec, MARGIN1, MARGIN2)
		tf.summary.scalar('pc_loss', pc_loss)

		
	if TRAINING_MODE != 1:		
		img_feat = tf.reshape(img_feat,[FEAT_BATCH_SIZE,BATCH_DATA_SIZE,img_feat.shape[1]])
		q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec = tf.split(img_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
		img_loss = lazy_quadruplet_loss(q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec, MARGIN1, MARGIN2)
		tf.summary.scalar('img_loss', img_loss)		
	
	
	if TRAINING_MODE == 3:
		pcai_feat = tf.reshape(pcai_feat,[FEAT_BATCH_SIZE,BATCH_DATA_SIZE,pcai_feat.shape[1]])
		q_vec, pos_vec, neg_vec, oth_vec = tf.split(pcai_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
		finally_loss = lazy_quadruplet_loss(q_vec, pos_vec, neg_vec, oth_vec, MARGIN1, MARGIN2)
		all_loss = finally_loss + 3*pc_loss + img_loss
		tf.summary.scalar('finally_loss', finally_loss)
		tf.summary.scalar('all_loss', all_loss)
		
		
	#learning rate strategy, all in one?
	epoch_num_placeholder = tf.placeholder(tf.float32, shape=())
	learning_rate = get_learning_rate(epoch_num_placeholder)
	tf.summary.scalar('learning_rate', learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	
	#variable update
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	variables = tf.trainable_variables()
	
	pc_variable = [v for v in variables if v.name.split('/')[0] =='pc_var']
	resnet_variable = [v for v in variables if v.name.split('/')[0] =='img_var' and v.name.split('/')[1] == 'resnet_v1_50']
	local_fusion_variable = [v for v in variables if v.name.split('/')[0] =='fusion_var']
	vlad_variable = [v for v in variables if v.name.split('/')[0] =='vlad_var']
	
	'''
	print("sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
	
	for var in variables:
		print(var)
	
	print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
	
	for var in pc_variable:
		print(var)
	
	print("ppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp")
	
	for var in resnet_variable:
		print(var)
	
	print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
	
	for var in local_fusion_variable:
		print(var)
	
	print("lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")
	
	for var in vlad_variable:
		print(var)
	
	print("gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
	exit()
	'''
	
	
	#training operation
	with tf.control_dependencies(update_ops):
		if TRAINING_MODE != 2:
			pc_train_op = optimizer.minimize(pc_loss, global_step=step, var_list=vlad_variable+local_fusion_variable)
		if TRAINING_MODE != 1:
			img_train_op = optimizer.minimize(img_loss, global_step=step, var_list=resnet_variable)
		if TRAINING_MODE == 3:
			all_train_op = optimizer.minimize(all_loss, global_step=step)
			fusion_train_op = optimizer.minimize(finally_loss, global_step=step,var_list=local_fusion_variable)
			
			
	
	
	
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
			"fusion_train_op":fusion_train_op,
			"all_train_op":all_train_op,
			"merged":merged,
			"pc_trans_feat":pc_trans_feat,
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
		train_saver['load_saver'].restore(sess,MODEL_PATH)
		print("all_model restored")
		#train_saver['pc_saver'].restore(sess,PC_MODEL_PATH)
		#print("pc_model restored")
		#train_saver['img_saver'].restore(sess,IMG_MODEL_PATH)
		#print("img_model restored")
		
		return

def init_train_saver():	
	all_saver = tf.train.Saver()
	
	variables = tf.contrib.framework.get_variables_to_restore()
	
	all_variable = [v for v in variables if v.name.split('/')[0] !='bn_decay']
	load_saver = tf.train.Saver(all_variable)
	

	pc_variable = [v for v in variables if v.name.split('/')[0] =='pc_var']
	#img_variable = [v for v in variables if v.name.split('/')[0] =='img_var']
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
		'img_saver':img_saver,
		'load_saver':load_saver}
	
	return train_saver
	
def prepare_batch_data(pc_data,img_data,trans_data,feat_batch,ops,ep):
	is_training = True
	if TRAINING_MODE != 2:
		feat_batch_pc = pc_data[feat_batch*BATCH_DATA_SIZE*FEAT_BATCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BATCH_SIZE]
	if TRAINING_MODE != 1:
		feat_batch_img = img_data[feat_batch*BATCH_DATA_SIZE*FEAT_BATCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BATCH_SIZE]
	if TRAINING_MODE == 3:
		feat_batch_trans = trans_data[feat_batch*BATCH_DATA_SIZE*FEAT_BATCH_SIZE:(feat_batch+1)*BATCH_DATA_SIZE*FEAT_BATCH_SIZE]
		
		
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
			#ops["trans_mat_placeholder"]:	feat_batch_trans,
			ops["epoch_num_placeholder"]:ep}
		return train_feed_dict
		
	print("prepare_batch_data_error,no_such train mode.")
	exit()

def train_one_step(sess,ops,train_feed_dict,train_writer,is_training = True):
	if not is_training:
		summary,step,all_loss = sess.run([ops["merged"],ops["step"],ops["all_loss"]],feed_dict = train_feed_dict)
		train_writer.add_summary(summary, step)
		return step,all_loss
		
		
	
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
			#pc_trans_feat,summary,step,all_loss,_,= sess.run([ops["pc_trans_feat"],ops["merged"],ops["step"],ops["all_loss"],ops["all_train_op"]],feed_dict = train_feed_dict)
			summary,step,all_loss,_= sess.run([ops["merged"],ops["step"],ops["all_loss"],ops["pc_train_op"]],feed_dict = train_feed_dict)
			#print("batch num = %d , all_loss = %f"%(step, all_loss))
								
	#other training strategy
	train_writer.add_summary(summary, step)
	return step,all_loss
	
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

def get_eval_batch_filename(eval_batch_key,quadruplet):
	pc_files = []
	img_files = []
	for key_cnt ,key in enumerate(eval_batch_key):
		pc_files.append(TEST_QUERIES[key]["query"])
		img_files.append("%s_stereo_centre.png"%(TEST_QUERIES[key]["query"][:-4]))
		random.shuffle(TEST_QUERIES[key]["positives"])

		cur_pos = 0;
		for i in range(POS_NUM):
			while True:
				filename = "%s_stereo_centre.png"%(TEST_QUERIES[TEST_QUERIES[key]["positives"][cur_pos]]["query"][:-4])

				if filename in img_files[BATCH_DATA_SIZE*(key_cnt)+1:BATCH_DATA_SIZE*(key_cnt)+1+i]:
					cur_pos = cur_pos+1
					continue
				if os.path.exists(filename):
					break
				cur_pos = cur_pos+1
				if cur_pos>len(TEST_QUERIES[key]["positives"]):
					print("line 259, error in positive number")
					exit()
			
			pc_files.append(TEST_QUERIES[TEST_QUERIES[key]["positives"][cur_pos]]["query"])
			img_files.append(filename)		
		
		neg_indices = []
		for i in range(NEG_NUM):
			while True:
				while True:
					neg_ind = random.randint(0,len(TEST_QUERIES.keys())-1)
					if is_negative(neg_ind,TEST_QUERIES[key]["not_negative"]):
						break
				

				filename = "%s_stereo_centre.png"%(TEST_QUERIES[neg_ind]["query"][:-4])
				if filename in img_files[BATCH_DATA_SIZE*(key_cnt)+1+POS_NUM:BATCH_DATA_SIZE*(key_cnt)+1+POS_NUM+i]:
					continue
				if os.path.exists(filename):
					break
					
			neg_indices.append(neg_ind)
			pc_files.append(TEST_QUERIES[neg_ind]["query"])
			img_files.append(filename)
		
		'''
		tmp_list = img_files[9*(key_cnt)+1+POS_NUM:9*(key_cnt)+1+POS_NUM+NEG_NUM]
		if len(tmp_list)!=len(set(tmp_list)):
			print("neg_duplicate")
			input()
		'''
		
		if quadruplet:
			neighbors=[]
			for pos in TEST_QUERIES[key]["positives"]:
				neighbors.append(pos)
			for neg in neg_indices:
				for pos in TEST_QUERIES[neg]["positives"]:
					neighbors.append(pos)
					
			#print("neighbors size = ",len(neighbors))
			while True:
				neg_ind = random.randint(0,len(TEST_QUERIES.keys())-1)
				if is_negative(neg_ind,neighbors):
					filename = "%s_stereo_centre.png"%(TEST_QUERIES[neg_ind]["query"][:-4])
					if os.path.exists(filename):
						break						
									
			pc_files.append(TEST_QUERIES[neg_ind]["query"])
			img_files.append(filename)
	
	if TRAINING_MODE == 1:
		return pc_files,None
	
	if TRAINING_MODE == 2:
		return None,img_files
		
	if TRAINING_MODE == 3:
		return pc_files,img_files
	
def get_load_batch_filename(load_batch_keys,quadruplet):		
	pc_files = []
	img_files = []
	for key_cnt ,key in enumerate(load_batch_keys):
		pc_files.append(TRAINING_QUERIES[key]["query"])
		img_files.append("%s_stereo_centre.png"%(TRAINING_QUERIES[key]["query"][:-4]))
		random.shuffle(TRAINING_QUERIES[key]["positives"])
		
		#print(TRAINING_QUERIES[key])
		cur_pos = 0;
		for i in range(POS_NUM):
			while True:
				filename = "%s_stereo_centre.png"%(TRAINING_QUERIES[TRAINING_QUERIES[key]["positives"][cur_pos]]["query"][:-4])

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
				

				filename = "%s_stereo_centre.png"%(TRAINING_QUERIES[neg_ind]["query"][:-4])
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
					filename = "%s_stereo_centre.png"%(TRAINING_QUERIES[neg_ind]["query"][:-4])
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
	
def get_eval_keys():
	eval_file_num = len(TEST_QUERIES.keys())
	eval_file_idxs = np.arange(0,eval_file_num)
	np.random.shuffle(eval_file_idxs)
	
	load_batch_keys = []
		
	eval_load = 0
	while len(load_batch_keys) < FEAT_BATCH_SIZE:			
		cur_key = eval_file_idxs[eval_load]
				
		if len(TEST_QUERIES[cur_key]["positives"]) < POS_NUM:
			eval_load = eval_load + 1
			continue
		
		filename = "%s_stereo_centre.png"%(TEST_QUERIES[cur_key]["query"][:-4])
		if not os.path.exists(filename):
			#print(TRAINING_QUERIES[cur_key]["query"])
			eval_load = eval_load + 1
			continue
		
		valid_pos = 0
		for i in range(len(TEST_QUERIES[cur_key]["positives"])):
			filename = "%s_stereo_centre.png"%(TEST_QUERIES[TEST_QUERIES[cur_key]["positives"][i]]["query"][:-4])
			if os.path.exists(filename):
					valid_pos = valid_pos + 1
				
		if valid_pos < POS_NUM:
			eval_load = eval_load + 1
			continue
						
		load_batch_keys.append(eval_file_idxs[eval_load])
		eval_load = eval_load + 1
		
	return False,load_batch_keys
	

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
		
		filename = "%s_stereo_centre.png"%(TRAINING_QUERIES[cur_key]["query"][:-4])
		if not os.path.exists(filename):
			#print(TRAINING_QUERIES[cur_key]["query"])
			CUR_LOAD = CUR_LOAD + 1
			skip_num = skip_num + 1
			continue
		
		valid_pos = 0
		for i in range(len(TRAINING_QUERIES[cur_key]["positives"])):
			filename = "%s_stereo_centre.png"%(TRAINING_QUERIES[TRAINING_QUERIES[cur_key]["positives"][i]]["query"][:-4])
			if os.path.exists(filename):
					valid_pos = valid_pos + 1
				
		if valid_pos < POS_NUM:
			skip_num = skip_num + 1
			CUR_LOAD = CUR_LOAD + 1
			continue
						
		load_batch_keys.append(train_file_idxs[CUR_LOAD])
		CUR_LOAD = CUR_LOAD + 1
		
	return False,load_batch_keys

def cal_trans_data(pc_dict,cnt = -1):
	posfile = pc_dict[0]
	pointcloud = pc_dict[1]
	pointcloud = np.hstack([pointcloud, np.ones((pointcloud.shape[0],1))])
	
	imgpos = {}
	with open(posfile) as imgpos_file:
		for line in imgpos_file:
			pos = [x for x in line.split(' ')]
			for i in range(len(pos)-2):
				pos[i+1] = float(pos[i+1])
			imgpos[pos[0]] = np.reshape(np.array(pos[1:-1]),[4,4])
	
	#translate pointcloud to image coordinate
	pointcloud = np.dot(np.linalg.inv(imgpos["stereo_centre"]),pointcloud.T)
	pointcloud = np.dot(G_CAMERA_POSESOURCE, pointcloud)
	uv = CAMERA_MODEL.project(pointcloud, [1280,960])	
	
	
	#print(CAMERA_MODEL.bilinear_lut[:, 1::-1].shape)
	lut = CAMERA_MODEL.bilinear_lut[:, 1::-1].T.reshape((2, 960, 1280))
	lut = np.swapaxes(lut,2,1)
	u = map_coordinates(lut[0, :, :], uv, order=1)
	v = map_coordinates(lut[1, :, :], uv, order=1)
	uv = np.array([u,v])
	
	zero0 = np.where(uv[1,:] == 0)
	zero1 = np.where(uv[0,:] == 0)
	zero_01 = np.intersect1d(zero0,zero1)
	nozero = np.setdiff1d(np.arange(4096),zero_01)
	uv = np.delete(uv,zero_01.tolist(),axis=1)
	#print(max(uv[0,:]),max(uv[1,:]))
	
	if cnt == 0:
		uv_show = uv/4
		plt.scatter(np.ravel(uv_show[1, :]), np.ravel(uv_show[0, :]), s=5, edgecolors='none', cmap='jet')
		plt.xlim(0, 320)
		plt.ylim(240, 0)
		plt.xticks([])
		plt.yticks([])
		plt.savefig("test.png")
		plt.cla()
		
	
	transform_matrix = np.zeros([80*4096,1])
	u = np.floor(uv[0,:]/120)
	v = np.floor(uv[1,:]/128)
	row = u*10 + v
	#print(min(u),min(v),min(row))
	#print(max(u),max(v),max(row))
	
	row1 = (row*4096+nozero).astype(int).tolist()
	transform_matrix[row1] = 1
	transform_matrix = transform_matrix.reshape([80,4096])
	
	'''
	aa = np.sum(transform_matrix,1).reshape([8,10])
	print(np.sum(aa))
	plt.figure(2)
	plt.imshow(aa)	
	plt.show()
	input()
	exit()
	'''
	
	return transform_matrix
	
	
	
def get_trans_datas(load_pc_filenames,pc_data,pool):
	dict_list = []
	for i in range(pc_data.shape[0]):
		para = ("%s_imgpos.txt"%(load_pc_filenames[i][:-4]),pc_data[i,:,:])
		#print(para_dict)
		dict_list.append(para)
	
	#trans_data = []
	#for i in range(len(dict_list)):
	#	trans_data.append(cal_trans_data(dict_list[i],i))
	#return
	
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
		
		is_training = True
		
		if (cnt*LOAD_FEAT_RATIO)%300 == 0:
			is_training = False
			_,eval_batch_key = get_eval_keys()
			#select load_batch tuple
			eval_pc_filenames,eval_img_filenames = get_eval_batch_filename(eval_batch_key,quadruplet)
			#load pc&img data from file
			pc_data,img_data = load_img_pc(eval_pc_filenames,eval_img_filenames,pool,False)
			
			trans_data = None
			if TRAINING_MODE != 2:
				trans_data = get_trans_datas(eval_pc_filenames,pc_data,pool)
				for i in range(len(pc_data)):
					posfile = "%s_imgpos.txt"%(eval_pc_filenames[i][:-4])
					cur_pc = pc_data[i]
					
					cur_pc = np.hstack([cur_pc, np.ones((cur_pc.shape[0],1))])
					imgpos = {}
					with open(posfile) as imgpos_file:
						for line in imgpos_file:
							pos = [x for x in line.split(' ')]
							for j in range(len(pos)-2):
								pos[j+1] = float(pos[j+1])
							imgpos[pos[0]] = np.reshape(np.array(pos[1:-1]),[4,4])
					#translate pointcloud to image coordinate
					cur_pc = np.dot(np.linalg.inv(imgpos["stereo_centre"]),cur_pc.T)
					cur_pc = np.dot(G_CAMERA_POSESOURCE, cur_pc)
					cur_pc = cur_pc[0:3,:].T
		
					mean_cur_pc = cur_pc.mean(axis = 0)
					cur_pc = cur_pc - mean_cur_pc
					mean_cur_pc = cur_pc.mean(axis = 0)
					cur_pc = cur_pc - mean_cur_pc
					
					scale = 0.5/(np.sum(np.linalg.norm(cur_pc, axis=1, keepdims=True))/cur_pc.shape[0])
					T = scale*np.eye(4)
					T[-1,-1] = 1
					cur_pc = np.hstack([cur_pc, np.ones((cur_pc.shape[0],1))])
					cur_pc = np.dot(T,cur_pc.T)
					pc_data[i] = cur_pc[0:3,:].T
			print("load evaluate batch",cnt)
		
		else:		
			BATCH_REACH_END,load_batch_keys = get_batch_keys(train_file_idxs,train_file_idxs.shape[0])
			if BATCH_REACH_END:
				print("load thread ended---------------------------------------------------------------------------------------------------")
				break
			
			
			#select load_batch tuple
			load_pc_filenames,load_img_filenames = get_load_batch_filename(load_batch_keys,quadruplet)
			
			#load pc&img data from file
			pc_data,img_data = load_img_pc(load_pc_filenames,load_img_filenames,pool,False)
			
			trans_data = None
			if TRAINING_MODE != 2:
				trans_data = get_trans_datas(load_pc_filenames,pc_data,pool)
				for i in range(len(pc_data)):
					posfile = "%s_imgpos.txt"%(load_pc_filenames[i][:-4])
					cur_pc = pc_data[i]
					
					cur_pc = np.hstack([cur_pc, np.ones((cur_pc.shape[0],1))])
					imgpos = {}
					with open(posfile) as imgpos_file:
						for line in imgpos_file:
							pos = [x for x in line.split(' ')]
							for j in range(len(pos)-2):
								pos[j+1] = float(pos[j+1])
							imgpos[pos[0]] = np.reshape(np.array(pos[1:-1]),[4,4])
					#translate pointcloud to image coordinate
					cur_pc = np.dot(np.linalg.inv(imgpos["stereo_centre"]),cur_pc.T)
					cur_pc = np.dot(G_CAMERA_POSESOURCE, cur_pc)
					cur_pc = cur_pc[0:3,:].T
		
					mean_cur_pc = cur_pc.mean(axis = 0)
					cur_pc = cur_pc - mean_cur_pc
					mean_cur_pc = cur_pc.mean(axis = 0)
					cur_pc = cur_pc - mean_cur_pc
					
					scale = 0.5/(np.sum(np.linalg.norm(cur_pc, axis=1, keepdims=True))/cur_pc.shape[0])
					T = scale*np.eye(4)
					T[-1,-1] = 1
					cur_pc = np.hstack([cur_pc, np.ones((cur_pc.shape[0],1))])
					cur_pc = np.dot(T,cur_pc.T)
					pc_data[i] = cur_pc[0:3,:].T
			print("load training batch",cnt)
					
		TRAINING_DATA_LOCK.acquire()
		TRAINING_DATA.append([pc_data,img_data,trans_data,is_training])
		TRAINING_DATA_LOCK.release()
		cnt = cnt + 1
	
	return
	

def training(sess,train_saver,train_writer,eval_writer,ops):
	global BATCH_REACH_END
	global EP
	global EPOCH
	
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
		is_training = cur_batch_data[3]
		
		print("consume one batch")
		
		if(is_training):
			for feat_batch in range(LOAD_FEAT_RATIO):
				#prepare this batch data
				train_feed_dict = prepare_batch_data(pc_data,img_data,trans_data,feat_batch,ops,EP)
												
				#training
				step,all_loss = train_one_step(sess,ops,train_feed_dict,train_writer,is_training)
				print("batch num = %d , all_loss = %f"%(step, all_loss))
							
				if step%3001 == 0 and EP > EPOCH-2:
					model_save(sess,step,train_saver)
		else:
			eval_feed_dict = prepare_batch_data(pc_data,img_data,trans_data,0,ops,EP)
			eval_step,eval_loss = train_one_step(sess,ops,eval_feed_dict,eval_writer,is_training)
			print("																evaluate loss = %f"%(eval_loss))
			
	
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
		train_writer = tf.summary.FileWriter(os.path.join(LOG_PATH, 'train'), sess.graph)
		eval_writer = tf.summary.FileWriter(os.path.join(LOG_PATH, 'eval'))
		
		#init_training thread
		training_thread = threading.Thread(target=training, args=(sess,train_saver,train_writer,eval_writer,ops,))
		training_thread.start()

		#start training
		for ep in range(EPOCH):
			train_file_num = len(TRAINING_QUERIES.keys())
			train_file_idxs = np.arange(0,train_file_num)
			np.random.shuffle(train_file_idxs)
			print('Eppch = %d, train_file_num = %f , FEAT_BATCH_SIZE = %f , iteration per batch = %f' %(ep,len(train_file_idxs), FEAT_BATCH_SIZE,len(train_file_idxs)//FEAT_BATCH_SIZE))
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