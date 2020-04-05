import numpy as np
from loading_input_v3 import *
from pointnetvlad_v3.pointnetvlad_trans import *
import pointnetvlad_v3.loupe as lp
import nets_v3.resnet_v1_trans as resnet
import tensorflow as tf
from time import *
import pickle
from multiprocessing.dummy import Pool as ThreadPool
sys.path.append('/data/lyh/lab/robotcar-dataset-sdk/python')
from camera_model import CameraModel
from transform import build_se3_transform
from image import load_image
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


#thread pool
pool = ThreadPool(10)

# 1 for point cloud only, 2 for image only, 3 for pc&img&fc
TRAINING_MODE = 3
BATCH_SIZE = 100
EMBBED_SIZE = 1000

DATABASE_FILE= 'generate_queries_v3/stereo_centre_trans_RobotCar_ground_oxford_evaluation_database.pickle'
QUERY_FILE= 'generate_queries_v3/stereo_centre_trans_RobotCar_ground_oxford_evaluation_query.pickle'
DATABASE_SETS= get_sets_dict(DATABASE_FILE)
QUERY_SETS= get_sets_dict(QUERY_FILE)

#model_path & image path
PC_MODEL_PATH = ""
IMG_MODEL_PATH = ""
MODEL_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v3_baseline_select/log/train_save_trans_data_5/model_00456152.ckpt"

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

def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
		pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)
	
def save_feat_to_file(database_feat,query_feat):
	if TRAINING_MODE == 1:
		output_to_file(database_feat["pc_feat"],"database_pc_feat_"+PC_MODEL_PATH[-13:-5]+".pickle")
		output_to_file(query_feat["pc_feat"],"query_pc_feat_"+PC_MODEL_PATH[-13:-5]+".pickle")
	
	if TRAINING_MODE == 2:
		output_to_file(database_feat["img_feat"],"database_img_feat_"+IMG_MODEL_PATH[-13:-5]+".pickle")
		output_to_file(query_feat["img_feat"],"query_img_feat_"+IMG_MODEL_PATH[-13:-5]+".pickle")
	
	if TRAINING_MODE == 3:
		output_to_file(database_feat["pc_feat"],"database_pc_feat_"+MODEL_PATH[-13:-5]+".pickle")
		output_to_file(query_feat["pc_feat"],"query_pc_feat_"+MODEL_PATH[-13:-5]+".pickle")
		output_to_file(database_feat["img_feat"],"database_img_feat_"+MODEL_PATH[-13:-5]+".pickle")
		output_to_file(query_feat["img_feat"],"query_img_feat_"+MODEL_PATH[-13:-5]+".pickle")
		output_to_file(database_feat["pcai_feat"],"database_pcai_feat_"+MODEL_PATH[-13:-5]+".pickle")
		output_to_file(query_feat["pcai_feat"],"query_pcai_feat_"+MODEL_PATH[-13:-5]+".pickle")
	
def get_load_batch_filename(dict_to_process,batch_keys,edge = False,remind_index = 0):
	pc_files = []
	img_files = []
	
	if edge :
		for i in range(BATCH_SIZE):
			cur_index = min(remind_index-1,i)
			pc_files.append(dict_to_process[batch_keys[cur_index]]["query"])			
			img_files.append("%s_stereo_centre.png"%(dict_to_process[batch_keys[cur_index]]["query"][:-4]))
	else:
		for i in range(BATCH_SIZE):
			pc_files.append(dict_to_process[batch_keys[i]]["query"])
			img_files.append("%s_stereo_centre.png"%(dict_to_process[batch_keys[i]]["query"][:-4]))
	
	if TRAINING_MODE == 1:
		return pc_files,None
	if TRAINING_MODE == 2:
		return None,img_files
	if TRAINING_MODE == 3:
		return pc_files,img_files

def prepare_batch_data(pc_data,img_data,trans_data,ops):
	is_training=False
	if TRAINING_MODE == 1:
		train_feed_dict = {
			ops["is_training_pl"]:is_training,
			ops["pc_placeholder"]:pc_data}
		return train_feed_dict
		
	if TRAINING_MODE == 2:
		train_feed_dict = {
			ops["img_placeholder"]:img_data}
		return train_feed_dict
		
	if TRAINING_MODE == 3:
		train_feed_dict = {
			ops["is_training_pl"]:is_training,
			ops["img_placeholder"]:img_data,
			ops["pc_placeholder"]:pc_data,
			ops["trans_mat_placeholder"]:trans_data}
		return train_feed_dict
		
	print("prepare_batch_data_error,no_such train mode.")
	exit()

def train_one_step(sess,ops,train_feed_dict):
	if TRAINING_MODE == 1:
		pc_feat= sess.run([ops["pc_feat"]],feed_dict = train_feed_dict)
		feat = {
			"pc_feat":pc_feat[0]}
		return feat

	if TRAINING_MODE == 2:
		img_feat= sess.run([ops["img_feat"]],feed_dict = train_feed_dict)
		feat = {
			"img_feat":img_feat[0]}
		return feat
		
	if TRAINING_MODE == 3:
		pc_feat,img_feat,pcai_feat= sess.run([ops["pc_feat"],ops["img_feat"],ops["pcai_feat"]],feed_dict = train_feed_dict)
		feat = {
			"pc_feat":pc_feat,
			"img_feat":img_feat,
			"pcai_feat":pcai_feat}
		return feat
		
def init_all_feat():
	if TRAINING_MODE != 2:
		pc_feat = np.empty([0,1000],dtype=np.float32)
	if TRAINING_MODE != 1:
		img_feat = np.empty([0,1000],dtype=np.float32)
	if TRAINING_MODE == 3:
		pcai_feat = np.empty([0,1000],dtype=np.float32)
	
	if TRAINING_MODE == 1:
		all_feat = {"pc_feat":pc_feat}
	if TRAINING_MODE == 2:
		all_feat = {"img_feat":img_feat}
	if TRAINING_MODE == 3:
		all_feat = {
			"pc_feat":pc_feat,
			"img_feat":img_feat,
			"pcai_feat":pcai_feat}
	
	return all_feat
	
def concatnate_all_feat(all_feat,feat):
	if TRAINING_MODE == 1:
		all_feat["pc_feat"] = np.concatenate((all_feat["pc_feat"],feat["pc_feat"]),axis=0)
	if TRAINING_MODE == 2:
		all_feat["img_feat"] = np.concatenate((all_feat["img_feat"],feat["img_feat"]),axis=0)
	if TRAINING_MODE == 3:
		all_feat["pc_feat"] = np.concatenate((all_feat["pc_feat"],feat["pc_feat"]),axis=0)
		all_feat["img_feat"] = np.concatenate((all_feat["img_feat"],feat["img_feat"]),axis=0)
		all_feat["pcai_feat"] = np.concatenate((all_feat["pcai_feat"],feat["pcai_feat"]),axis=0)
	return all_feat

def get_unique_all_feat(all_feat,dict_to_process):
	if TRAINING_MODE == 1:
		all_feat["pc_feat"] = all_feat["pc_feat"][0:len(dict_to_process.keys()),:]
	if TRAINING_MODE == 2:
		all_feat["img_feat"] = all_feat["img_feat"][0:len(dict_to_process.keys()),:]
	if TRAINING_MODE == 3:
		all_feat["pc_feat"] = all_feat["pc_feat"][0:len(dict_to_process.keys()),:]
		all_feat["img_feat"] = all_feat["img_feat"][0:len(dict_to_process.keys()),:]	
		all_feat["pcai_feat"] = all_feat["pcai_feat"][0:len(dict_to_process.keys()),:]			
	return all_feat

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


def get_latent_vectors(sess,ops,dict_to_process):
	print("dict_size = ",len(dict_to_process.keys()))
	train_file_idxs = np.arange(0,len(dict_to_process.keys()))
	all_feat = init_all_feat()
	for i in range(len(train_file_idxs)//BATCH_SIZE):
		batch_keys = train_file_idxs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
		pc_files=[]
		img_files=[]
		if i<0:
			print("Error, ready for delete")
			continue
		
		
		#select load_batch tuple
		load_pc_filenames,load_img_filenames = get_load_batch_filename(dict_to_process,batch_keys)
		
		begin_time = time()
		
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
		
		end_time = time()
		
		print ('load time ',end_time - begin_time)
		
		train_feed_dict = prepare_batch_data(pc_data,img_data,trans_data,ops)
		
		begin_time = time()
		feat = train_one_step(sess,ops,train_feed_dict)
		end_time = time()
		print ('feature time ',end_time - begin_time)
		
		all_feat = concatnate_all_feat(all_feat,feat)
		
	#no edge case
	if len(train_file_idxs)%BATCH_SIZE == 0:
		return all_feat
	
	#hold edge case
	remind_index = len(train_file_idxs)%BATCH_SIZE
	tot_batches = len(train_file_idxs)//BATCH_SIZE		
	batch_keys = train_file_idxs[tot_batches*BATCH_SIZE:tot_batches*BATCH_SIZE+remind_index]
	
	load_pc_filenames,load_img_filenames = get_load_batch_filename(dict_to_process,batch_keys,True,remind_index)
	
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
		
	
	train_feed_dict = prepare_batch_data(pc_data,img_data,trans_data,ops)
	
	feat = train_one_step(sess,ops,train_feed_dict)
	
	all_feat = concatnate_all_feat(all_feat,feat)
	all_feat = get_unique_all_feat(all_feat,dict_to_process)
	return all_feat
	
def	append_feat(all_feat,cur_feat):
	if TRAINING_MODE != 2:
		all_feat["pc_feat"].append(cur_feat["pc_feat"])
	if TRAINING_MODE != 1:
		all_feat["img_feat"].append(cur_feat["img_feat"])
	if TRAINING_MODE == 3:
		all_feat["pcai_feat"].append(cur_feat["pcai_feat"])
	return all_feat
	
def cal_all_features(ops,sess):
	if TRAINING_MODE != 2:
		database_pc_feat = []
		query_pc_feat = []
	if TRAINING_MODE != 1:
		database_img_feat = []
		query_img_feat = []
	if TRAINING_MODE == 3:
		database_pcai_feat = []
		query_pcai_feat = []
		
	if TRAINING_MODE == 1:
		database_feat = {
			"pc_feat":database_pc_feat}
		query_feat = {
			"pc_feat":query_pc_feat}
	if TRAINING_MODE == 2:
		database_feat = {
			"img_feat":database_img_feat}
		query_feat = {
			"img_feat":query_img_feat}
	if TRAINING_MODE == 3:
		database_feat = {
			"pc_feat":database_pc_feat,
			"img_feat":database_img_feat,
			"pcai_feat":database_pcai_feat}
		query_feat = {
			"pc_feat":query_pc_feat,
			"img_feat":query_img_feat,
			"pcai_feat":query_pcai_feat}
	
	
	for i in range(len(DATABASE_SETS)):
		cur_feat = get_latent_vectors(sess, ops, DATABASE_SETS[i])
		
		database_feat = append_feat(database_feat,cur_feat)
			
	for j in range(len(QUERY_SETS)):
		cur_feat = get_latent_vectors(sess, ops, QUERY_SETS[j])
		query_feat = append_feat(query_feat,cur_feat)
	
	save_feat_to_file(database_feat,query_feat)	

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
	bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,step*BATCH_SIZE,BN_DECAY_DECAY_STEP,BN_DECAY_DECAY_RATE,staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay


def init_imgnetwork(pc_trans_feat):
	with tf.variable_scope("img_var"):
		img_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,240,320,3])
		img_feat, img_pc_feat = resnet.endpoints(img_placeholder,pc_trans_feat,is_training=False)
	return img_placeholder, img_feat, img_pc_feat
	

def init_pcnetwork(step):
	with tf.variable_scope("pc_var"):
		pc_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,4096,3])
		trans_mat_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,80,4096])
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
		
		
	#output of pcainetwork init
	if TRAINING_MODE == 1:
		ops = {
			"is_training_pl":is_training_pl,
			"pc_placeholder":pc_placeholder,
			"pc_feat":pc_feat}
		return ops
		
	if TRAINING_MODE == 2:
		ops = {
			"img_placeholder":img_placeholder,
			"img_feat":img_feat}
		return ops
		
	if TRAINING_MODE == 3:
		ops = {
			"is_training_pl":is_training_pl,
			"pc_placeholder":pc_placeholder,
			"img_placeholder":img_placeholder,
			"trans_mat_placeholder":trans_mat_placeholder,
			"pc_feat":pc_feat,
			"img_feat":img_feat,
			"pcai_feat":pcai_feat}
		return ops
		
		
def init_network_variable(sess,train_saver):
	sess.run(tf.global_variables_initializer())
	
	if TRAINING_MODE == 1:
		train_saver['pc_saver'].restore(sess,PC_MODEL_PATH)
		print("pc_model restored")
		return
		
	if TRAINING_MODE == 2:
		train_saver['img_saver'].restore(sess,IMG_MODEL_PATH)
		print("img_model restored")
		return
	
	if TRAINING_MODE == 3:
		train_saver['all_saver'].restore(sess,MODEL_PATH)
		print("all_model restored")
		return


def init_train_saver():
	all_saver = tf.train.Saver()
	variables = tf.contrib.framework.get_variables_to_restore()
	pc_variable = [v for v in variables if v.name.split('/')[0] =='pc_var']
	img_variable = [v for v in variables if v.name.split('/')[0] =='img_var']
	
	'''
	other_var = pc_variable = [v for v in variables if v.name.split('/')[0] !='pc_var']
	for var in other_var:
		print(var)
	exit()
	'''
	
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
	

def main():
	#init_camera_model_posture
	init_camera_model_posture()
	
	#init network pipeline
	ops = init_pcainetwork()
	
	#init train saver
	train_saver = init_train_saver()

	#init GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	
	init_network_variable(sess,train_saver)
	print("model restored")
	
	cal_all_features(ops,sess)


if __name__ == "__main__":
	main()
