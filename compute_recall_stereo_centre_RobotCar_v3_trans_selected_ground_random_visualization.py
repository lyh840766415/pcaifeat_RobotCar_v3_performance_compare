import numpy as np
import pickle
from loading_input_v3 import *
from sklearn.neighbors import KDTree
import random
import shutil
import pandas as pd

def get_queries_dict(filename):
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("feature Loaded.")
		return queries

DATABASE_VECTORS_FILENAME = "database_pcai_feat_00441147.pickle"
QUERY_VECTORS_FILENAME = "query_pcai_feat_00441147.pickle"

#result output
output_file = "result_img_trans_mono_left_00240080.txt"
#load feature
DATABASE_VECTORS = get_queries_dict(DATABASE_VECTORS_FILENAME)
QUERY_VECTORS = get_queries_dict(QUERY_VECTORS_FILENAME)
#load label
QUERY_FILE= 'generate_queries_v3/stereo_centre_trans_RobotCar_ground_selected_oxford_evaluation_query.pickle'
QUERY_SETS= get_sets_dict(QUERY_FILE)

SEARCH_FIELD = 0
SEARCH_SUCCEED = 0

RANDOM_TIME = 0

NIGHT_FAILED = 0

IMG_PATH = "/data/lyh/RobotCar/pc_img_ground_0310/20m_20dis_color_resize"
PC_PATH = "/data/lyh/RobotCar/pc_img_ground_0310/20m_20dis"

df_locations = pd.read_csv("/data/lyh/RobotCar/pc_img_ground_0310/20m_20dis/2014-05-19-13-20-57/pointcloud_locations_20m_20dis.csv",sep=',')
pc_filename = os.path.join(PC_PATH,"2014-05-19-13-20-57","pointclouds","%d.txt"%(df_locations['timestamp'][10]))
img_filename = os.path.join(PC_PATH,"2014-05-19-13-20-57","pointclouds","%d_stereo_centre.png"%(df_locations['timestamp'][10]))
print(img_filename)
if(os.path.exists(img_filename)):
	print("exist")

if not os.path.exists("bad_case"):
	os.makedirs("bad_case")

night_seq = [0,4,7]
SEQ_NAME = [
"2014-11-14-16-34-33",
"2014-11-18-13-20-12",
"2014-12-02-15-30-08",
"2014-12-09-13-21-02",
"2014-12-10-18-10-50",
"2014-12-12-10-45-15",
"2014-12-16-09-14-09",
"2014-12-16-18-44-24",
"2015-02-03-08-45-10",
"2015-02-10-11-58-05",
"2015-02-13-09-16-26",
"2015-02-17-14-42-12",
"2015-03-10-14-18-10",
"2015-03-17-11-08-44",
"2015-05-19-14-06-38",
"2015-06-09-15-06-29",
"2015-08-12-15-04-18",
"2015-08-13-16-02-58",
"2015-08-14-14-54-57",
"2015-08-28-09-50-22",
"2015-10-30-13-52-14",
"2015-11-12-11-22-05",
"2015-11-13-10-28-08"]

for m in range(len(QUERY_SETS)):
	print(len(QUERY_VECTORS[m]))
	print(len(QUERY_SETS[m]))
	print(len(DATABASE_VECTORS[m]))
	if len(QUERY_SETS[m]) != len(QUERY_VECTORS[m]):
		print("not equal")


def get_recall(m, n):
	global DATABASE_VECTORS
	global QUERY_VECTORS
	global SEARCH_FIELD
	global SEARCH_SUCCEED
	global RANDOM_TIME
	global NIGHT_FAILED

	database_output= DATABASE_VECTORS[m]
	queries_output= QUERY_VECTORS[n]

	print(len(queries_output))
	database_nbrs = KDTree(database_output)

	num_neighbors=25
	recall=[0]*num_neighbors

	top1_similarity_score=[]
	one_percent_retrieved=0
	threshold=max(int(round(len(database_output)/100.0)),1)

	num_evaluated=0
	for i in range(len(queries_output)):
		true_neighbors= QUERY_SETS[n][i][m]
		if(len(true_neighbors)==0):
			continue
		num_evaluated+=1
		
		distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=num_neighbors)
		for j in range(len(indices[0])):
			if indices[0][j] in true_neighbors:
				if(j==0):
					similarity= np.dot(queries_output[i],database_output[indices[0][j]])
					top1_similarity_score.append(similarity)
				recall[j]+=1
				break
				
		if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))>0:
			one_percent_retrieved+=1
			SEARCH_SUCCEED = SEARCH_SUCCEED + 1
		else:
			SEARCH_FIELD = SEARCH_FIELD + 1
			print("search failed query seq ", SEQ_NAME[n] ," query idx ",i," retrived ",indices[0][0:threshold]," database seq ",SEQ_NAME[m]," ground truth ",true_neighbors)
			
			if m in night_seq or n in night_seq:
				NIGHT_FAILED = NIGHT_FAILED + 1
				
			'''
			if random.randint(0,599) == 7:
				RANDOM_TIME = RANDOM_TIME + 1
				
				#copy query
				pc_filename = QUERY_SETS[n][i]['query'].split('/')[-1]
				query_pc_filename = os.path.join(PC_PATH,SEQ_NAME[n],"pointclouds","%s.txt"%(pc_filename[:-4]))
				query_img_filename = os.path.join(IMG_PATH,SEQ_NAME[n],"pointclouds","%s_stereo_centre.png"%(pc_filename[:-4]))
				
				print(query_pc_filename)
				print(query_img_filename)
				if(os.path.exists(query_pc_filename)):
					print("pc exists")
				
				if(os.path.exists(query_img_filename)):
					print("img exists")
				
				if not os.path.exists("bad_case/%04d/query"%(RANDOM_TIME)):
					os.makedirs("bad_case/%04d/query"%(RANDOM_TIME))
				
				shutil.copy(query_pc_filename,"bad_case/%04d/query"%(RANDOM_TIME))
				shutil.copy(query_img_filename,"bad_case/%04d/query"%(RANDOM_TIME))
				
				df_locations = pd.read_csv(os.path.join(PC_PATH,SEQ_NAME[m],"pointcloud_locations_20m_20dis.csv"),sep=',')
				
				#copy retrived
				if not os.path.exists("bad_case/%04d/retrived"%(RANDOM_TIME)):
					os.makedirs("bad_case/%04d/retrived"%(RANDOM_TIME))
				
				for retrived in indices[0][0:threshold]:
					retrived_pc_filename = os.path.join(PC_PATH,SEQ_NAME[m],"pointclouds","%d.txt"%(df_locations['timestamp'][retrived]))
					retrived_img_filename = os.path.join(IMG_PATH,SEQ_NAME[m],"pointclouds","%d_stereo_centre.png"%(df_locations['timestamp'][retrived]))
					
					print(retrived_pc_filename)
					if os.path.exists(retrived_pc_filename):
						print("exists")
					
					print(retrived_img_filename)
					if os.path.exists(retrived_img_filename):
						print("exists")
					
					shutil.copy(retrived_pc_filename,"bad_case/%04d/retrived"%(RANDOM_TIME))
					shutil.copy(retrived_img_filename,"bad_case/%04d/retrived"%(RANDOM_TIME))
					
				#copy true_neighbors
				if not os.path.exists("bad_case/%04d/true_neighbors"%(RANDOM_TIME)):
					os.makedirs("bad_case/%04d/true_neighbors"%(RANDOM_TIME))
				
				for positive in true_neighbors:
					positive_pc_filename = os.path.join(PC_PATH,SEQ_NAME[m],"pointclouds","%d.txt"%(df_locations['timestamp'][positive]))
					positive_img_filename = os.path.join(IMG_PATH,SEQ_NAME[m],"pointclouds","%d_stereo_centre.png"%(df_locations['timestamp'][positive]))
					
					print(positive_pc_filename)
					if os.path.exists(positive_pc_filename):
						print("exists")
					
					print(positive_img_filename)
					if os.path.exists(positive_img_filename):
						print("exists")
					
					shutil.copy(positive_pc_filename,"bad_case/%04d/true_neighbors"%(RANDOM_TIME))
					shutil.copy(positive_img_filename,"bad_case/%04d/true_neighbors"%(RANDOM_TIME))
			'''
			
	if num_evaluated == 0:
		return -1,-1,-1
	one_percent_recall=(one_percent_retrieved/float(num_evaluated))*100
	recall=(np.cumsum(recall)/float(num_evaluated))*100
	print(recall)
	#print(np.mean(top1_similarity_score))
	print(one_percent_recall)
	return recall, top1_similarity_score, one_percent_recall 

def get_similarity(m, n):
	global DATABASE_VECTORS
	global QUERY_VECTORS

	database_output= DATABASE_VECTORS[m]
	queries_output= QUERY_VECTORS[n]

	threshold= len(queries_output)
	print(len(queries_output))
	database_nbrs = KDTree(database_output)

	similarity=[]
	for i in range(len(queries_output)):
		distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=1)
		for j in range(len(indices[0])):
			q_sim= np.dot(q_output[i], database_output[indices[0][j]])
			similarity.append(q_sim)
	average_similarity=np.mean(similarity)
	average_similarity= 1
	print(average_similarity)
	return average_similarity
	
	
def main():
	global SEARCH_FIELD
	global SEARCH_SUCCEED
	global RANDOM_TIME
	global NIGHT_FAILED
	
	
	for i in range(len(DATABASE_VECTORS)):
		print("database feature shape",i,DATABASE_VECTORS[i].shape)
	
	for i in range(len(QUERY_VECTORS)):
		print("query feature shape",i,QUERY_VECTORS[i].shape)
		
	#convert feature to target format
	recall= np.zeros(25)
	count = 0
	similarity=[]
	one_percent_recall=[]
	
	#compute recall
	for m in range(len(QUERY_SETS)):
		for n in range(len(QUERY_SETS)):
			if(m==n):
				continue
			pair_recall, pair_similarity, pair_opr = get_recall(m, n)
			if(pair_opr == -1):
				continue
			recall+=np.array(pair_recall)
			count+=1
			one_percent_recall.append(pair_opr)
			for x in pair_similarity:
				similarity.append(x)
			
	print()
	ave_recall=recall/count
	print(ave_recall)
	
	#print(similarity)
	average_similarity= np.mean(similarity)
	print(average_similarity)
	
	print("Average Similarity:\n")
	ave_one_percent_recall= np.mean(one_percent_recall)
	
	
	print("Average Top 1% Recall:\n")
	print(ave_one_percent_recall)
	
	print("SEARCH_FIELD cnt = \n")
	print(SEARCH_FIELD)
	
	print("SEARCH_SUCCEED cnt = \n")
	print(SEARCH_SUCCEED)
	
	print("RANDOM_TIME cnt = \n")
	print(RANDOM_TIME)
	
	print("NIGHT_FAILED cnt = \n")
	print(NIGHT_FAILED)
	
	#filename=RESULTS_FOLDER +'average_recall_oxford_netmax_sg(finetune_conv5).txt'
	with open(output_file, "w") as output:
		output.write("Average Recall @N:\n")
		output.write(str(ave_recall))
		output.write("\n\n")
		output.write("Average Similarity:\n")
		output.write(str(average_similarity))
		output.write("\n\n")
		output.write("Average Top 1% Recall:\n")
		output.write(str(ave_one_percent_recall))

if __name__ == "__main__":
	main()
	