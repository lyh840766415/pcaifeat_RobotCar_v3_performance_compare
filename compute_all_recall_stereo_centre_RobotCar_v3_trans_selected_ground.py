import numpy as np
import pickle
from loading_input_v3 import *
from sklearn.neighbors import KDTree

def get_queries_dict(filename):
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("feature Loaded.")
		return queries

DATABASE_PCAI_VECTORS_FILENAME = "database_pcai_feat_00441147.pickle"
QUERY_PCAI_VECTORS_FILENAME = "query_pcai_feat_00441147.pickle"

DATABASE_PC_VECTORS_FILENAME = "database_pc_feat_00441147.pickle"
QUERY_PC_VECTORS_FILENAME = "query_pc_feat_00441147.pickle"

DATABASE_IMG_VECTORS_FILENAME = "database_img_feat_00441147.pickle"
QUERY_IMG_VECTORS_FILENAME = "query_img_feat_00441147.pickle"
#result output
output_file = "result_img_trans_mono_left_00240080.txt"
#load feature
DATABASE_PCAI_VECTORS = get_queries_dict(DATABASE_PCAI_VECTORS_FILENAME)
QUERY_PCAI_VECTORS = get_queries_dict(QUERY_PCAI_VECTORS_FILENAME)

DATABASE_PC_VECTORS = get_queries_dict(DATABASE_PC_VECTORS_FILENAME)
QUERY_PC_VECTORS = get_queries_dict(QUERY_PC_VECTORS_FILENAME)

DATABASE_IMG_VECTORS = get_queries_dict(DATABASE_IMG_VECTORS_FILENAME)
QUERY_IMG_VECTORS = get_queries_dict(QUERY_IMG_VECTORS_FILENAME)

#load label
QUERY_FILE= 'generate_queries_v3/stereo_centre_trans_RobotCar_ground_selected_oxford_evaluation_query.pickle'
QUERY_SETS= get_sets_dict(QUERY_FILE)

for m in range(len(QUERY_SETS)):
	print(len(QUERY_PCAI_VECTORS[m]))
	print(len(QUERY_SETS[m]))
	if len(QUERY_SETS[m]) != len(QUERY_PCAI_VECTORS[m]):
		print("not equal")


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS):
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
	one_percent_recall_data = [0]*len(queries_output)
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
			one_percent_recall_data[i]=1
			
			
	if num_evaluated == 0:
		return None,None,None,None
	one_percent_recall=(one_percent_retrieved/float(num_evaluated))*100
	recall=(np.cumsum(recall)/float(num_evaluated))*100
	print(recall)
	#print(np.mean(top1_similarity_score))
	print(one_percent_recall)
	return recall, top1_similarity_score, one_percent_recall,one_percent_recall_data
	
def main():
	for i in range(len(DATABASE_PCAI_VECTORS)):
		print("database feature shape",i,DATABASE_PCAI_VECTORS[i].shape)
	
	for i in range(len(QUERY_PCAI_VECTORS)):
		print("query feature shape",i,QUERY_PCAI_VECTORS[i].shape)
		
	#convert feature to target format
	recall_pcai= np.zeros(25)
	recall_pc= np.zeros(25)
	recall_img= np.zeros(25)
	count = 0
	one_percent_recall_pcai=[]
	one_percent_recall_pc=[]
	one_percent_recall_img=[]
	
	
	#compute recall
	for m in range(len(QUERY_SETS)):
		for n in range(len(QUERY_SETS)):
			if(m==n):
				continue
			pair_recall_pcai, pair_similarity_pcai, pair_opr_pcai, one_per_data_pcai = get_recall(m, n, DATABASE_PCAI_VECTORS, QUERY_PCAI_VECTORS)
			pair_recall_pc, pair_similarity_pc, pair_opr_pc, one_per_data_pc = get_recall(m, n, DATABASE_PC_VECTORS, QUERY_PC_VECTORS)
			pair_recall_img, pair_similarity_img, pair_opr_img, one_per_data_img = get_recall(m, n, DATABASE_IMG_VECTORS, QUERY_IMG_VECTORS)
			
			if(pair_recall_pcai is None):
				continue
				
			recall_pcai+=np.array(pair_recall_pcai)
			recall_pc+=np.array(pair_recall_pc)
			recall_img+=np.array(pair_recall_img)
			
			count+=1
			one_percent_recall_pcai.append(pair_opr_pcai)
			one_percent_recall_pc.append(pair_opr_pc)
			one_percent_recall_img.append(pair_opr_img)

	recall_pcai=recall_pcai/count
	recall_pc=recall_pc/count
	recall_img=recall_img/count
	
	print(recall_pcai)
	print(recall_pc)
	print(recall_img)
	
	one_percent_recall_pcai= np.mean(one_percent_recall_pcai)
	one_percent_recall_pc= np.mean(one_percent_recall_pc)
	one_percent_recall_img= np.mean(one_percent_recall_img)
	
	print(one_percent_recall_pcai)
	print(one_percent_recall_pc)
	print(one_percent_recall_img)

if __name__ == "__main__":
	main()
	