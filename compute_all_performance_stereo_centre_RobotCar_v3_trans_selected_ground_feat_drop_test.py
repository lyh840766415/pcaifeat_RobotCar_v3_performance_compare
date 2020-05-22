import argparse
import numpy as np
import pickle
from loading_input_v3 import *
from sklearn.neighbors import KDTree

def get_queries_dict(filename):
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("feature Loaded.")
		return queries
		
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--drop_feat", type=int, default=0)
args = parser.parse_args()
print(args.drop_feat)

DATABASE_PCAI_VECTORS_FILENAME = "database_pcai_feat_00219073.pickle"
QUERY_PCAI_VECTORS_FILENAME = "query_pcai_feat_00219073.pickle"

DATABASE_PC_VECTORS_FILENAME = "database_pc_feat_00219073.pickle"
QUERY_PC_VECTORS_FILENAME = "query_pc_feat_00219073.pickle"

DATABASE_IMG_VECTORS_FILENAME = "database_img_feat_00219073.pickle"
QUERY_IMG_VECTORS_FILENAME = "query_img_feat_00219073.pickle"
#result output
output_file = "result_img_trans_mono_left_00240080.txt"

'''
pc_reweight = np.loadtxt("inv_pc_reweight.txt")
img_reweight = np.loadtxt("inv_img_reweight.txt")
img_pc_reweight = np.concatenate((pc_reweight,img_reweight))
print(img_pc_reweight.shape)
'''

img_reweight = 1*np.ones(2048)
pc_reweight = 1*np.ones(1000)
img_pc_reweight = np.concatenate((pc_reweight,img_reweight))
img_pc_reweight[args.drop_feat]=0
print(img_pc_reweight.shape)
print(img_pc_reweight[args.drop_feat])

#load feature
DATABASE_PCAI_VECTORS = get_queries_dict(DATABASE_PCAI_VECTORS_FILENAME)
QUERY_PCAI_VECTORS = get_queries_dict(QUERY_PCAI_VECTORS_FILENAME)
for i in range(len(DATABASE_PCAI_VECTORS)):
	reweight = np.tile(img_pc_reweight,(DATABASE_PCAI_VECTORS[i].shape[0],1))
	DATABASE_PCAI_VECTORS[i] = DATABASE_PCAI_VECTORS[i]*reweight
for i in range(len(QUERY_PCAI_VECTORS)):
	reweight = np.tile(img_pc_reweight,(QUERY_PCAI_VECTORS[i].shape[0],1))
	QUERY_PCAI_VECTORS[i] = QUERY_PCAI_VECTORS[i]*reweight

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
		return None,None,None,None,None
	one_percent_recall=(one_percent_retrieved/float(num_evaluated))*100
	recall=(np.cumsum(recall)/float(num_evaluated))*100
	print(recall)
	#print(np.mean(top1_similarity_score))
	print(one_percent_recall)
	one_percent_recall_data = np.array(one_percent_recall_data)
	return recall, top1_similarity_score, one_percent_recall,one_percent_recall_data,num_evaluated

def intersect(a,b):
	c = a + b
	c[c<=1] = 0
	c[c>1] = 1
	return c

def combine(a,b):
	c = a + b
	c[c<=0] = 0
	c[c>0] = 1
	return c

def diff(a,b):
	c = a - b
	c[c>0] = 1
	c[c<=0] =0
	return c

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
	img_intersect_fusion = []
	pc_intersect_fusion = []
	pc_combine_img = []
	pc_intersect_img=[]
	img_diff_pc = []
	pc_diff_img = []
	img_combine_pc_diff_fusion = []
	fusion_diff_img_combine_pc = []
	
	
	#compute recall
	for m in range(len(QUERY_SETS)):
		for n in range(len(QUERY_SETS)):
			if(m==n):
				continue
			pair_recall_pcai, pair_similarity_pcai, pair_opr_pcai, one_per_data_pcai, num_evaluated_pcai = get_recall(m, n, DATABASE_PCAI_VECTORS, QUERY_PCAI_VECTORS)
			
			if(pair_recall_pcai is None):
				continue
				
			recall_pcai+=np.array(pair_recall_pcai)
			
			count+=1
			one_percent_recall_pcai.append(pair_opr_pcai)

	
	one_percent_recall_pcai= np.mean(one_percent_recall_pcai)
	print(args.drop_feat,one_percent_recall_pcai)
	
	file = 'drop_feat_test.txt'
	with open(file, 'a+') as f:
		f.write(str(args.drop_feat)+' '+str(one_percent_recall_pcai)+'\n')


if __name__ == "__main__":
	main()
	