from loading_input_v3 import *
TRAIN_FILE = 'generate_queries_v3/training_queries_RobotCar_trans_no_ground.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)

print(len(TRAINING_QUERIES.keys()))

TRAIN_FILE = 'generate_queries_v3/training_queries_RobotCar_trans_ground.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
print(len(TRAINING_QUERIES.keys()))