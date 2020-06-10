from tensorflow.python import pywrap_tensorflow

checkpoint_path = "/data/lyh/lab/pcaifeat_RobotCar_v3_performance_compare/log/train_save_trans_exp_4_15/img_model_00441147.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
print(reader.debug_string().decode("utf-8"))
exit()
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
	print("tensor_name: ", key)