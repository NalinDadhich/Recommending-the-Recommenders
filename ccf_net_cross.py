import tensorflow as tf
from dataio import data_reader
import math
from time import clock
import numpy as np


def grid_search(infile, logfile):
	# default params:
	params = {
		'cf_dim': 16,
		'user_attr_rank': 16,
		'item_attr_rank': 16,
		'layer_sizes': [16, 8],
		'lr': 0.1,
		'lamb': 0.001,
		'mu_1': 3.8,
		'mu_2': 3.7,
		'mu_1': 4.0,
		'mu_2': 4.0,
		'n_eopch': 100,
		'batch_size': 500,
		'init_value': 0.01
		}

	dataset = data_reader.movie_lens_data_repos(infile)
	wt = open(logfile, 'w')

	wt.write('best_train_rmse,best_test_rmse,best_eval_rmse,best_epoch,time,params\n')

	lambs = [0.001, 0.0001, 0.0005, 0.005]
	lrs = [0.1, 0.05, ]
	# layer_sizes_list = [[16], [16, 8]]
	init_values = [0.001, 0.01, 0.1]
	mu_1 = dataset.training_ratings_score_1.mean()
	mu_2 = dataset.training_ratings_score_2.mean()

	# params['mu_1'], params['mu_2'] = mu_1, mu_2
	# for lamb in lambs:
	#     for lr in lrs:
	#         for init_value in init_values:
	#             params['lamb'] = lamb
	#             params['lr'] = lr
	#             params['init_value'] =init_value
	#             run_with_parameters(dataset, params, wt)
	run_with_parameters(dataset, params, wt)
	wt.close()


def run_with_parameters(dataset, params, wt):

	start = clock()
	tf.reset_default_graph()
	best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx = single_run(dataset, params)
	end = clock()
	wt.write('%f,%f,%f,%d,%f,%s\n' % (
		best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx, (end-start)/60, str(params)))
	wt.flush()


def single_run(dataset, params):
	cf_dim, user_attr_rank, item_attr_rank, layer_sizes, lr, lamb, mu_1, mu_2, n_eopch, batch_size, \
		init_value = params['cf_dim'], params['user_attr_rank'], params['item_attr_rank'], params['layer_sizes'],\
		params['lr'], params['lamb'], params['mu_1'], params['mu_2'], params['n_eopch'], params['batch_size'], \
		params['init_value']

	# compose features from SVD
	user_cnt, user_attr_cnt = dataset.n_user, dataset.n_user_attr
	# print("training_ratings_item_1 ki lengtttttthhhhhhhhhhhhhhh::", dataset.training_ratings_item_1)
	# print("data")
	item_cnt_1, item_attr_cnt_1 = dataset.n_item_1, dataset.n_item_attr_1
	item_cnt_2, item_attr_cnt_2 = dataset.n_item_2, dataset.n_item_attr_2
	# item_cnt_2 = item_cnt_2 - item_cnt_1

	item_cnt_2 = item_cnt_2 - item_cnt_1
	# print("user_cnt: {}, user_attr_cnt: {}".format(user_cnt, user_attr_cnt))
	# print("item_cnt_1: {}, item_attr_cnt_1: {}".format(item_cnt_1, item_attr_cnt_1))
	# print("item_cnt_2: {},item_cnt_2: item_attr_cnt_2: {}".format(item_cnt_2, item_attr_cnt_2))
	W_user = tf.Variable(tf.truncated_normal([user_cnt, cf_dim], stddev=init_value/math.sqrt(float(cf_dim)), mean=0),
						 name='user_cf_embedding', dtype=tf.float32)
	# print("!!!!!!!!!")
	W_item_1 = tf.Variable(
		tf.truncated_normal([item_cnt_1, cf_dim], stddev=init_value/math.sqrt(float(cf_dim)), mean=0),
		name='item_cf_embedding_1', dtype=tf.float32)  # stddev:stand deviation
	# print("##########")
	W_item_2 = tf.Variable(
		tf.truncated_normal([item_cnt_2, cf_dim], stddev=init_value/math.sqrt(float(cf_dim)), mean=0),
		name='item_cf_embedding_2', dtype=tf.float32)
	# print("11111111111111111111111111111111")
	W_user_bias = tf.concat([W_user, tf.ones((user_cnt, 1), dtype=tf.float32)], axis=1, name='user_cf_embedding_bias')
	# print("222222222222222222222")
	W_item_bias_1 = tf.concat([tf.ones((item_cnt_1, 1), dtype=tf.float32), W_item_1],
							  axis=1, name='item_cf_embedding_bias_1')
	# print("33333333333333333333")
	W_item_bias_2 = tf.concat([tf.ones((item_cnt_2, 1), dtype=tf.float32), W_item_2],
							  axis=1, name='item_cf_embedding_bias_2')
	# print("4444444")
	# compose features from attributes
	user_attr_indices, user_attr_indices_values, user_attr_indices_weights = \
		compose_vector_for_sparse_tensor(dataset.user_attr)
	item_attr_indices_1, item_attr_indices_values_1, item_attr_indices_weights_1 = \
		compose_vector_for_sparse_tensor(dataset.item_attr_1)
	item_attr_indices_2, item_attr_indices_values_2, item_attr_indices_weights_2 = \
		compose_vector_for_sparse_tensor(dataset.item_attr_2)
	# print("22")

	# print(user_attr_indices, user_attr_indices_values, user_attr_indices_weights)
	user_sp_ids = tf.SparseTensor(indices=user_attr_indices, values=user_attr_indices_values,
								  dense_shape=[user_cnt, user_attr_cnt])
	# print("11")
	# user_sp_ids = tf.Print(user_sp_ids,[user_sp_ids])
	user_sp_ids = tf.SparseTensor(indices=user_attr_indices, values=user_attr_indices_values,
								  dense_shape=[user_cnt, user_attr_cnt])
	# print("11")
	user_sp_weights = tf.SparseTensor(indices=user_attr_indices, values=user_attr_indices_weights,
									  dense_shape=[user_cnt, user_attr_cnt])
	# print("00")
# 	print(item_attr_indices_1)
	item_sp_ids_1 = tf.SparseTensor(indices=item_attr_indices_1, values=item_attr_indices_values_1,
									dense_shape=[item_cnt_1, item_attr_cnt_1])
	# print("5555555")
	# print(item_attr_indices_2)
	item_sp_ids_2 = tf.SparseTensor(indices=item_attr_indices_2, values=item_attr_indices_values_2,
									dense_shape=[item_cnt_2, item_attr_cnt_2])
	# print("6666666")
	item_sp_weights_1 = tf.SparseTensor(indices=item_attr_indices_1, values=item_attr_indices_weights_1,
										dense_shape=[item_cnt_1, item_attr_cnt_1])
	item_sp_weights_2 = tf.SparseTensor(indices=item_attr_indices_2, values=item_attr_indices_weights_2,
										dense_shape=[item_cnt_2, item_attr_cnt_2])

	W_user_attr = tf.Variable(tf.truncated_normal(
		[user_attr_cnt, user_attr_rank], stddev=init_value/math.sqrt(float(user_attr_rank)), mean=0),
		name='user_attr_embedding', dtype=tf.float32)

	W_item_attr_1 = tf.Variable(
		tf.truncated_normal([item_attr_cnt_1, item_attr_rank], stddev=init_value/math.sqrt(float(item_attr_rank)),
							mean=0), name='item_attr_embedding_1', dtype=tf.float32)
	W_item_attr_2 = tf.Variable(
		tf.truncated_normal([item_attr_cnt_2, item_attr_rank], stddev=init_value / math.sqrt(float(item_attr_rank)),
							mean=0), name='item_attr_embedding_2', dtype=tf.float32)

	user_embeddings = tf.nn.embedding_lookup_sparse(W_user_attr, user_sp_ids, user_sp_weights,
													name='user_embeddings', combiner='sum')
	item_embeddings_1 = tf.nn.embedding_lookup_sparse(W_item_attr_1, item_sp_ids_1, item_sp_weights_1,
													  name='item_embeddings_1', combiner='sum')
	item_embeddings_2 = tf.nn.embedding_lookup_sparse(W_item_attr_2, item_sp_ids_2, item_sp_weights_2,
													  name='item_embeddings_2', combiner='sum')

	user_indices_1 = tf.placeholder(tf.int32, [None])
	user_indices_2 = tf.placeholder(tf.int32, [None])
	item_indices_1 = tf.placeholder(tf.int32, [None])
	item_indices_2 = tf.placeholder(tf.int32, [None])
	ratings_1 = tf.placeholder(tf.float32, [None])
	ratings_2 = tf.placeholder(tf.float32, [None])

	user_cf_feature_1 = tf.nn.embedding_lookup(W_user_bias, user_indices_1, name='user_cf_feature_1')
	user_cf_feature_2 = tf.nn.embedding_lookup(W_user_bias, user_indices_2, name='user_cf_feature_2')
	item_cf_feature_1 = tf.nn.embedding_lookup(W_item_bias_1, item_indices_1, name='item_cf_feature_1')
	item_cf_feature_2 = tf.nn.embedding_lookup(W_item_bias_2, item_indices_2, name='item_cf_feature_2')

	user_attr_feature_1 = tf.nn.embedding_lookup(user_embeddings, user_indices_1, name='user_attr_feature_1')
	user_attr_feature_2 = tf.nn.embedding_lookup(user_embeddings, user_indices_2, name='user_attr_feature_2')

	item_attr_feature_1 = tf.nn.embedding_lookup(item_embeddings_1, item_indices_1, name='item_attr_feature_1')
	item_attr_feature_2 = tf.nn.embedding_lookup(item_embeddings_2, item_indices_2, name='item_attr_feature_2')

	train_step, square_error, loss, merged_summary = \
		build_model(user_cf_feature_1, user_cf_feature_2, user_attr_feature_1, user_attr_feature_2, user_attr_rank,
					item_cf_feature_1, item_cf_feature_2, item_attr_feature_1, item_attr_feature_2, item_attr_rank,
					ratings_1, ratings_2, layer_sizes, W_user, W_item_1, W_item_2,
					W_user_attr, W_item_attr_1, W_item_attr_2,
					lamb, lr, mu_1, mu_2)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	# print(sess.run(user_embeddings))

	train_writer = tf.summary.FileWriter(r'debug_logs', sess.graph)

	n_instances = min(dataset.training_ratings_user_1.shape[0], dataset.training_ratings_user_2.shape[0])

	best_train_rmse, best_test_rmse, best_eval_rmse = -1, -1, -1

	best_eopch_idx = -1

	for ite in range(n_eopch):
		start = clock()
		for i in range(n_instances//batch_size):
			start_idx = i * batch_size
			end_idx = start_idx + batch_size
			cur_user_1, cur_user_2 = \
				dataset.training_ratings_user_1[start_idx:end_idx], \
				dataset.training_ratings_user_2[start_idx:end_idx]
			cur_item_1, cur_score_1, cur_item_2, cur_score_2 = \
				dataset.training_ratings_item_1[start_idx:end_idx], \
				dataset.training_ratings_score_1[start_idx:end_idx], \
				dataset.training_ratings_item_2[start_idx:end_idx], \
				dataset.training_ratings_score_2[start_idx:end_idx]

			sess.run(train_step,
					 {user_indices_1: cur_user_1, user_indices_2: cur_user_2, item_indices_1: cur_item_1,
					  item_indices_2: cur_item_2, ratings_1: cur_score_1, ratings_2: cur_score_2})

		error_traing = sess.run(square_error,
								{user_indices_1: dataset.training_ratings_user_1,
								 user_indices_2: dataset.training_ratings_user_2,
								 item_indices_1: dataset.training_ratings_item_1,
								 item_indices_2: dataset.training_ratings_item_2,
								 ratings_1: dataset.training_ratings_score_1,
								 ratings_2: dataset.training_ratings_score_2})

		error_test = sess.run(square_error,
		                      {user_indices_1: dataset.testing_ratings_user_1,
		                       user_indices_2: dataset.testing_ratings_user_2,
		                       item_indices_1: dataset.testing_ratings_item_1,
		                       item_indices_2: dataset.testing_ratings_item_2,
		                       ratings_1: dataset.testing_ratings_score_1,
		                       ratings_2: dataset.testing_ratings_score_2})

		# error_eval = sess.run(square_error,
		#                       {user_indices_1: dataset.eval_ratings_user_1,
		#                        user_indices_2: dataset.eval_ratings_user_2,
		#                        item_indices_1: dataset.eval_ratings_item_1,
		#                        item_indices_2: dataset.eval_ratings_item_2,
		#                        ratings_1: dataset.eval_ratings_score_1,
		#                        ratings_2: dataset.eval_ratings_score_2})
		# error_test = 0
		error_eval = 0

		loss_traing = sess.run(loss,
							   {user_indices_1: dataset.training_ratings_user_1,
								user_indices_2: dataset.training_ratings_user_2,
								item_indices_1: dataset.training_ratings_item_1,
								item_indices_2: dataset.training_ratings_item_2,
								ratings_1: dataset.training_ratings_score_1,
								ratings_2: dataset.training_ratings_score_2})

		# summary = sess.run(merged_summary,
		# 				   {user_indices_1: dataset.training_ratings_user_1,
		# 					user_indices_2: dataset.training_ratings_user_2,
		# 					item_indices_1: dataset.training_ratings_item_1,
		# 					item_indices_2: dataset.training_ratings_item_2,
		# 					ratings_1: dataset.training_ratings_score_1,
		# 					ratings_2: dataset.training_ratings_score_2})

		# train_writer.add_summary(summary, ite)

		# error_traing =0 
		# loss_traing =0
		end = clock()
		print("Iteration %d  RMSE(train): %f  RMSE(test): %f   RMSE(eval): %f   LOSS(train): %f  minutes: %f" %
			  (ite, error_traing, error_test , error_eval, loss_traing, (end-start)/60))

		#if best_test_rmse < 0 or best_test_rmse > error_test:
			#best_train_rmse, best_test_rmse, best_eval_rmse = error_traing, error_test, error_eval
			#best_eopch_idx = ite
		#else:
			#if ite - best_eopch_idx > 10:
				#break

	train_writer.close()
	sess.close()
	return best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx



def build_model(user_cf_feature_1, user_cf_feature_2, user_attr_feature_1, user_attr_feature_2, user_attr_rank,
				item_cf_feature_1, item_cf_feature_2, item_attr_feature_1, item_attr_feature_2, item_attr_rank,
				ratings_1, ratings_2, layer_size,
				W_user, W_item_1, W_item_2,
				W_user_attr, W_item_attr_1, W_item_attr_2,
				lamb, lr, mu_1, mu_2):
	
	print("*************************************************************************************")
	layer_cnt = len(layer_size)

	hiddens_user_1 = []
	hiddens_item_1 = []

	hiddens_user_1.append(user_attr_feature_1)
	hiddens_item_1.append(item_attr_feature_1)

	W_user_list = []

	W_item_list_1 = []
	W_item_list_2 = []

	hiddens_user_2 = []
	hiddens_item_2 = []

	hiddens_user_2.append(user_attr_feature_2)
	hiddens_item_2.append(item_attr_feature_2)

	b_user_list = []

	b_item_list_1 = []
	b_item_list_2 = []

	for i in range(layer_cnt):
		with tf.name_scope('layer_'+str(i)):
			b_user_list.append(tf.Variable(tf.truncated_normal([layer_size[i]]), name='user_bias'))
			b_item_list_1.append(tf.Variable(tf.truncated_normal([layer_size[i]]), name='item_bias_1'))
			b_item_list_2.append(tf.Variable(tf.truncated_normal([layer_size[i]]), name='item_bias_2'))

			if i == 0:
				W_user_list.append(tf.Variable(
						tf.truncated_normal([user_attr_rank, layer_size[i]],
											stddev=1/math.sqrt(float(layer_size[i])), mean=0), name='W_user'))
				W_item_list_1.append(tf.Variable(
						tf.truncated_normal([item_attr_rank, layer_size[i]],
											stddev=1/math.sqrt(float(layer_size[i])), mean=0), name='W_item_1'))

				W_item_list_2.append(tf.Variable(
					tf.truncated_normal([item_attr_rank, layer_size[i]],
										stddev=1 / math.sqrt(float(layer_size[i])), mean=0), name='W_item_2'))

				user_middle_1 = tf.matmul(user_attr_feature_1, W_user_list[i]) + b_user_list[i]
				item_middle_1 = tf.matmul(item_attr_feature_1, W_item_list_1[i]) + b_item_list_1[i]

				user_middle_2 = tf.matmul(user_attr_feature_2, W_user_list[i]) + b_user_list[i]
				item_middle_2 = tf.matmul(item_attr_feature_2, W_item_list_2[i]) + b_item_list_2[i]

			else:
				W_user_list.append(tf.Variable(
					tf.truncated_normal([layer_size[i-1], layer_size[i]],
										stddev=1/math.sqrt(float(layer_size[i])), mean=0), name='W_user'))
				W_item_list_1.append(tf.Variable(
					tf.truncated_normal([layer_size[i - 1], layer_size[i]],
										stddev=1/math.sqrt(float(layer_size[i])), mean=0), name='W_item_1'))
				W_item_list_2.append(tf.Variable(
					tf.truncated_normal([layer_size[i - 1], layer_size[i]],
										stddev=1 / math.sqrt(float(layer_size[i])), mean=0), name='W_item_2'))

				user_middle_1 = tf.matmul(hiddens_user_1[i], W_user_list[i]) + b_user_list[i]
				item_middle_1 = tf.matmul(hiddens_item_1[i], W_item_list_1[i]) + b_item_list_1[i]

				user_middle_2 = tf.matmul(hiddens_user_2[i], W_user_list[i]) + b_user_list[i]
				item_middle_2 = tf.matmul(hiddens_item_2[i], W_item_list_2[i]) + b_item_list_2[i]

			hiddens_user_1.append(tf.identity(user_middle_1, name='factor_user_1'))  # identity ,sigmoid
			hiddens_item_1.append(tf.identity(item_middle_1, name='factor_item_1'))

			hiddens_user_2.append(tf.identity(user_middle_2, name='factor_user_2'))  # identity ,sigmoid
			hiddens_item_2.append(tf.identity(item_middle_2, name='factor_item_2'))

	factor_user_1 = hiddens_user_1[layer_cnt]
	factor_item_1 = hiddens_item_1[layer_cnt]

	factor_user_2 = hiddens_user_2[layer_cnt]
	factor_item_2 = hiddens_item_2[layer_cnt]

	k=1
	preds_1 = k*(tf.nn.l2_normalize(tf.reduce_sum(tf.multiply(user_cf_feature_1, item_cf_feature_1), 1) +
			   tf.reduce_sum(tf.multiply(factor_user_1, factor_item_1), 1))+0.8)
	# print(preds_1.shape)
	preds_2 = k*(tf.nn.l2_normalize(tf.reduce_sum(tf.multiply(user_cf_feature_2, item_cf_feature_2), 1) +
			   tf.reduce_sum(tf.multiply(factor_user_2, factor_item_2), 1))+1.0)
	preds_1 = (tf.reduce_sum(tf.multiply(user_cf_feature_1, item_cf_feature_1), 1) +
			   tf.reduce_sum(tf.multiply(factor_user_1, factor_item_1), 1)) + mu_1

	preds_2 = (tf.reduce_sum(tf.multiply(user_cf_feature_2, item_cf_feature_2), 1) +
			   tf.reduce_sum(tf.multiply(factor_user_2, factor_item_2), 1)) + mu_2

	temp1 = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds_1, ratings_1)))
	# temp1 = tf.Print(temp1,[temp1],message="predicted raring on first domain")

	temp2 = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds_2, ratings_2)))
	# temp2 = tf.Print(temp2,[temp2],message="predicted ratings on second domain")
	
	# print(temp1.get_shape().as_list())
	# print(temp2.get_shape().as_list())
	square_error = 0.5*temp1 + 0.5*temp2
	# print("preds_1: {}, ratings_1: {}, preds_2:{}, ratings_2: {}".format(preds_1, ratings_1, preds_2, ratings_2))
	loss = square_error

	# loss = tf.Print(loss,[loss],message="Loss##########################################################3")
	square_error = 0.5*tf.sqrt(tf.reduce_mean(tf.squared_difference(preds_1, ratings_1))) + 0.5*tf.sqrt(tf.reduce_mean(tf.squared_difference(preds_2, ratings_2)))
	# print("preds_1: {}, ratings_1: {}, preds_2:{}, ratings_2: {}".format(preds_1, ratings_1, preds_2, ratings_2))
	loss = square_error

	loss = tf.Print(loss,[loss],message="Loss##########################################################3")
	# print("Loss=",loss)
	for i in range(layer_cnt):
		loss = loss + lamb*(
							tf.reduce_mean(tf.nn.l2_loss(W_user)) +
							tf.reduce_mean(tf.nn.l2_loss(W_item_1)) +
							tf.reduce_mean(tf.nn.l2_loss(W_item_2)) +
							tf.reduce_mean(tf.nn.l2_loss(W_user_attr)) +
							tf.reduce_mean(tf.nn.l2_loss(W_item_attr_1)) +
							tf.reduce_mean(tf.nn.l2_loss(W_item_attr_2)) +
							tf.reduce_mean(tf.nn.l2_loss(W_user_list[i])) +
							tf.reduce_mean(tf.nn.l2_loss(W_item_list_1[i])) +
							tf.reduce_mean(tf.nn.l2_loss(b_user_list[i])) +
							tf.reduce_mean(tf.nn.l2_loss(b_item_list_1[i])) +
							tf.reduce_mean(tf.nn.l2_loss(W_item_list_2[i])) +
							tf.reduce_mean(tf.nn.l2_loss(b_item_list_2[i]))
						)

	tf.summary.scalar('square_error', square_error)
	tf.summary.scalar('loss', loss)

	merged_summary = tf.summary.merge_all()
	train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

	return train_step, square_error, loss, merged_summary


def compose_vector_for_sparse_tensor(entity2attr_list):
	indices = []
	indices_values = []  # feature indice
	weight_values = []

	N = len(entity2attr_list)
	for i in range(N):
		if len(entity2attr_list[i]) > 0:
			cnt = 0
			for attr_pair in entity2attr_list[i]:
				indices.append([i, cnt])
				indices_values.append(attr_pair[0])
				weight_values.append(attr_pair[1])
				cnt += 1
		else:
			indices.append([i, 0])
			indices_values.append(0)
			weight_values.append(0)
	return indices, indices_values, weight_values


if __name__ == '__main__':
	grid_search(r'cross_movielens_100k.pkl', r'debug_logs/CCFNet_movielens100k.csv')