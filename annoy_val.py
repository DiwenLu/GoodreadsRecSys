#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Usage:
	$ spark-submit annoy_val.py dirname rank regParam k random_seed nns
'''

import sys
import os
import time
import datetime
from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import udf, col, explode, struct, collect_set, collect_list
from pyspark.mllib.evaluation import RankingMetrics



def annoy_val(spark, dirname, rank, regParam, k, random_seed, nns):
	
	# Load validation data, trained ALS model, and annoy index map
	val_data = spark.read.parquet(f'{dirname}/val.parquet')
	als_model = ALSModel.load(f'{dirname}/{rank}_{regParam}_model')
	annoy_index_map = spark.read.parquet(f'{dirname}_{rank}_{regParam}_annoy_index_map.parquet')
	tree_ann_path = f'{os.getcwd()}/{dirname}_{rank}_{regParam}_tree.ann'

	val_user = val_data.select('user_id').distinct()
	user_factors = als_model.userFactors.withColumnRenamed('id', 'user_id')
	val_user_factors = val_user.join(user_factors, on='user_id', how='inner')
	
	# Build candidate set
	start_time = time.time()

	@udf(returnType=ArrayType(IntegerType()))
	def find_candidates_udf(u_factor):
		from annoy import AnnoyIndex  # must import here !
		u = AnnoyIndex(rank, 'dot')
		u.set_seed(random_seed)
		u.load(SparkFiles.get(tree_ann_path)) # tree_ann_path must be absolute path
		return u.get_nns_by_vector(u_factor, n = nns, search_k=-1, include_distances=False)

	val_user_candidates = val_user_factors.select('user_id', find_candidates_udf(col('features')).alias('candidates'))
	val_user_candidates = val_user_candidates.select('user_id', explode('candidates').alias('annoy_id'))	
	val_user_candidates = val_user_candidates.join(annoy_index_map, on='annoy_id', how='inner')
	val_user_candidates = val_user_candidates.select('user_id', col('id').alias('book_id'))

	# Predict ratings for candidate set
	pred = als_model.transform(val_user_candidates)
	pred = pred.filter(col('prediction') >= 3.0).select('user_id', struct('book_id', 'prediction').alias('pred'))
	pred = pred.groupBy('user_id').agg(collect_list('pred').alias('pred'))

	# Make top k recommendations

	@udf(returnType=ArrayType(IntegerType()))
	def top_k_udf(l):
		res = sorted(l, key=lambda l: l[1], reverse=True)
		return [res[i][0] for i in range(min(k, len(res)))]

	rec = pred.select('user_id', top_k_udf(col('pred')).alias('recommendations'))
	
	# Get ground-truth items
	ground_truth = val_data.filter(col('rating') >= 3.0).groupBy('user_id').agg(collect_set('book_id').alias('true_books'))

	# Build ranking metrics
	prediction_and_labels = rec.join(ground_truth, on='user_id', how='inner').drop('user_id').rdd
	metrics = RankingMetrics(prediction_and_labels)

	precisionAtK = metrics.precisionAt(k)
	mAP = metrics.meanAveragePrecision
	
	end_time = time.time()
	
	print(f'Total evaluation over {dirname}/{rank}_{regParam}_model using Annoy takes {str(datetime.timedelta(seconds = end_time - start_time))}')
	print(f'p@{k} = {precisionAtK}')
	print(f'mAP = {mAP}')


if __name__ == '__main__':
	spark = SparkSession.builder \
						.appName('annoy_val') \
						.master('yarn') \
						.config('spark.executor.memory', '5g') \
						.config('spark.driver.memory', '5g') \
						.getOrCreate()
	
	dirname = sys.argv[1]
	rank = int(sys.argv[2])
	regParam = float(sys.argv[3])
	k = int(sys.argv[4])
	random_seed = int(sys.argv[5])
	nns = int(sys.argv[6])

	annoy_val(spark, dirname, rank, regParam, k, random_seed, nns)
